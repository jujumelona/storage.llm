#include "moe_pc_engine.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
using socket_handle_t = SOCKET;
static const socket_handle_t invalid_socket_handle = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_handle_t = int;
static const socket_handle_t invalid_socket_handle = -1;
#endif

#include "../../byte_size.h"
#include "../../json_request.h"
#include "../../system_memory.h"
#include "../../loader/wide_path.h"

static volatile std::sig_atomic_t g_server_shutdown_requested = 0;

static void handle_server_shutdown_signal(int) {
    g_server_shutdown_requested = 1;
}

struct server_options {
    bool allow_remote = false;
    std::string host = "127.0.0.1";
    uint16_t port = 8000;
    std::string model_id = "storagellm-offload";
    bool model_id_explicit = false;
    std::string topology_path;
    std::string table_path;
    std::string model_root;
    std::string scale4_path;
    std::string openclaw_config_path;
    bool openclaw_config_only = false;
    bool ram_budget_override = false;
    bool vram_budget_override = false;
    moe_execution_policy_t policy = moe_EXECUTION_PERFORMANCE;
    moe_pc_engine_config_t engine_config = moe_pc_default_config();
};

struct http_request {
    std::string method;
    std::string target;
    std::string path;
    std::string body;
};

struct server_runtime_state {
    std::atomic<moe_pc_engine_t*> engine{nullptr};
    std::atomic<int> model_ready{0};
    std::atomic<int> model_failed{0};
    std::atomic<int> shutdown_requested{0};
    moe_backend_caps_t initial_caps{};
    mutable std::mutex error_mutex;
    std::string error_message;
    mutable std::mutex stage_mutex;
    std::string load_stage{"not_started"};
    mutable std::mutex root_check_mutex;
    std::string root_check_model_root;
    moe_model_root_check_t root_check{};
    int root_check_ready = 0;
};

#include "pc_server_runtime_config.inc"
#include "pc_server_prefetch.inc"

static void close_socket_handle(socket_handle_t s) {
    if (s == invalid_socket_handle) {
        return;
    }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

static bool network_startup() {
#ifdef _WIN32
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
    return true;
#endif
}

static void network_cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

static bool is_loopback_host(const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

static std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (char ch : value) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(ch));
                    out += buf;
                } else {
                    out += ch;
                }
                break;
        }
    }
    return out;
}

static int64_t unix_time_seconds() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

static std::string lower_ascii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static bool request_wants_stream(const std::string& body) {
    std::string compact;
    compact.reserve(body.size());
    for (char ch : body) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            compact.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return compact.find("\"stream\":true") != std::string::npos;
}

static void append_utf8(std::string* out, uint32_t cp) {
    if (!out) return;
    if (cp <= 0x7Fu) {
        out->push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FFu) {
        out->push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else if (cp <= 0xFFFFu) {
        out->push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    } else {
        out->push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
        out->push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
    }
}

static int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
    return -1;
}

static bool parse_json_string_at(const std::string& text, size_t quote, std::string* out, size_t* next_pos) {
    if (quote >= text.size() || text[quote] != '"' || !out) {
        return false;
    }
    out->clear();
    for (size_t i = quote + 1; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '"') {
            if (next_pos) *next_pos = i + 1;
            return true;
        }
        if (ch != '\\') {
            out->push_back(ch);
            continue;
        }
        if (++i >= text.size()) {
            return false;
        }
        const char esc = text[i];
        switch (esc) {
            case '"': out->push_back('"'); break;
            case '\\': out->push_back('\\'); break;
            case '/': out->push_back('/'); break;
            case 'b': out->push_back('\b'); break;
            case 'f': out->push_back('\f'); break;
            case 'n': out->push_back('\n'); break;
            case 'r': out->push_back('\r'); break;
            case 't': out->push_back('\t'); break;
            case 'u': {
                if (i + 4 >= text.size()) return false;
                uint32_t cp = 0;
                for (int k = 0; k < 4; ++k) {
                    const int hv = hex_value(text[i + 1 + k]);
                    if (hv < 0) return false;
                    cp = (cp << 4) | (uint32_t)hv;
                }
                i += 4;
                append_utf8(out, cp);
                break;
            }
            default:
                out->push_back(esc);
                break;
        }
    }
    return false;
}

static bool json_read_string_value(const std::string& body, const char* key, std::string* out) {
    if (!key || !out) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = body.find(needle);
    const size_t colon = pos == std::string::npos ? pos : body.find(':', pos + needle.size());
    const size_t quote = colon == std::string::npos ? colon : body.find('"', colon + 1);
    return quote != std::string::npos && parse_json_string_at(body, quote, out, nullptr);
}

static std::vector<std::string> json_collect_string_values(const std::string& body, const char* key) {
    std::vector<std::string> values;
    if (!key) return values;
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = 0;
    while ((pos = body.find(needle, pos)) != std::string::npos) {
        const size_t colon = body.find(':', pos + needle.size());
        const size_t quote = colon == std::string::npos ? colon : body.find('"', colon + 1);
        std::string value;
        size_t next = quote;
        if (quote != std::string::npos && parse_json_string_at(body, quote, &value, &next)) {
            values.push_back(std::move(value));
            pos = next;
        } else {
            pos += needle.size();
        }
    }
    return values;
}

static std::string join_model_file(const std::string& root, const char* name) {
    if (root.empty() || !name || !name[0]) return "";
    const char sep =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    if (root.back() == '\\' || root.back() == '/') {
        return root + name;
    }
    return root + sep + name;
}

static bool read_text_file(const std::string& path, std::string* out) {
    if (!out || path.empty()) return false;
    std::ifstream input(path, std::ios::binary);
#ifdef _WIN32
    if (!input) {
        const std::wstring wide = storagellm::wide_path_from_utf8(path);
        if (!wide.empty()) {
            input.open(wide.c_str(), std::ios::binary);
        }
    }
#endif
    if (!input) return false;
    std::ostringstream ss;
    ss << input.rdbuf();
    *out = ss.str();
    return true;
}

static bool file_exists_utf8(const std::string& path) {
    if (path.empty()) return false;
    std::ifstream input(path, std::ios::binary);
#ifdef _WIN32
    if (!input) {
        const std::wstring wide = storagellm::wide_path_from_utf8(path);
        if (!wide.empty()) {
            input.open(wide.c_str(), std::ios::binary);
        }
    }
#endif
    return static_cast<bool>(input);
}

static bool write_text_file_utf8(const std::string& path, const std::string& text) {
    if (path.empty()) return false;
    std::ofstream output;
#ifdef _WIN32
    const std::wstring wide = storagellm::wide_path_from_utf8(path);
    if (!wide.empty()) {
        output.open(wide.c_str(), std::ios::binary);
    }
#else
    output.open(path, std::ios::binary);
#endif
    if (!output) return false;
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    return static_cast<bool>(output);
}

struct server_tokenizer {
    bool attempted = false;
    bool loaded = false;
    bool byte_level = false;
    bool has_merges = false;
    std::string model_root;
    std::unordered_map<uint32_t, std::string> id_to_text;
    std::unordered_map<uint32_t, std::string> id_to_piece;
    std::unordered_map<std::string, uint32_t> text_to_id;
    std::unordered_map<std::string, uint32_t> piece_to_id;
    std::unordered_map<std::string, uint32_t> bpe_rank;
    std::unordered_set<uint32_t> special_ids;
    size_t max_piece_bytes = 0;
    uint32_t eos_token_id = 0;
    uint32_t unk_token_id = 0;
    bool has_eos = false;
    bool has_unk = false;
    std::string error;
};

static void append_utf8_codepoint(std::string* out, uint32_t cp) {
    append_utf8(out, cp);
}

static uint32_t utf8_next_codepoint(const std::string& text, size_t* pos) {
    if (!pos || *pos >= text.size()) return 0;
    const unsigned char c0 = static_cast<unsigned char>(text[*pos]);
    if (c0 < 0x80u) {
        ++(*pos);
        return c0;
    }
    if ((c0 & 0xE0u) == 0xC0u && *pos + 1 < text.size()) {
        const uint32_t cp =
            ((uint32_t)(c0 & 0x1Fu) << 6) |
            (uint32_t)(static_cast<unsigned char>(text[*pos + 1]) & 0x3Fu);
        *pos += 2;
        return cp;
    }
    if ((c0 & 0xF0u) == 0xE0u && *pos + 2 < text.size()) {
        const uint32_t cp =
            ((uint32_t)(c0 & 0x0Fu) << 12) |
            ((uint32_t)(static_cast<unsigned char>(text[*pos + 1]) & 0x3Fu) << 6) |
            (uint32_t)(static_cast<unsigned char>(text[*pos + 2]) & 0x3Fu);
        *pos += 3;
        return cp;
    }
    if ((c0 & 0xF8u) == 0xF0u && *pos + 3 < text.size()) {
        const uint32_t cp =
            ((uint32_t)(c0 & 0x07u) << 18) |
            ((uint32_t)(static_cast<unsigned char>(text[*pos + 1]) & 0x3Fu) << 12) |
            ((uint32_t)(static_cast<unsigned char>(text[*pos + 2]) & 0x3Fu) << 6) |
            (uint32_t)(static_cast<unsigned char>(text[*pos + 3]) & 0x3Fu);
        *pos += 4;
        return cp;
    }
    ++(*pos);
    return c0;
}

static uint32_t byte_level_codepoint_for_byte(uint8_t b) {
    if ((b >= 33u && b <= 126u) || (b >= 161u && b <= 172u) || (b >= 174u && b <= 255u)) {
        return b;
    }
    uint32_t n = 0;
    for (uint32_t x = 0; x < b; ++x) {
        if (!((x >= 33u && x <= 126u) || (x >= 161u && x <= 172u) || (x >= 174u && x <= 255u))) {
            ++n;
        }
    }
    return 256u + n;
}

static int byte_level_byte_for_codepoint(uint32_t cp, uint8_t* out) {
    if (!out) return 0;
    if ((cp >= 33u && cp <= 126u) || (cp >= 161u && cp <= 172u) || (cp >= 174u && cp <= 255u)) {
        *out = static_cast<uint8_t>(cp);
        return 1;
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b <= 255u; ++b) {
        if ((b >= 33u && b <= 126u) || (b >= 161u && b <= 172u) || (b >= 174u && b <= 255u)) {
            continue;
        }
        if (256u + n == cp) {
            *out = static_cast<uint8_t>(b);
            return 1;
        }
        ++n;
    }
    return 0;
}

static std::string byte_level_encode_text(const std::string& text) {
    std::string out;
    out.reserve(text.size() * 2u);
    for (unsigned char ch : text) {
        append_utf8_codepoint(&out, byte_level_codepoint_for_byte(ch));
    }
    return out;
}

static std::string byte_level_decode_text(const std::string& encoded) {
    std::string bytes;
    bytes.reserve(encoded.size());
    size_t pos = 0;
    while (pos < encoded.size()) {
        const size_t before = pos;
        const uint32_t cp = utf8_next_codepoint(encoded, &pos);
        uint8_t b = 0;
        if (byte_level_byte_for_codepoint(cp, &b)) {
            bytes.push_back(static_cast<char>(b));
        } else {
            bytes.append(encoded, before, pos - before);
        }
    }
    return bytes;
}

static bool utf8_next_symbol(const std::string& text, size_t* pos, std::string* out) {
    if (!pos || !out || *pos >= text.size()) return false;
    const size_t start = *pos;
    (void)utf8_next_codepoint(text, pos);
    out->assign(text.data() + start, *pos - start);
    return true;
}

static bool utf8_symbol_is_ascii_space(const std::string& symbol) {
    return symbol.size() == 1u && std::isspace(static_cast<unsigned char>(symbol[0]));
}

static bool utf8_symbol_is_word_char(const std::string& symbol) {
    if (symbol.size() != 1u) {
        return true;
    }
    const unsigned char ch = static_cast<unsigned char>(symbol[0]);
    return std::isalnum(ch) != 0;
}

static std::vector<std::string> bytelevel_pretokenize_text(const std::string& text) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t token_start = pos;
        std::string symbol;
        if (!utf8_next_symbol(text, &pos, &symbol)) {
            break;
        }
        const bool leading_space = utf8_symbol_is_ascii_space(symbol);
        if (leading_space) {
            if (pos >= text.size()) {
                out.push_back(symbol);
                break;
            }
            size_t next_pos = pos;
            std::string next_symbol;
            if (!utf8_next_symbol(text, &next_pos, &next_symbol)) {
                out.push_back(symbol);
                break;
            }
            if (utf8_symbol_is_ascii_space(next_symbol)) {
                while (next_pos < text.size()) {
                    size_t peek_pos = next_pos;
                    std::string peek;
                    if (!utf8_next_symbol(text, &peek_pos, &peek) || !utf8_symbol_is_ascii_space(peek)) {
                        break;
                    }
                    next_pos = peek_pos;
                }
                out.emplace_back(text.data() + token_start, next_pos - token_start);
                pos = next_pos;
                continue;
            }
            pos = next_pos;
            symbol = next_symbol;
        }

        const bool word_run = utf8_symbol_is_word_char(symbol);
        while (pos < text.size()) {
            size_t peek_pos = pos;
            std::string peek;
            if (!utf8_next_symbol(text, &peek_pos, &peek)) {
                break;
            }
            if (utf8_symbol_is_ascii_space(peek) || utf8_symbol_is_word_char(peek) != word_run) {
                break;
            }
            pos = peek_pos;
        }
        out.emplace_back(text.data() + token_start, pos - token_start);
    }
    return out;
}

static std::string tokenizer_piece_to_text(const std::string& raw, bool byte_level) {
    if (raw.empty()) return "";
    if (raw.size() >= 2 && raw[0] == '<' && raw[1] == '|') {
        return "";
    }
    if (byte_level) {
        return byte_level_decode_text(raw);
    }
    std::string out;
    for (size_t i = 0; i < raw.size();) {
        if (i + 2 <= raw.size() &&
            static_cast<unsigned char>(raw[i]) == 0xC4u &&
            static_cast<unsigned char>(raw[i + 1]) == 0xA0u) {
            out.push_back(' ');
            i += 2;
            continue;
        }
        if (i + 2 <= raw.size() &&
            static_cast<unsigned char>(raw[i]) == 0xC4u &&
            static_cast<unsigned char>(raw[i + 1]) == 0x8Au) {
            out.push_back('\n');
            i += 2;
            continue;
        }
        if (i + 3 <= raw.size() &&
            static_cast<unsigned char>(raw[i]) == 0xE2u &&
            static_cast<unsigned char>(raw[i + 1]) == 0x96u &&
            static_cast<unsigned char>(raw[i + 2]) == 0x81u) {
            out.push_back(' ');
            i += 3;
            continue;
        }
        if (raw.compare(i, 4, "</w>") == 0) {
            i += 4;
            continue;
        }
        out.push_back(raw[i++]);
    }
    return out;
}

static size_t json_find_matching(const std::string& text, size_t open_pos, char open_ch, char close_ch) {
    if (open_pos >= text.size() || text[open_pos] != open_ch) return std::string::npos;
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == open_ch) {
            ++depth;
        } else if (ch == close_ch) {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

static bool json_find_key_object_range(
    const std::string& text,
    const char* key,
    size_t start,
    size_t* out_begin,
    size_t* out_end
) {
    if (!key || !out_begin || !out_end) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = text.find(needle, start);
    const size_t colon = pos == std::string::npos ? pos : text.find(':', pos + needle.size());
    const size_t begin = colon == std::string::npos ? colon : text.find('{', colon + 1);
    if (begin == std::string::npos) return false;
    const size_t end = json_find_matching(text, begin, '{', '}');
    if (end == std::string::npos) return false;
    *out_begin = begin;
    *out_end = end;
    return true;
}

static bool json_find_key_array_range(
    const std::string& text,
    const char* key,
    size_t start,
    size_t* out_begin,
    size_t* out_end
) {
    if (!key || !out_begin || !out_end) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = text.find(needle, start);
    const size_t colon = pos == std::string::npos ? pos : text.find(':', pos + needle.size());
    const size_t begin = colon == std::string::npos ? colon : text.find('[', colon + 1);
    if (begin == std::string::npos) return false;
    const size_t end = json_find_matching(text, begin, '[', ']');
    if (end == std::string::npos) return false;
    *out_begin = begin;
    *out_end = end;
    return true;
}

static bool json_read_u32_in_range(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* key,
    uint32_t* out
) {
    if (!key || !out || begin >= end || end > text.size()) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = text.find(needle, begin);
    if (pos == std::string::npos || pos >= end) return false;
    const size_t colon = text.find(':', pos + needle.size());
    if (colon == std::string::npos || colon >= end) return false;
    char* parse_end = nullptr;
    const unsigned long value = std::strtoul(text.c_str() + colon + 1, &parse_end, 10);
    if (!parse_end || parse_end == text.c_str() + colon + 1) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

static bool json_read_string_in_range(
    const std::string& text,
    size_t begin,
    size_t end,
    const char* key,
    std::string* out
) {
    if (!key || !out || begin >= end || end > text.size()) return false;
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = text.find(needle, begin);
    if (pos == std::string::npos || pos >= end) return false;
    const size_t colon = text.find(':', pos + needle.size());
    const size_t quote = colon == std::string::npos ? colon : text.find('"', colon + 1);
    if (quote == std::string::npos || quote >= end) return false;
    return parse_json_string_at(text, quote, out, nullptr);
}

static std::string bpe_pair_key(const std::string& a, const std::string& b) {
    std::string key;
    key.reserve(a.size() + b.size() + 1u);
    key += a;
    key.push_back('\x1f');
    key += b;
    return key;
}

static void tokenizer_add_piece(server_tokenizer* tok, const std::string& raw_piece, uint32_t id) {
    if (!tok) return;
    tok->id_to_piece[id] = raw_piece;
    tok->piece_to_id[raw_piece] = id;
    tok->max_piece_bytes = std::max(tok->max_piece_bytes, raw_piece.size());
    const std::string text_piece = tokenizer_piece_to_text(raw_piece, tok->byte_level);
    tok->id_to_text[id] = text_piece;
    if (!text_piece.empty()) {
        tok->text_to_id.emplace(text_piece, id);
        tok->max_piece_bytes = std::max(tok->max_piece_bytes, text_piece.size());
    }
    if (raw_piece.size() >= 1 && raw_piece[0] == '<') {
        tok->special_ids.insert(id);
    }
    const std::string lowered = lower_ascii(raw_piece);
    if (!tok->has_unk && lowered.find("unk") != std::string::npos) {
        tok->unk_token_id = id;
        tok->has_unk = true;
    }
    if (!tok->has_eos && (lowered.find("eos") != std::string::npos ||
                          lowered.find("end") != std::string::npos ||
                          lowered.find("eot") != std::string::npos)) {
        tok->eos_token_id = id;
        tok->has_eos = true;
    }
}

static bool load_tokenizer_vocab(server_tokenizer* tok, const std::string& json) {
    if (!tok) return false;
    tok->byte_level =
        json.find("\"ByteLevel\"") != std::string::npos ||
        json.find("\\u0120") != std::string::npos ||
        json.find("\\u010A") != std::string::npos ||
        json.find("\xC4\xA0") != std::string::npos ||
        json.find("\xC4\x8A") != std::string::npos;

    size_t model_begin = 0;
    size_t model_end = json.size();
    size_t tmp_begin = 0;
    size_t tmp_end = 0;
    if (json_find_key_object_range(json, "model", 0, &tmp_begin, &tmp_end)) {
        model_begin = tmp_begin;
        model_end = tmp_end;
    }

    size_t lbrace = 0;
    size_t rbrace = 0;
    if (!json_find_key_object_range(json, "vocab", model_begin, &lbrace, &rbrace) || lbrace > model_end) {
        tok->error = "tokenizer.json does not expose model.vocab";
        return false;
    }
    if (lbrace == std::string::npos) {
        tok->error = "tokenizer.json does not expose model.vocab";
        return false;
    }
    size_t p = lbrace + 1;
    while (p < rbrace) {
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p >= rbrace) break;
        if (json[p] != '"') {
            ++p;
            continue;
        }
        std::string raw_piece;
        size_t next = p;
        if (!parse_json_string_at(json, p, &raw_piece, &next)) {
            ++p;
            continue;
        }
        p = next;
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        if (p >= json.size() || json[p] != ':') {
            continue;
        }
        ++p;
        while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) ++p;
        char* end = nullptr;
        const long id_long = std::strtol(json.c_str() + p, &end, 10);
        if (!end || end == json.c_str() + p || id_long < 0) {
            continue;
        }
        p = (size_t)(end - json.c_str());
        tokenizer_add_piece(tok, raw_piece, (uint32_t)id_long);
    }

    size_t added_begin = 0;
    size_t added_end = 0;
    if (json_find_key_array_range(json, "added_tokens", 0, &added_begin, &added_end)) {
        size_t ap = added_begin + 1;
        while (ap < added_end) {
            const size_t obj_begin = json.find('{', ap);
            if (obj_begin == std::string::npos || obj_begin >= added_end) break;
            const size_t obj_end = json_find_matching(json, obj_begin, '{', '}');
            if (obj_end == std::string::npos || obj_end > added_end) break;
            uint32_t id = 0;
            std::string content;
            if (json_read_u32_in_range(json, obj_begin, obj_end, "id", &id) &&
                json_read_string_in_range(json, obj_begin, obj_end, "content", &content)) {
                tokenizer_add_piece(tok, content, id);
                tok->special_ids.insert(id);
            }
            ap = obj_end + 1;
        }
    }

    size_t merges_begin = 0;
    size_t merges_end = 0;
    if (json_find_key_array_range(json, "merges", model_begin, &merges_begin, &merges_end) && merges_begin < model_end) {
        uint32_t rank = 0;
        size_t mp = merges_begin + 1;
        while (mp < merges_end) {
            while (mp < merges_end && (std::isspace(static_cast<unsigned char>(json[mp])) || json[mp] == ',')) ++mp;
            if (mp >= merges_end) break;
            std::string left;
            std::string right;
            if (json[mp] == '"') {
                std::string merge;
                size_t next = mp;
                if (!parse_json_string_at(json, mp, &merge, &next)) break;
                const size_t split = merge.find(' ');
                if (split != std::string::npos) {
                    left = merge.substr(0, split);
                    right = merge.substr(split + 1);
                }
                mp = next;
            } else if (json[mp] == '[') {
                const size_t arr_end = json_find_matching(json, mp, '[', ']');
                if (arr_end == std::string::npos || arr_end > merges_end) break;
                size_t sp = mp + 1;
                while (sp < arr_end && json[sp] != '"') ++sp;
                size_t next = sp;
                if (sp < arr_end) parse_json_string_at(json, sp, &left, &next);
                sp = next;
                while (sp < arr_end && json[sp] != '"') ++sp;
                next = sp;
                if (sp < arr_end) parse_json_string_at(json, sp, &right, &next);
                mp = arr_end + 1;
            } else {
                ++mp;
                continue;
            }
            if (!left.empty() || !right.empty()) {
                tok->bpe_rank.emplace(bpe_pair_key(left, right), rank++);
            }
        }
        tok->has_merges = !tok->bpe_rank.empty();
    }

    tok->loaded = !tok->id_to_piece.empty() && !tok->piece_to_id.empty();
    if (!tok->loaded) {
        tok->error = "tokenizer vocab parse produced no usable pieces";
    }
    return tok->loaded;
}

static server_tokenizer& get_server_tokenizer(const server_options& opts) {
    static std::mutex mtx;
    static server_tokenizer tok;
    std::lock_guard<std::mutex> lock(mtx);
    if (tok.attempted && tok.model_root == opts.model_root) {
        return tok;
    }
    tok = server_tokenizer{};
    tok.attempted = true;
    tok.model_root = opts.model_root;
    if (opts.model_root.empty()) {
        tok.error = "model_root is required for tokenizer.json";
        return tok;
    }
    const std::string path = join_model_file(opts.model_root, "tokenizer.json");
    std::string json;
    if (!read_text_file(path, &json)) {
        tok.error = "tokenizer.json was not found under model_root";
        return tok;
    }
    load_tokenizer_vocab(&tok, json);
    return tok;
}

static bool tokenizer_encode_greedy(const server_tokenizer& tok, const std::string& text, std::vector<int32_t>* out_ids) {
    if (!tok.loaded || !out_ids) return false;
    out_ids->clear();

    if (tok.byte_level && tok.has_merges) {
        bool ok = true;
        const std::vector<std::string> pretokens = bytelevel_pretokenize_text(text);
        for (const std::string& pretoken : pretokens) {
            std::vector<std::string> word;
            const std::string encoded = byte_level_encode_text(pretoken);
            size_t p = 0;
            std::string sym;
            while (utf8_next_symbol(encoded, &p, &sym)) {
                word.push_back(sym);
            }
            while (word.size() > 1u) {
                uint32_t best_rank = UINT32_MAX;
                size_t best_index = SIZE_MAX;
                for (size_t i = 0; i + 1u < word.size(); ++i) {
                    auto it = tok.bpe_rank.find(bpe_pair_key(word[i], word[i + 1u]));
                    if (it != tok.bpe_rank.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_index = i;
                    }
                }
                if (best_index == SIZE_MAX) break;
                std::vector<std::string> merged;
                merged.reserve(word.size() - 1u);
                for (size_t i = 0; i < word.size();) {
                    if (i == best_index && i + 1u < word.size()) {
                        merged.push_back(word[i] + word[i + 1u]);
                        i += 2u;
                    } else {
                        merged.push_back(word[i]);
                        ++i;
                    }
                }
                word.swap(merged);
            }
            for (const std::string& piece : word) {
                auto it = tok.piece_to_id.find(piece);
                if (it != tok.piece_to_id.end()) {
                    out_ids->push_back((int32_t)it->second);
                } else {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (ok && !out_ids->empty()) {
            return true;
        }
        out_ids->clear();
    }

    const std::string encoded_text = tok.byte_level ? byte_level_encode_text(text) : text;
    size_t pos = 0;
    while (pos < encoded_text.size()) {
        const size_t max_len = std::min(tok.max_piece_bytes ? tok.max_piece_bytes : encoded_text.size(), encoded_text.size() - pos);
        bool matched = false;
        for (size_t len = max_len; len > 0; --len) {
            auto it = tok.piece_to_id.find(encoded_text.substr(pos, len));
            if (it != tok.piece_to_id.end()) {
                out_ids->push_back((int32_t)it->second);
                pos += len;
                matched = true;
                break;
            }
            auto text_it = tok.text_to_id.find(encoded_text.substr(pos, len));
            if (text_it != tok.text_to_id.end()) {
                out_ids->push_back((int32_t)text_it->second);
                pos += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            if (tok.has_unk) {
                out_ids->push_back((int32_t)tok.unk_token_id);
                size_t next = pos;
                std::string ignored;
                if (utf8_next_symbol(encoded_text, &next, &ignored) && next > pos) {
                    pos = next;
                } else {
                    ++pos;
                }
            } else {
                return false;
            }
        }
    }
    return !out_ids->empty();
}

static std::string tokenizer_decode_ids(const server_tokenizer& tok, const std::vector<uint32_t>& ids) {
    std::string pieces;
    for (uint32_t id : ids) {
        if (tok.special_ids.count(id)) {
            continue;
        }
        auto raw_it = tok.id_to_piece.find(id);
        if (raw_it != tok.id_to_piece.end()) {
            pieces += raw_it->second;
            continue;
        }
        auto text_it = tok.id_to_text.find(id);
        if (text_it != tok.id_to_text.end()) {
            pieces += text_it->second;
        } else {
            pieces += "<token:";
            pieces += std::to_string(id);
            pieces += ">";
        }
    }
    if (tok.byte_level) {
        return byte_level_decode_text(pieces);
    }
    return tokenizer_piece_to_text(pieces, false);
}

static std::string extract_generation_text(const std::string& body) {
    std::vector<std::string> contents = json_collect_string_values(body, "content");
    if (!contents.empty()) {
        std::string prompt;
        for (const std::string& item : contents) {
            if (!prompt.empty()) prompt += "\n";
            prompt += item;
        }
        return prompt;
    }
    std::vector<std::string> text_items = json_collect_string_values(body, "text");
    if (!text_items.empty()) {
        std::string prompt;
        for (const std::string& item : text_items) {
            if (!prompt.empty()) prompt += "\n";
            prompt += item;
        }
        return prompt;
    }
    std::string value;
    if (json_read_string_value(body, "prompt", &value)) return value;
    if (json_read_string_value(body, "input", &value)) return value;
    return "";
}

struct server_generation_result {
    int ok = 0;
    int http_status = 200;
    std::string text;
    std::string error_code;
    std::string error_message;
    uint32_t prompt_tokens = 0;
    uint32_t completion_tokens = 0;
    std::string finish_reason = "stop";
};

static server_generation_result run_server_generation(
    moe_pc_engine_t* engine,
    std::mutex* engine_mutex,
    const server_options& opts,
    const std::string& body
) {
    server_generation_result result;
    std::vector<int> request_ids = storagellm::json_read_int_array(body, "input_ids");
    std::vector<int32_t> input_ids;
    server_tokenizer& tok = get_server_tokenizer(opts);
    if (!request_ids.empty()) {
        input_ids.reserve(request_ids.size());
        for (int id : request_ids) {
            input_ids.push_back((int32_t)id);
        }
    } else {
        const std::string text = extract_generation_text(body);
        if (text.empty()) {
            result.http_status = 400;
            result.error_code = "invalid_request_error";
            result.error_message = "OpenAI request must include messages/content, prompt, input, or input_ids";
            return result;
        }
        if (!tok.loaded) {
            result.http_status = 503;
            result.error_code = "tokenizer_unavailable";
            result.error_message = tok.error.empty() ? "tokenizer.json is not loaded" : tok.error;
            return result;
        }
        if (!tokenizer_encode_greedy(tok, text, &input_ids)) {
            result.http_status = 422;
            result.error_code = "tokenization_failed";
            result.error_message = "tokenizer vocab could not encode the request text";
            return result;
        }
    }

    int max_tokens = 128;
    int parsed = 0;
    if (storagellm::json_read_int(body, "max_output_tokens", &parsed) && parsed > 0) {
        max_tokens = parsed;
    } else if (storagellm::json_read_int(body, "max_completion_tokens", &parsed) && parsed > 0) {
        max_tokens = parsed;
    } else if (storagellm::json_read_int(body, "max_tokens", &parsed) && parsed > 0) {
        max_tokens = parsed;
    }
    if (max_tokens > 4096) max_tokens = 4096;

    std::vector<uint32_t> out_ids((size_t)max_tokens);
    moe_generation_config_t cfg{};
    cfg.max_new_tokens = (uint32_t)max_tokens;
    cfg.stop_on_eos = tok.has_eos ? 1 : 0;
    cfg.eos_token_id = tok.eos_token_id;
    moe_generation_stats_t stats{};
    int generated = 0;
    if (engine_mutex) {
        std::lock_guard<std::mutex> lock(*engine_mutex);
        server_orchestrate_request_prefetch(engine, opts, body);
        generated = moe_pc_engine_generate_token_ids(
            engine,
            input_ids.data(),
            static_cast<uint32_t>(input_ids.size()),
            &cfg,
            out_ids.data(),
            static_cast<uint32_t>(out_ids.size()),
            &stats
        );
    } else {
        generated = moe_pc_engine_generate_token_ids(
            engine,
            input_ids.data(),
            static_cast<uint32_t>(input_ids.size()),
            &cfg,
            out_ids.data(),
            static_cast<uint32_t>(out_ids.size()),
            &stats
        );
    }
    if (!generated) {
        result.http_status = 503;
        result.error_code = "generation_not_ready";
        result.error_message = stats.error[0] ? stats.error : "generation failed";
        return result;
    }
    out_ids.resize(stats.completion_tokens);
    result.ok = 1;
    result.prompt_tokens = stats.prompt_tokens;
    result.completion_tokens = stats.completion_tokens;
    result.finish_reason = stats.finish_reason == 1u ? "length" : "stop";
    result.text = tok.loaded ? tokenizer_decode_ids(tok, out_ids) : tokenizer_decode_ids(server_tokenizer{}, out_ids);
    return result;
}

struct server_eval_result {
    int ok = 0;
    int http_status = 200;
    std::string error_code;
    std::string error_message;
    uint32_t input_tokens = 0;
    uint32_t evaluated_tokens = 0;
    double nll = 0.0;
    double mean_nll = 0.0;
    double perplexity = 0.0;
};

static server_eval_result run_server_eval(
    moe_pc_engine_t* engine,
    std::mutex* engine_mutex,
    const server_options& opts,
    const std::string& body
) {
    server_eval_result result;
    std::vector<int> request_ids = storagellm::json_read_int_array(body, "input_ids");
    std::vector<int32_t> input_ids;
    server_tokenizer& tok = get_server_tokenizer(opts);
    if (!request_ids.empty()) {
        input_ids.reserve(request_ids.size());
        for (int id : request_ids) {
            input_ids.push_back((int32_t)id);
        }
    } else {
        const std::string text = extract_generation_text(body);
        if (text.empty()) {
            result.http_status = 400;
            result.error_code = "invalid_request_error";
            result.error_message = "eval request must include input, prompt, messages/content, or input_ids";
            return result;
        }
        if (!tok.loaded) {
            result.http_status = 503;
            result.error_code = "tokenizer_unavailable";
            result.error_message = tok.error.empty() ? "tokenizer.json is not loaded" : tok.error;
            return result;
        }
        if (!tokenizer_encode_greedy(tok, text, &input_ids)) {
            result.http_status = 422;
            result.error_code = "tokenization_failed";
            result.error_message = "tokenizer vocab could not encode the eval text";
            return result;
        }
    }
    if (input_ids.size() < 2) {
        result.http_status = 400;
        result.error_code = "invalid_request_error";
        result.error_message = "eval requires at least two tokens";
        return result;
    }

    moe_eval_stats_t stats{};
    int evaluated = 0;
    if (engine_mutex) {
        std::lock_guard<std::mutex> lock(*engine_mutex);
        server_orchestrate_request_prefetch(engine, opts, body);
        evaluated = moe_pc_engine_eval_token_ids(
            engine,
            input_ids.data(),
            static_cast<uint32_t>(input_ids.size()),
            &stats
        );
    } else {
        evaluated = moe_pc_engine_eval_token_ids(
            engine,
            input_ids.data(),
            static_cast<uint32_t>(input_ids.size()),
            &stats
        );
    }
    if (!evaluated) {
        result.http_status = 503;
        result.error_code = "eval_not_ready";
        result.error_message = stats.error[0] ? stats.error : "eval failed";
        return result;
    }

    result.ok = 1;
    result.input_tokens = stats.input_tokens;
    result.evaluated_tokens = stats.evaluated_tokens;
    result.nll = stats.nll;
    result.mean_nll = stats.mean_nll;
    result.perplexity = stats.perplexity;
    return result;
}

static std::string make_openai_error_json(const server_generation_result& result) {
    std::ostringstream out;
    out << "{"
        << "\"error\":{"
        << "\"message\":\"" << json_escape(result.error_message) << "\","
        << "\"type\":\"" << json_escape(result.error_code.empty() ? "server_error" : result.error_code) << "\","
        << "\"code\":\"" << json_escape(result.error_code.empty() ? "server_error" : result.error_code) << "\""
        << "}"
        << "}";
    return out.str();
}

static std::string make_eval_error_json(const server_eval_result& result) {
    std::ostringstream out;
    out << "{"
        << "\"error\":{"
        << "\"message\":\"" << json_escape(result.error_message) << "\","
        << "\"type\":\"" << json_escape(result.error_code.empty() ? "server_error" : result.error_code) << "\","
        << "\"code\":\"" << json_escape(result.error_code.empty() ? "server_error" : result.error_code) << "\""
        << "}"
        << "}";
    return out.str();
}

static const char* generation_phase_name(uint32_t phase) {
    switch (phase) {
        case moe_GEN_PHASE_INIT: return "init";
        case moe_GEN_PHASE_PREFILL: return "prefill";
        case moe_GEN_PHASE_ATTENTION: return "attention";
        case moe_GEN_PHASE_MLP: return "mlp";
        case moe_GEN_PHASE_LM_HEAD: return "lm_head";
        case moe_GEN_PHASE_DONE: return "done";
        default: return "idle";
    }
}

static bool get_cached_model_root_check(
    const server_runtime_state* runtime,
    const std::string& model_root,
    moe_model_root_check_t* out_check
);

static const char* forward_adapter_name(uint32_t adapter) {
    switch (adapter) {
        case 1: return "glm_raw";
        case 2: return "gguf_generic";
        case 3: return "gguf_gemma";
        default: return "none";
    }
}

static std::string make_health_json(
    const server_options& opts,
    const moe_backend_caps_t& caps,
    const moe_forward_status_t& forward,
    const moe_pc_engine_stats_t& stats,
    const moe_io_stats_t& io_stats,
    const server_runtime_state* runtime = nullptr
) {
    std::ostringstream out;
    const bool loading = runtime && runtime->model_ready.load(std::memory_order_acquire) == 0 &&
        runtime->model_failed.load(std::memory_order_acquire) == 0;
    const bool failed = runtime && runtime->model_failed.load(std::memory_order_acquire) != 0;
    const bool loaded = !loading && !failed;
    const bool generation_ready = loaded && forward.decode_loop_ready != 0;
    out << "{"
        << "\"status\":\"ok\","
        << "\"modelLoading\":" << (loading ? "true" : "false") << ","
        << "\"modelReady\":" << (loaded ? "true" : "false") << ","
        << "\"modelLoaded\":" << (loaded ? "true" : "false") << ","
        << "\"generationReady\":" << (generation_ready ? "true" : "false") << ","
        << "\"modelFailed\":" << (failed ? "true" : "false") << ","
        << "\"mode\":\"openclaw\","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"base_url\":\"http://" << json_escape(opts.host) << ":" << opts.port << "/v1\","
        << "\"backend\":\"" << moe_backend_name(caps.backend) << "\","
        << "\"platform\":\"" << moe_platform_name(caps.platform) << "\","
        << "\"forwardPath\":\"" << forward_adapter_name(forward.forward_adapter) << "\","
        << "\"forwardAdapter\":\"" << forward_adapter_name(forward.forward_adapter) << "\","
        << "\"forwardAdapterId\":" << forward.forward_adapter << ","
        << "\"forwardAdapterExecutable\":" << (forward.forward_adapter_executable ? "true" : "false") << ","
        << "\"dynamicShapeReady\":" << (forward.dynamic_shape_ready ? "true" : "false") << ","
        << "\"dynamicNumHiddenLayers\":" << forward.dynamic_num_hidden_layers << ","
        << "\"dynamicHiddenSize\":" << forward.dynamic_hidden_size << ","
        << "\"dynamicVocabSize\":" << forward.dynamic_vocab_size << ","
        << "\"storageStateValid\":" << (forward.storage_state_valid ? "true" : "false") << ","
        << "\"tensorTableLoaded\":" << (forward.tensor_table_loaded ? "true" : "false") << ","
        << "\"expertTripletAvailable\":" << (forward.expert_triplet_available ? "true" : "false") << ","
        << "\"tokenizerReady\":" << (forward.tokenizer_ready ? "true" : "false") << ","
        << "\"embeddingReady\":" << (forward.embedding_ready ? "true" : "false") << ","
        << "\"lmHeadReady\":" << (forward.lm_head_ready ? "true" : "false") << ","
        << "\"attentionProjectorReady\":" << (forward.attention_projector_ready ? "true" : "false") << ","
        << "\"attentionDecodeReady\":" << (forward.attention_decode_ready ? "true" : "false") << ","
        << "\"kvCacheReady\":" << (forward.kv_cache_ready ? "true" : "false") << ","
        << "\"attentionReady\":" << (forward.attention_ready ? "true" : "false") << ","
        << "\"samplerReady\":" << (forward.sampler_ready ? "true" : "false") << ","
        << "\"decodeLoopReady\":" << (forward.decode_loop_ready ? "true" : "false") << ","
        << "\"chatGraphReady\":" << (forward.chat_graph_ready ? "true" : "false") << ","
        << "\"tensorCount\":" << forward.tensor_count << ","
        << "\"vramBudgetBytes\":" << opts.engine_config.vram_budget_bytes << ","
        << "\"ramBudgetBytes\":" << opts.engine_config.ram_budget_bytes << ","
        << "\"vramUsedBytes\":" << stats.vram_used_bytes << ","
        << "\"commonVramReservedBytes\":" << stats.common_vram_reserved_bytes << ","
        << "\"vramExpertUsedBytes\":" << stats.vram_expert_used_bytes << ","
        << "\"commonRawPrefetchedBytes\":" << stats.common_raw_prefetched_bytes << ","
        << "\"commonRawPrefetchedSpans\":" << stats.common_raw_prefetched_spans << ","
        << "\"commonRawTensorCount\":" << stats.common_raw_tensor_count << ","
        << "\"processRssBytes\":" << storagellm::current_process_rss_bytes() << ","
        << "\"availableRamBytes\":" << storagellm::available_ram_bytes() << ","
        << "\"deviceAllocatedBytes\":" << stats.device_allocated_bytes << ","
        << "\"deviceAllocationCount\":" << stats.device_allocation_count << ","
        << "\"deviceFixedBytes\":" << stats.device_fixed_bytes << ","
        << "\"deviceActivationBytes\":" << stats.device_activation_bytes << ","
        << "\"deviceExpertBytes\":" << stats.device_expert_bytes << ","
        << "\"deviceStreamBytes\":" << stats.device_stream_bytes << ","
        << "\"deviceTotalBytes\":" << stats.device_total_bytes << ","
        << "\"deviceFreeBytes\":" << stats.device_free_bytes << ","
        << "\"modelLibLoaded\":" << (stats.model_lib_loaded ? "true" : "false") << ","
        << "\"modelLibGenerated\":" << (stats.model_lib_generated ? "true" : "false") << ","
        << "\"modelLibCompileAttempted\":" << (stats.model_lib_compile_attempted ? "true" : "false") << ","
        << "\"modelLibCompileSucceeded\":" << (stats.model_lib_compile_succeeded ? "true" : "false") << ","
        << "\"modelLibPath\":\"" << json_escape(stats.model_lib_path) << "\","
        << "\"ramUsedBytes\":" << stats.ram_used_bytes << ","
        << "\"storageTierBytes\":" << stats.db_used_bytes << ","
        << "\"vramExpertCount\":" << stats.vram_expert_count << ","
        << "\"ramExpertCount\":" << stats.ram_expert_count << ","
        << "\"storageExpertCount\":" << stats.db_expert_count << ","
        << "\"generationStarted\":" << stats.generation_started << ","
        << "\"generationCompleted\":" << stats.generation_completed << ","
        << "\"generationFailed\":" << stats.generation_failed << ","
        << "\"generationActive\":" << stats.generation_active << ","
        << "\"generationToken\":" << stats.generation_token << ","
        << "\"generationLayer\":" << stats.generation_layer << ","
        << "\"generationPhase\":\"" << generation_phase_name(stats.generation_phase) << "\","
        << "\"kvMode\":\"" << moe_kv_mode_name(stats.kv_mode) << "\","
        << "\"offloadGgufValid\":" << (stats.offload_gguf_valid ? "true" : "false") << ","
        << "\"offloadGgufFileCount\":" << stats.offload_gguf_file_count << ","
        << "\"offloadGgufTensorHeaders\":" << stats.offload_gguf_tensor_count << ","
        << "\"offloadGgufExecutableTensors\":" << stats.offload_gguf_executable_tensor_count << ","
        << "\"qkvForcedByFormat\":" << (stats.qkv_forced_by_format ? "true" : "false") << ","
        << "\"qkvKBits\":" << stats.qkv_k_bits << ","
        << "\"qkvVBits\":" << stats.qkv_v_bits << ","
        << "\"qkvGroupSize\":" << stats.qkv_group_size << ","
        << "\"qkvPageSizeTokens\":" << stats.qkv_page_size_tokens << ","
        << "\"qkvSinkTokens\":" << stats.qkv_sink_tokens << ","
        << "\"weightQuantBits\":" << stats.weight_quant_bits << ","
        << "\"weightQuantEncoding\":" << stats.weight_quant_encoding << ","
        << "\"weightQuantBlockSize\":" << stats.weight_quant_block_size << ","
        << "\"weightQuantFamily\":\"" << json_escape(stats.weight_quant_family) << "\","
        << "\"weightKernelFamily\":\"" << json_escape(stats.weight_kernel_family) << "\","
        << "\"offloadGgufFirstFile\":\"" << json_escape(stats.offload_gguf_first_file) << "\","
        << "\"recommendedVramCacheBytes\":" << caps.recommended_vram_cache_bytes << ","
        << "\"recommendedVramCommonBytes\":" << caps.recommended_vram_common_bytes << ","
        << "\"recommendedPinnedBytes\":" << caps.recommended_pinned_bytes << ","
        << "\"queuedRequests\":" << io_stats.queued_requests << ","
        << "\"completedRequests\":" << io_stats.completed_requests << ","
        << "\"failedRequests\":" << io_stats.failed_requests << ","
        << "\"droppedRequests\":" << io_stats.dropped_requests << ","
        << "\"deviceAllocFailures\":" << io_stats.device_alloc_failures << ","
        << "\"deviceCopyFailures\":" << io_stats.device_copy_failures << ","
        << "\"diskQueueDepth\":" << io_stats.disk_queue_depth << ","
        << "\"pinnedQueueDepth\":" << io_stats.pinned_queue_depth << ","
        << "\"gpuQueueDepth\":" << io_stats.gpu_queue_depth << ","
        << "\"activeWorkers\":" << io_stats.active_workers << ","
        << "\"stagingSlotDeficit\":" << io_stats.staging_slot_deficit << ","
        << "\"recommendedStagingBytes\":" << io_stats.recommended_staging_bytes;
    if (!opts.model_root.empty()) {
        moe_model_root_check_t root_check{};
        if (get_cached_model_root_check(runtime, opts.model_root, &root_check) ||
            moe_storage_validate_model_root(opts.model_root.c_str(), &root_check)) {
            out << ",\"modelRoot\":\"" << json_escape(opts.model_root) << "\""
                << ",\"modelRootValid\":" << (root_check.valid ? "true" : "false")
                << ",\"modelRootExpectedParts\":" << root_check.expected_part_count
                << ",\"modelRootPresentParts\":" << root_check.present_part_count
                << ",\"modelRootMissingParts\":" << root_check.missing_part_count
                << ",\"modelRootSizeMismatches\":" << root_check.size_mismatch_count
                << ",\"modelRootExpectedBytes\":" << root_check.expected_total_bytes
                << ",\"modelRootPresentBytes\":" << root_check.present_total_bytes;
            if (root_check.first_missing_part) {
                out << ",\"modelRootFirstMissingPart\":" << root_check.first_missing_part
                    << ",\"modelRootFirstMissingPath\":\"" << json_escape(root_check.first_missing_path) << "\"";
            }
            if (root_check.first_size_mismatch_part) {
                out << ",\"modelRootFirstSizeMismatchPart\":" << root_check.first_size_mismatch_part
                    << ",\"modelRootFirstSizeMismatchPath\":\"" << json_escape(root_check.first_size_mismatch_path) << "\""
                    << ",\"modelRootFirstExpectedBytes\":" << root_check.first_expected_bytes
                    << ",\"modelRootFirstActualBytes\":" << root_check.first_actual_bytes;
            }
        }
    }
    out << ","
        << "\"loopback_only\":" << (is_loopback_host(opts.host) ? "true" : "false")
        ;
    if (failed && runtime) {
        std::lock_guard<std::mutex> lock(runtime->error_mutex);
        out << ",\"modelLoadError\":\"" << json_escape(runtime->error_message) << "\"";
    }
    if (runtime) {
        std::lock_guard<std::mutex> lock(runtime->stage_mutex);
        out << ",\"modelLoadStage\":\"" << json_escape(runtime->load_stage) << "\"";
    }
    out << "}";
    return out.str();
}

static std::string make_models_json(const server_options& opts) {
    std::ostringstream out;
    out << "{"
        << "\"object\":\"list\","
        << "\"data\":[{"
        << "\"id\":\"" << json_escape(opts.model_id) << "\","
        << "\"object\":\"model\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"owned_by\":\"storagellm\""
        << "}]"
        << "}";
    return out.str();
}

static std::string make_openclaw_config_json(const server_options& opts) {
    const std::string base_url = "http://" + opts.host + ":" + std::to_string(opts.port) + "/v1";
    const std::string model_ref = "storagellm/" + opts.model_id;
    std::ostringstream out;
    out << "{\n"
        << "  \"agents\": {\n"
        << "    \"defaults\": {\n"
        << "      \"model\": { \"primary\": \"" << json_escape(model_ref) << "\" },\n"
        << "      \"models\": {\n"
        << "        \"" << json_escape(model_ref) << "\": { \"alias\": \"StorageLLM\" }\n"
        << "      }\n"
        << "    }\n"
        << "  },\n"
        << "  \"models\": {\n"
        << "    \"mode\": \"merge\",\n"
        << "    \"providers\": {\n"
        << "      \"storagellm\": {\n"
        << "        \"baseUrl\": \"" << json_escape(base_url) << "\",\n"
        << "        \"apiKey\": \"sk-local\",\n"
        << "        \"api\": \"openai-responses\",\n"
        << "        \"models\": [\n"
        << "          {\n"
        << "            \"id\": \"" << json_escape(opts.model_id) << "\",\n"
        << "            \"name\": \"" << json_escape(opts.model_id) << "\",\n"
        << "            \"reasoning\": false,\n"
        << "            \"input\": [\"text\"],\n"
        << "            \"contextWindow\": 131072,\n"
        << "            \"maxTokens\": 4096,\n"
        << "            \"cost\": { \"input\": 0, \"output\": 0, \"cacheRead\": 0, \"cacheWrite\": 0 },\n"
        << "            \"compat\": { \"requiresStringContent\": true, \"supportsTools\": false }\n"
        << "          }\n"
        << "        ]\n"
        << "      }\n"
        << "    }\n"
        << "  }\n"
        << "}\n";
    return out.str();
}

static bool write_openclaw_config_file(const server_options& opts) {
    if (opts.openclaw_config_path.empty()) {
        return true;
    }
    return write_text_file_utf8(opts.openclaw_config_path, make_openclaw_config_json(opts));
}

static std::string generation_status_text(
    const server_options& opts,
    const moe_backend_caps_t& caps,
    const moe_forward_status_t& forward
) {
    std::ostringstream out;
    out << "StorageLLM local OpenClaw server is running at http://"
        << opts.host << ":" << opts.port << "/v1. "
        << "Backend auto-detected as " << moe_backend_name(caps.backend)
        << " on " << moe_platform_name(caps.platform)
        << ". Storage state valid=" << (forward.storage_state_valid ? "true" : "false")
        << ", tensor table loaded=" << (forward.tensor_table_loaded ? "true" : "false")
        << ", expert triplet runtime=" << (forward.expert_triplet_available ? "ready" : "waiting")
        << ", tokenizer=" << (forward.tokenizer_ready ? "ready" : "waiting")
        << ", embedding=" << (forward.embedding_ready ? "ready" : "waiting")
        << ", lm_head=" << (forward.lm_head_ready ? "ready" : "waiting")
        << ", attention_projector=" << (forward.attention_projector_ready ? "ready" : "waiting")
        << ", attention_decode=" << (forward.attention_decode_ready ? "ready" : "waiting")
        << ", kv_cache=" << (forward.kv_cache_ready ? "ready" : "waiting")
        << ", attention=" << (forward.attention_ready ? "ready" : "waiting")
        << ", sampler=" << (forward.sampler_ready ? "ready" : "waiting")
        << ", decode_loop=" << (forward.decode_loop_ready ? "ready" : "waiting")
        << ", chat graph ready=" << (forward.chat_graph_ready ? "true" : "false") << ".";
    return out.str();
}

static std::string make_chat_json(
    const server_options& opts,
    const server_generation_result& generated
) {
    std::ostringstream out;
    out << "{"
        << "\"id\":\"chatcmpl-storagellm-local\","
        << "\"object\":\"chat.completion\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{"
        << "\"index\":0,"
        << "\"message\":{\"role\":\"assistant\",\"content\":\"" << json_escape(generated.text) << "\"},"
        << "\"finish_reason\":\"" << generated.finish_reason << "\""
        << "}],"
        << "\"usage\":{\"prompt_tokens\":" << generated.prompt_tokens
        << ",\"completion_tokens\":" << generated.completion_tokens
        << ",\"total_tokens\":" << (generated.prompt_tokens + generated.completion_tokens) << "}"
        << "}";
    return out.str();
}

static std::string make_chat_sse(
    const server_options& opts,
    const server_generation_result& generated
) {
    std::ostringstream out;
    out << "data: {"
        << "\"id\":\"chatcmpl-storagellm-local\","
        << "\"object\":\"chat.completion.chunk\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\""
        << json_escape(generated.text)
        << "\"},\"finish_reason\":null}]"
        << "}\n\n";
    out << "data: {"
        << "\"id\":\"chatcmpl-storagellm-local\","
        << "\"object\":\"chat.completion.chunk\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" << generated.finish_reason << "\"}]"
        << "}\n\n";
    out << "data: [DONE]\n\n";
    return out.str();
}

static std::string make_completion_json(
    const server_options& opts,
    const server_generation_result& generated
) {
    std::ostringstream out;
    out << "{"
        << "\"id\":\"cmpl-storagellm-local\","
        << "\"object\":\"text_completion\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{\"index\":0,\"text\":\"" << json_escape(generated.text)
        << "\",\"finish_reason\":\"" << generated.finish_reason << "\"}],"
        << "\"usage\":{\"prompt_tokens\":" << generated.prompt_tokens
        << ",\"completion_tokens\":" << generated.completion_tokens
        << ",\"total_tokens\":" << (generated.prompt_tokens + generated.completion_tokens) << "}"
        << "}";
    return out.str();
}

static std::string make_completion_sse(
    const server_options& opts,
    const server_generation_result& generated
) {
    std::ostringstream out;
    out << "data: {"
        << "\"id\":\"cmpl-storagellm-local\","
        << "\"object\":\"text_completion\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{\"index\":0,\"text\":\"" << json_escape(generated.text)
        << "\",\"finish_reason\":null}]"
        << "}\n\n";
    out << "data: {"
        << "\"id\":\"cmpl-storagellm-local\","
        << "\"object\":\"text_completion\","
        << "\"created\":" << unix_time_seconds() << ","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"" << generated.finish_reason << "\"}]"
        << "}\n\n";
    out << "data: [DONE]\n\n";
    return out.str();
}

static std::string make_response_json(
    const server_options& opts,
    const server_generation_result& generated
) {
    std::ostringstream out;
    out << "{"
        << "\"id\":\"resp-storagellm-local\","
        << "\"object\":\"response\","
        << "\"created_at\":" << unix_time_seconds() << ","
        << "\"status\":\"completed\","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"output\":[{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\""
        << json_escape(generated.text) << "\"}]}],"
        << "\"usage\":{\"input_tokens\":" << generated.prompt_tokens
        << ",\"output_tokens\":" << generated.completion_tokens
        << ",\"total_tokens\":" << (generated.prompt_tokens + generated.completion_tokens) << "}"
        << "}";
    return out.str();
}

static std::string json_double(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream out;
    out.precision(17);
    out << value;
    return out.str();
}

static std::string make_eval_json(
    const server_options& opts,
    const server_eval_result& evaluated
) {
    std::ostringstream out;
    out << "{"
        << "\"object\":\"storagellm.eval\","
        << "\"model\":\"" << json_escape(opts.model_id) << "\","
        << "\"input_tokens\":" << evaluated.input_tokens << ","
        << "\"evaluated_tokens\":" << evaluated.evaluated_tokens << ","
        << "\"nll\":" << json_double(evaluated.nll) << ","
        << "\"mean_nll\":" << json_double(evaluated.mean_nll) << ","
        << "\"ppl\":" << json_double(evaluated.perplexity) << ","
        << "\"perplexity\":" << json_double(evaluated.perplexity)
        << "}";
    return out.str();
}

static std::string make_chat_not_ready_json(const moe_forward_status_t& forward) {
    std::ostringstream out;
    out << "{"
        << "\"error\":{"
        << "\"message\":\"GGUF offload server status surface is not ready; storage="
        << (forward.storage_state_valid ? "valid" : "invalid")
        << ", tensorTable=" << (forward.tensor_table_loaded ? "loaded" : "missing")
        << ", expertTriplet=" << (forward.expert_triplet_available ? "ready" : "missing")
        << ", tokenizer=" << (forward.tokenizer_ready ? "ready" : "missing")
        << ", embedding=" << (forward.embedding_ready ? "ready" : "missing")
        << ", lmHead=" << (forward.lm_head_ready ? "ready" : "missing")
        << ", attentionProjector=" << (forward.attention_projector_ready ? "ready" : "missing")
        << ", attentionDecode=" << (forward.attention_decode_ready ? "ready" : "missing")
        << ", kvCache=" << (forward.kv_cache_ready ? "ready" : "missing")
        << ", attention=" << (forward.attention_ready ? "ready" : "missing")
        << ", sampler=" << (forward.sampler_ready ? "ready" : "missing")
        << ", decodeLoop=" << (forward.decode_loop_ready ? "ready" : "missing")
        << "\","
        << "\"type\":\"service_unavailable\","
        << "\"code\":\"runtime_status_not_ready\""
        << "}"
        << "}";
    return out.str();
}

static bool server_chat_status_surface_ready(const moe_forward_status_t& forward) {
    return forward.storage_state_valid &&
        forward.tensor_table_loaded &&
        forward.expert_triplet_available;
}

static std::string http_reason(int status) {
    switch (status) {
        case 200:
            return "OK";
        case 204:
            return "No Content";
        case 400:
            return "Bad Request";
        case 404:
            return "Not Found";
        case 422:
            return "Unprocessable Entity";
        case 405:
            return "Method Not Allowed";
        case 500:
            return "Internal Server Error";
        case 503:
            return "Service Unavailable";
        default:
            return "OK";
    }
}

static std::string make_response(
    int status,
    const std::string& content_type,
    const std::string& body
) {
    std::string typed_content = content_type;
    if (typed_content == "application/json" ||
        typed_content == "text/event-stream" ||
        typed_content == "text/plain") {
        typed_content += "; charset=utf-8";
    }
    std::ostringstream out;
    out << "HTTP/1.1 " << status << " " << http_reason(status) << "\r\n"
        << "Content-Type: " << typed_content << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Headers: authorization, content-type\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;
    return out.str();
}

static bool send_all(socket_handle_t client, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
#ifdef _WIN32
        const int n = send(client, data.data() + sent, static_cast<int>(data.size() - sent), 0);
#else
        const ssize_t n = send(client, data.data() + sent, data.size() - sent, 0);
#endif
        if (n <= 0) {
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

static bool parse_http_request(socket_handle_t client, http_request* out_req) {
    if (!out_req) {
        return false;
    }
    std::string raw;
    char buf[4096];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos && raw.size() < 1024 * 1024) {
#ifdef _WIN32
        const int n = recv(client, buf, static_cast<int>(sizeof(buf)), 0);
#else
        const ssize_t n = recv(client, buf, sizeof(buf), 0);
#endif
        if (n <= 0) {
            return false;
        }
        raw.append(buf, buf + n);
        header_end = raw.find("\r\n\r\n");
    }
    if (header_end == std::string::npos) {
        return false;
    }

    const std::string header = raw.substr(0, header_end + 4);
    std::istringstream input(header);
    std::string request_line;
    if (!std::getline(input, request_line)) {
        return false;
    }
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }
    std::istringstream first(request_line);
    std::string version;
    first >> out_req->method >> out_req->target >> version;
    if (out_req->method.empty() || out_req->target.empty()) {
        return false;
    }
    const size_t qmark = out_req->target.find('?');
    out_req->path = qmark == std::string::npos ? out_req->target : out_req->target.substr(0, qmark);

    size_t content_length = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t sep = line.find(':');
        if (sep == std::string::npos) {
            continue;
        }
        const std::string name = lower_ascii(line.substr(0, sep));
        std::string value = line.substr(sep + 1);
        while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
            value.erase(value.begin());
        }
        if (name == "content-length") {
            content_length = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
        }
    }

    const size_t body_start = header_end + 4;
    if (raw.size() > body_start) {
        out_req->body = raw.substr(body_start);
    }
    while (out_req->body.size() < content_length && raw.size() < 64ull * 1024ull * 1024ull) {
#ifdef _WIN32
        const int n = recv(client, buf, static_cast<int>(sizeof(buf)), 0);
#else
        const ssize_t n = recv(client, buf, sizeof(buf), 0);
#endif
        if (n <= 0) {
            return false;
        }
        out_req->body.append(buf, buf + n);
    }
    if (out_req->body.size() > content_length) {
        out_req->body.resize(content_length);
    }
    return true;
}

static std::string route_request(
    const http_request& req,
    const server_options& opts,
    moe_pc_engine_t* engine,
    std::mutex* engine_mutex,
    const server_runtime_state* runtime = nullptr
) {
    if (!engine && runtime) {
        engine = runtime->engine.load(std::memory_order_acquire);
    }
    if (req.method == "OPTIONS") {
        return make_response(204, "text/plain", "");
    }
    if (req.method == "GET" && (req.path == "/v1/models" || req.path == "/models")) {
        return make_response(200, "application/json", make_models_json(opts));
    }
    if (req.method == "GET" && req.path == "/openclaw/config") {
        return make_response(200, "application/json", make_openclaw_config_json(opts));
    }
    auto load_backend_caps = [&]() {
        moe_backend_caps_t caps{};
        const bool ready = !runtime || runtime->model_ready.load(std::memory_order_acquire) != 0;
        if (engine && ready && !moe_pc_engine_get_backend_caps(engine, &caps)) {
            moe_pc_detect_backend_caps(opts.engine_config.preferred_backend, opts.engine_config.platform, &caps);
        } else if (!engine && runtime) {
            caps = runtime->initial_caps;
        } else if (!ready && runtime) {
            caps = runtime->initial_caps;
        }
        return caps;
    };

    if (req.method == "GET" && req.path == "/health") {
        const moe_backend_caps_t caps = load_backend_caps();
        const bool ready = !runtime || runtime->model_ready.load(std::memory_order_acquire) != 0;
        moe_forward_status_t forward{};
        if (engine && ready) {
            moe_pc_engine_get_forward_status(engine, &forward);
        }
        moe_pc_engine_stats_t stats{};
        if (engine && ready) {
            moe_pc_engine_get_stats(engine, &stats);
        }
        moe_io_stats_t io_stats{};
        if (engine && ready) {
            moe_pc_engine_get_io_stats(engine, &io_stats);
        }
        return make_response(200, "application/json", make_health_json(opts, caps, forward, stats, io_stats, runtime));
    }
    if (runtime && runtime->model_ready.load(std::memory_order_acquire) == 0) {
        const std::string body = runtime->model_failed.load(std::memory_order_acquire) != 0
            ? "{\"error\":{\"message\":\"model load failed; check /health\",\"type\":\"model_load_failed\"}}"
            : "{\"error\":{\"message\":\"model is still loading; check /health\",\"type\":\"model_loading\"}}";
        return make_response(503, "application/json", body);
    }
    if (!engine) {
        return make_response(
            503,
            "application/json",
            "{\"error\":{\"message\":\"model is not ready; check /health\",\"type\":\"model_not_ready\"}}"
        );
    }
    if (req.method == "POST" && req.path == "/v1/chat/completions") {
        const server_generation_result generated = run_server_generation(engine, engine_mutex, opts, req.body);
        if (!generated.ok) {
            return make_response(generated.http_status, "application/json", make_openai_error_json(generated));
        }
        if (request_wants_stream(req.body)) {
            return make_response(200, "text/event-stream", make_chat_sse(opts, generated));
        }
        return make_response(200, "application/json", make_chat_json(opts, generated));
    }
    if (req.method == "POST" && req.path == "/v1/completions") {
        const server_generation_result generated = run_server_generation(engine, engine_mutex, opts, req.body);
        if (!generated.ok) {
            return make_response(generated.http_status, "application/json", make_openai_error_json(generated));
        }
        if (request_wants_stream(req.body)) {
            return make_response(200, "text/event-stream", make_completion_sse(opts, generated));
        }
        return make_response(200, "application/json", make_completion_json(opts, generated));
    }
    if (req.method == "POST" && req.path == "/v1/responses") {
        const server_generation_result generated = run_server_generation(engine, engine_mutex, opts, req.body);
        if (!generated.ok) {
            return make_response(generated.http_status, "application/json", make_openai_error_json(generated));
        }
        return make_response(200, "application/json", make_response_json(opts, generated));
    }
    if (req.method == "POST" &&
        (req.path == "/v1/storagellm/eval" ||
         req.path == "/v1/storagellm/perplexity" ||
         req.path == "/v1/perplexity")) {
        const server_eval_result evaluated = run_server_eval(engine, engine_mutex, opts, req.body);
        if (!evaluated.ok) {
            return make_response(evaluated.http_status, "application/json", make_eval_error_json(evaluated));
        }
        return make_response(200, "application/json", make_eval_json(opts, evaluated));
    }

    const std::string body = "{\"error\":{\"message\":\"unknown endpoint\",\"type\":\"not_found\"}}";
    return make_response(404, "application/json", body);
}

static void handle_client(
    socket_handle_t client,
    const server_options* opts,
    moe_pc_engine_t* engine,
    std::mutex* engine_mutex,
    const server_runtime_state* runtime = nullptr
) {
    std::unique_ptr<socket_handle_t, void(*)(socket_handle_t*)> guard(
        new socket_handle_t(client),
        [](socket_handle_t* s) {
            if (s) {
                close_socket_handle(*s);
                delete s;
            }
        });
    http_request req;
    std::string response;
    if (parse_http_request(client, &req)) {
        response = route_request(req, *opts, engine, engine_mutex, runtime);
    } else {
        response = make_response(
            400,
            "application/json",
            "{\"error\":{\"message\":\"bad request\",\"type\":\"bad_request\"}}"
        );
    }
    send_all(client, response);
}

static uint32_t max_http_workers(const moe_backend_caps_t& caps) {
    const uint32_t streams = caps.compute_streams + caps.copy_streams;
    return std::max<uint32_t>(2u, streams ? streams : 2u);
}

static socket_handle_t create_server_socket(const server_options& opts) {
    if (!opts.allow_remote && !is_loopback_host(opts.host)) {
        std::cerr << "Refusing non-loopback host without --allow-remote: " << opts.host << "\n";
        return invalid_socket_handle;
    }

    socket_handle_t server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == invalid_socket_handle) {
        return invalid_socket_handle;
    }

    int yes = 1;
#ifdef _WIN32
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(opts.port);
    const std::string bind_host = opts.host == "localhost" ? "127.0.0.1" : opts.host;
    if (inet_pton(AF_INET, bind_host.c_str(), &addr.sin_addr) != 1) {
        close_socket_handle(server);
        return invalid_socket_handle;
    }

    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket_handle(server);
        return invalid_socket_handle;
    }
    if (listen(server, 16) != 0) {
        close_socket_handle(server);
        return invalid_socket_handle;
    }
    return server;
}

static bool prepare_model_paths(server_options* opts) {
    if (!opts || opts->model_root.empty()) {
        return true;
    }
    if (opts->table_path.empty()) {
        const char* index_names[] = {"tensor_index.bin", "tensors.sltidx"};
        for (const char* name : index_names) {
            const std::string index = join_model_file(opts->model_root, name);
            if (file_exists_utf8(index)) {
                opts->table_path = index;
                break;
            }
        }
        if (opts->table_path.empty()) {
            const std::string table = join_model_file(opts->model_root, "tensors.csv");
            if (file_exists_utf8(table)) {
                opts->table_path = table;
            }
        }
    }
    if (opts->scale4_path.empty()) {
        const std::string scale4 = join_model_file(opts->model_root, "moe_scale4.gsc4");
        if (file_exists_utf8(scale4)) {
            opts->scale4_path = scale4;
        }
    }
    return true;
}

static void mark_model_load_failed(server_runtime_state* runtime, const std::string& message) {
    if (!runtime) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(runtime->error_mutex);
        runtime->error_message = message;
    }
    runtime->model_failed.store(1, std::memory_order_release);
}

static void set_model_load_stage(server_runtime_state* runtime, const char* stage) {
    if (!runtime || !stage) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(runtime->stage_mutex);
        runtime->load_stage = stage;
    }
    std::cerr << "[storagellm] load stage: " << stage << "\n" << std::flush;
}

static void cache_model_root_check(
    server_runtime_state* runtime,
    const std::string& model_root,
    const moe_model_root_check_t& root_check
) {
    if (!runtime) {
        return;
    }
    std::lock_guard<std::mutex> lock(runtime->root_check_mutex);
    runtime->root_check_model_root = model_root;
    runtime->root_check = root_check;
    runtime->root_check_ready = 1;
}

static bool get_cached_model_root_check(
    const server_runtime_state* runtime,
    const std::string& model_root,
    moe_model_root_check_t* out_check
) {
    if (!runtime || !out_check) {
        return false;
    }
    std::lock_guard<std::mutex> lock(runtime->root_check_mutex);
    if (!runtime->root_check_ready || runtime->root_check_model_root != model_root) {
        return false;
    }
    *out_check = runtime->root_check;
    return true;
}

static bool cleanup_background_engine_if_shutdown(server_runtime_state* runtime) {
    if (!runtime || runtime->shutdown_requested.load(std::memory_order_acquire) == 0) {
        return false;
    }
    set_model_load_stage(runtime, "shutdown");
    if (moe_pc_engine_t* engine = runtime->engine.exchange(nullptr, std::memory_order_acq_rel)) {
        moe_pc_engine_destroy(engine);
    }
    return true;
}

static void fail_model_load_and_destroy(server_runtime_state* runtime, const std::string& message) {
    mark_model_load_failed(runtime, message);
    set_model_load_stage(runtime, "failed");
    if (runtime) {
        if (moe_pc_engine_t* engine = runtime->engine.exchange(nullptr, std::memory_order_acq_rel)) {
            moe_pc_engine_destroy(engine);
        }
    }
}

static void load_server_model_background(
    server_options opts,
    moe_io_config_t io_config,
    server_runtime_state* runtime
) {
    if (!runtime) {
        return;
    }
    set_model_load_stage(runtime, "engine_create");
    moe_pc_engine_t* engine = moe_pc_engine_create(&opts.engine_config);
    if (!engine) {
        mark_model_load_failed(runtime, "engine create failed");
        std::cerr << "Engine create failed\n";
        return;
    }
    runtime->engine.store(engine, std::memory_order_release);
    if (cleanup_background_engine_if_shutdown(runtime)) {
        return;
    }
    set_model_load_stage(runtime, "configure_io");
    moe_pc_engine_configure_io(engine, &io_config);
    if (cleanup_background_engine_if_shutdown(runtime)) {
        return;
    }
    set_model_load_stage(runtime, "resolve_model_paths");
    if (!opts.model_root.empty()) {
        moe_pc_engine_set_model_root(engine, opts.model_root.c_str());
    }
    if (cleanup_background_engine_if_shutdown(runtime)) {
        return;
    }
    prepare_model_paths(&opts);
    {
        const char* root = opts.model_root.empty() ? nullptr : opts.model_root.c_str();
        const char* scale4 = opts.scale4_path.empty() ? nullptr : opts.scale4_path.c_str();
        if (opts.table_path.empty()) {
            set_model_load_stage(runtime, "load_offload_gguf_header");
            std::cerr << "[storagellm] no tensor index found; trying offload GGUF header index\n" << std::flush;
            if (!moe_pc_engine_load_codec_table(engine, nullptr, root, scale4)) {
                fail_model_load_and_destroy(
                    runtime,
                    "missing tensor index or offload GGUF metadata under model root"
                );
                std::cerr << "Missing tensor index or offload GGUF metadata under model root\n";
                return;
            }
            std::cerr << "[storagellm] offload GGUF header index loaded\n" << std::flush;
        } else {
            set_model_load_stage(runtime, "load_tensor_index");
            std::cerr << "[storagellm] loading tensor index: " << opts.table_path << "\n" << std::flush;
            if (!moe_pc_engine_load_codec_table(engine, opts.table_path.c_str(), root, scale4)) {
                fail_model_load_and_destroy(runtime, "failed to load tensor index: " + opts.table_path);
                std::cerr << "Failed to load tensor index: " << opts.table_path << "\n";
                return;
            }
            std::cerr << "[storagellm] tensor index loaded\n" << std::flush;
        }
    }
    if (cleanup_background_engine_if_shutdown(runtime)) {
        return;
    }
    if (!opts.topology_path.empty()) {
        set_model_load_stage(runtime, "load_topology");
        moe_pc_engine_load_topology(engine, opts.topology_path.c_str());
    }
    if (cleanup_background_engine_if_shutdown(runtime)) {
        return;
    }
    set_model_load_stage(runtime, "ready");
    runtime->model_ready.store(1, std::memory_order_release);
    print_optimization_plan(engine);
    set_model_load_stage(runtime, "startup_warm");
    std::cerr << "[storagellm] startup warm begin\n" << std::flush;
    const int warm_started = moe_pc_engine_startup_warm_model(engine);
    std::cerr << "[storagellm] startup warm " << (warm_started ? "queued" : "skipped") << "\n" << std::flush;
    set_model_load_stage(runtime, "ready");
    if (opts.engine_config.prefer_gpu && io_config.enable_common_raw_vram_pin) {
        set_model_load_stage(runtime, "warm_common_raw");
        moe_pc_engine_prefetch_common_raw(engine);
        set_model_load_stage(runtime, "ready");
    }
}

static void print_usage() {
    std::cout
        << "Usage:\n"
        << "  moe_pc_engine_server [model_root]\n"
        << "  moe_pc_engine_server --model-root path\n"
        << "\n"
        << "Backend, platform, RAM, VRAM, and local API mode are detected automatically.\n"
        << "QKV is the default KV cache path; qkv/--qkv is accepted for compatibility.\n"
        << "\n"
        << "Endpoints:\n"
        << "  GET  /health\n"
        << "  GET  /v1/models\n"
        << "  POST /v1/chat/completions\n"
        << "  POST /v1/completions\n"
        << "  POST /v1/responses\n"
        << "  POST /v1/storagellm/eval\n"
        << "  GET  /openclaw/config\n";
}

static server_options parse_args(int argc, char** argv) {
    server_options opts;
    opts.engine_config.allow_db_streaming = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--openclaw") == 0 || std::strcmp(argv[i], "--serve-openai") == 0) {
            continue;
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            opts.host = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            const int port = std::atoi(argv[++i]);
            if (port > 0 && port <= 65535) {
                opts.port = static_cast<uint16_t>(port);
            }
        } else if (std::strcmp(argv[i], "--model-id") == 0 && i + 1 < argc) {
            opts.model_id = argv[++i];
            opts.model_id_explicit = true;
        } else if (std::strcmp(argv[i], "--allow-remote") == 0) {
            opts.allow_remote = true;
        } else if (std::strcmp(argv[i], "--qkv") == 0 || std::strcmp(argv[i], "qkv") == 0) {
            opts.engine_config.kv_mode = moe_KV_MODE_QKV;
        } else if (std::strcmp(argv[i], "--topology") == 0 && i + 1 < argc) {
            opts.topology_path = argv[++i];
        } else if (std::strcmp(argv[i], "--table") == 0 && i + 1 < argc) {
            opts.table_path = argv[++i];
        } else if (std::strcmp(argv[i], "--model-root") == 0 && i + 1 < argc) {
            opts.model_root = argv[++i];
        } else if (std::strcmp(argv[i], "--scale4") == 0 && i + 1 < argc) {
            opts.scale4_path = argv[++i];
        } else if ((std::strcmp(argv[i], "--openclaw-config") == 0 ||
                    std::strcmp(argv[i], "--write-openclaw-config") == 0) && i + 1 < argc) {
            opts.openclaw_config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--openclaw-config-only") == 0) {
            opts.openclaw_config_only = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            std::exit(0);
        } else if (argv[i][0] != '-' && opts.model_root.empty()) {
            opts.model_root = argv[i];
        }
    }
    return opts;
}

static void apply_format_model_id(server_options* opts) {
    if (!opts || opts->model_id_explicit || opts->model_root.empty()) {
        return;
    }
    char inferred[160]{};
    if (moe_storage_infer_model_id(opts->model_root.c_str(), inferred, sizeof(inferred)) && inferred[0]) {
        opts->model_id = inferred;
    }
}

static int server_main(int argc, char** argv) {
    server_options opts = parse_args(argc, argv);
    apply_format_model_id(&opts);

    g_server_shutdown_requested = 0;
    std::signal(SIGINT, handle_server_shutdown_signal);
    std::signal(SIGTERM, handle_server_shutdown_signal);
#ifdef SIGBREAK
    std::signal(SIGBREAK, handle_server_shutdown_signal);
#endif

    if (!network_startup()) {
        std::cerr << "Network startup failed\n";
        return 1;
    }

    moe_backend_caps_t caps{};
    if (!moe_pc_detect_backend_caps(
            opts.engine_config.preferred_backend,
            opts.engine_config.platform,
            &caps)) {
        network_cleanup();
        std::cerr << "Backend capability detection failed\n";
        return 1;
    }
    apply_detected_budgets(&opts, caps);
    if (!write_openclaw_config_file(opts)) {
        network_cleanup();
        std::cerr << "Failed to write OpenClaw config: " << opts.openclaw_config_path << "\n";
        return 1;
    }
    if (!opts.openclaw_config_path.empty()) {
        std::cout << "OpenClaw config written: " << opts.openclaw_config_path << "\n";
        if (opts.openclaw_config_only) {
            network_cleanup();
            return 0;
        }
    }

    moe_io_config_t io_config = make_server_io_config(opts, caps);

    socket_handle_t server = create_server_socket(opts);
    if (server == invalid_socket_handle) {
        network_cleanup();
        std::cerr << "Failed to bind " << opts.host << ":" << opts.port << "\n";
        return 1;
    }
    server_runtime_state runtime;
    runtime.initial_caps = caps;
    set_model_load_stage(&runtime, "queued");
    moe_model_root_check_t startup_root_check{};
    const bool startup_root_check_ok =
        !opts.model_root.empty() &&
        moe_storage_validate_model_root(opts.model_root.c_str(), &startup_root_check);
    if (startup_root_check_ok) {
        cache_model_root_check(&runtime, opts.model_root, startup_root_check);
    }
    std::thread loader_thread(load_server_model_background, opts, io_config, &runtime);

    std::cout << "StorageLLM server listening on http://" << opts.host << ":" << opts.port << "/v1\n";
    std::cout << "Model: " << opts.model_id << "\n";
    std::cout << "Backend: " << moe_backend_name(caps.backend) << " / " << moe_platform_name(caps.platform) << "\n";
    std::cout << "Budgets: VRAM=" << opts.engine_config.vram_budget_bytes
              << " RAM=" << opts.engine_config.ram_budget_bytes
              << " KV=" << moe_kv_mode_name(opts.engine_config.kv_mode) << "\n";
    std::cout << "IO: disk=" << io_config.disk_worker_threads
              << " pinned=" << io_config.pinned_worker_threads
              << " gpu=" << io_config.gpu_worker_threads
              << " direct_min=" << io_config.direct_upload_min_bytes
              << " staging=" << io_config.pinned_staging_bytes << "\n";
    if (startup_root_check_ok) {
        std::cout << "Model root: valid=" << startup_root_check.valid
                  << " present=" << startup_root_check.present_part_count
                  << "/" << startup_root_check.expected_part_count
                  << " missing=" << startup_root_check.missing_part_count
                  << " size_mismatch=" << startup_root_check.size_mismatch_count << "\n";
    }
    if (!opts.topology_path.empty()) {
        std::cout << "Topology: " << opts.topology_path << "\n";
    }
    std::cout << "OpenClaw config: http://" << opts.host << ":" << opts.port << "/openclaw/config\n";

    std::mutex engine_mutex;
    const uint32_t worker_limit = max_http_workers(caps);
    auto active_clients = std::make_shared<std::atomic<uint32_t>>(0);
    while (!g_server_shutdown_requested) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server, &readfds);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 250000;
#ifdef _WIN32
        const int ready = select(0, &readfds, nullptr, nullptr, &timeout);
#else
        const int ready = select(server + 1, &readfds, nullptr, nullptr, &timeout);
#endif
        if (ready <= 0) {
            continue;
        }
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        socket_handle_t client = accept(server, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client == invalid_socket_handle) {
            continue;
        }

        while (active_clients->load() >= worker_limit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        active_clients->fetch_add(1);
        std::thread([client, &opts, &engine_mutex, active_clients, &runtime]() {
            handle_client(client, &opts, nullptr, &engine_mutex, &runtime);
            active_clients->fetch_sub(1);
        }).detach();
    }

    runtime.shutdown_requested.store(1, std::memory_order_release);
    close_socket_handle(server);
    while (active_clients->load(std::memory_order_acquire) > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (loader_thread.joinable()) {
        loader_thread.join();
    }
    if (moe_pc_engine_t* engine = runtime.engine.exchange(nullptr, std::memory_order_acq_rel)) {
        moe_pc_engine_destroy(engine);
    }
    network_cleanup();
    return 0;
}

#ifdef _WIN32
static std::string wide_to_utf8_arg(const wchar_t* value) {
    if (!value) {
        return std::string();
    }
    const int count = WideCharToMultiByte(CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (count <= 0) {
        return std::string();
    }
    std::string out((size_t)count, '\0');
    if (!WideCharToMultiByte(CP_UTF8, 0, value, -1, &out[0], count, nullptr, nullptr)) {
        return std::string();
    }
    if (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}

int wmain(int argc, wchar_t** wargv) {
    std::vector<std::string> args;
    std::vector<char*> argv8;
    args.reserve((size_t)argc);
    argv8.reserve((size_t)argc + 1);
    for (int i = 0; i < argc; ++i) {
        args.push_back(wide_to_utf8_arg(wargv[i]));
    }
    for (std::string& arg : args) {
        argv8.push_back(arg.empty() ? const_cast<char*>("") : &arg[0]);
    }
    argv8.push_back(nullptr);
    return server_main(argc, argv8.data());
}
#else
int main(int argc, char** argv) {
    return server_main(argc, argv);
}
#endif

