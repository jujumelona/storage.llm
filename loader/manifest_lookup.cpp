#include "manifest_lookup.h"

#include <cstdlib>
#include <string>
#include <utility>

#include "file_read.h"
#include "json_scan.h"
#include "manifest_projection.h"

namespace storagellm {

std::string manifest_expert_key(uint32_t layer, uint32_t expert);

static uint64_t manifest_expert_cache_key(uint32_t layer, uint32_t expert) {
    return (static_cast<uint64_t>(layer) << 32) | expert;
}

static bool parse_u32_after_token(
    const std::string& text,
    size_t token_pos,
    size_t token_len,
    uint32_t* out,
    size_t* after_number
) {
    if (!out || token_pos == std::string::npos || token_pos + token_len >= text.size()) {
        return false;
    }
    const char* begin = text.c_str() + token_pos + token_len;
    char* endptr = nullptr;
    const unsigned long parsed = std::strtoul(begin, &endptr, 10);
    if (!endptr || endptr == begin || parsed > UINT32_MAX) {
        return false;
    }
    *out = static_cast<uint32_t>(parsed);
    if (after_number) {
        *after_number = static_cast<size_t>(endptr - text.c_str());
    }
    return true;
}

static bool parse_projection_alias_key(
    const std::string& key,
    uint32_t* layer,
    uint32_t* expert,
    const char** projection_name
) {
    if (!layer || !expert || !projection_name) {
        return false;
    }
    const size_t layer_pos = key.find("layers.");
    if (!parse_u32_after_token(key, layer_pos, 7, layer, nullptr)) {
        return false;
    }

    size_t expert_pos = key.find("experts.", layer_pos == std::string::npos ? 0 : layer_pos);
    size_t token_len = 8;
    if (expert_pos == std::string::npos) {
        expert_pos = key.find("expert_", layer_pos == std::string::npos ? 0 : layer_pos);
        token_len = 7;
    }
    if (!parse_u32_after_token(key, expert_pos, token_len, expert, nullptr)) {
        return false;
    }

    if (key.find("gate_proj", expert_pos) != std::string::npos ||
        key.find("w1", expert_pos) != std::string::npos) {
        *projection_name = "gate_proj";
        return true;
    }
    if (key.find("up_proj", expert_pos) != std::string::npos ||
        key.find("w3", expert_pos) != std::string::npos) {
        *projection_name = "up_proj";
        return true;
    }
    if (key.find("down_proj", expert_pos) != std::string::npos ||
        key.find("w2", expert_pos) != std::string::npos) {
        *projection_name = "down_proj";
        return true;
    }
    return false;
}

static void assign_projection_blocks(
    ExpertManifestEntry* entry,
    const char* projection_name,
    const ProjectionBlocks& blocks
) {
    if (!entry || !projection_name) {
        return;
    }
    const std::string proj(projection_name);
    if (proj == "gate_proj") {
        entry->gate = blocks;
    } else if (proj == "up_proj") {
        entry->up = blocks;
    } else if (proj == "down_proj") {
        entry->down = blocks;
    }
}

static bool projection_blocks_present(const ProjectionBlocks& blocks) {
    return blocks.weight_block != UINT32_MAX && blocks.rows > 0 && blocks.cols > 0;
}

static bool expert_entry_has_all_projections(const ExpertManifestEntry& entry) {
    return projection_blocks_present(entry.gate) &&
           projection_blocks_present(entry.up) &&
           projection_blocks_present(entry.down);
}

static bool parse_expert_entry(
    const JsonSlice& expert_obj,
    uint32_t layer,
    uint32_t expert,
    ExpertManifestEntry* out
) {
    if (!out) return false;
    uint64_t value = 0;
    out->layer = layer;
    out->expert = expert;
    if (json_get_u64(expert_obj, "part", &value)) {
        out->part = value > UINT32_MAX ?
            UINT32_MAX : static_cast<uint32_t>(value);
    }
    json_get_u64(expert_obj, "bundle_offset", &out->bundle_offset);
    json_get_u64(expert_obj, "bundle_length", &out->bundle_length);
    json_get_string(expert_obj, "part_path", &out->part_path);
    JsonSlice projections{};
    if (!json_find_member_object(expert_obj, "projections", &projections)) {
        return false;
    }
    JsonSlice gate{}, up{}, down{};
    return json_find_member_object(projections, "gate_proj", &gate) &&
           json_find_member_object(projections, "up_proj", &up) &&
           json_find_member_object(projections, "down_proj", &down) &&
           parse_projection_blocks(gate, &out->gate) &&
           parse_projection_blocks(up, &out->up) &&
           parse_projection_blocks(down, &out->down);
}

static void load_l_dot_e_entries(
    const std::string& text,
    std::unordered_map<uint64_t, ExpertManifestEntry>* expert_cache
) {
    if (!expert_cache) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find("\"L", pos)) != std::string::npos) {
        const size_t key_begin = pos + 2;
        char* endptr = nullptr;
        unsigned long layer_ul = std::strtoul(text.c_str() + key_begin, &endptr, 10);
        if (layer_ul > UINT32_MAX) {
            ++pos;
            continue;
        }
        const uint32_t layer = static_cast<uint32_t>(layer_ul);
        if (!endptr || endptr[0] != '.' || endptr[1] != 'E') {
            ++pos;
            continue;
        }
        char* expert_end = nullptr;
        unsigned long expert_ul = std::strtoul(endptr + 2, &expert_end, 10);
        if (expert_ul > UINT32_MAX) {
            ++pos;
            continue;
        }
        const uint32_t expert = static_cast<uint32_t>(expert_ul);
        if (!expert_end || expert_end[0] != '"') {
            ++pos;
            continue;
        }
        if (expert_end < text.c_str() ||
            static_cast<size_t>(expert_end - text.c_str()) > text.size()) {
            ++pos;
            continue;
        }
        const size_t object_begin = text.find(
            '{', static_cast<size_t>(expert_end - text.c_str()));
        if (object_begin == std::string::npos) {
            ++pos;
            continue;
        }
        const size_t object_end = json_match_object(text, object_begin);
        if (object_end == std::string::npos || object_end > text.size() || object_end <= object_begin) {
            pos = object_begin + 1;
            continue;
        }
        ExpertManifestEntry entry{};
        JsonSlice slice{&text, object_begin, object_end};
        if (parse_expert_entry(slice, layer, expert, &entry)) {
            (*expert_cache)[manifest_expert_cache_key(layer, expert)] = std::move(entry);
        }
        pos = object_end;
    }
}

static void load_tensor_alias_entries(
    const std::string& text,
    std::unordered_map<uint64_t, ExpertManifestEntry>* expert_cache
) {
    if (!expert_cache) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        const size_t key_begin = pos + 1;
        const size_t key_end = text.find('"', key_begin);
        if (key_end == std::string::npos) {
            break;
        }
        const std::string key = text.substr(key_begin, key_end - key_begin);
        uint32_t layer = 0;
        uint32_t expert = 0;
        const char* projection_name = nullptr;
        if (!parse_projection_alias_key(key, &layer, &expert, &projection_name)) {
            pos = key_end + 1;
            continue;
        }
        const size_t colon = text.find_first_not_of(" \t\r\n", key_end + 1);
        if (colon == std::string::npos || colon >= text.size() || text[colon] != ':') {
            pos = key_end + 1;
            continue;
        }
        const size_t object_begin = text.find('{', colon + 1);
        if (object_begin == std::string::npos) {
            pos = key_end + 1;
            continue;
        }
        const size_t object_end = json_match_object(text, object_begin);
        if (object_end == std::string::npos || object_end > text.size() || object_end <= object_begin) {
            pos = object_begin + 1;
            continue;
        }

        ProjectionBlocks blocks{};
        JsonSlice slice{&text, object_begin, object_end};
        if (parse_projection_blocks(slice, &blocks)) {
            const uint64_t cache_key = manifest_expert_cache_key(layer, expert);
            ExpertManifestEntry& entry = (*expert_cache)[cache_key];
            entry.layer = layer;
            entry.expert = expert;
            uint64_t value = 0;
            if (json_get_u64(slice, "part", &value)) {
                entry.part = value > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(value);
            }
            json_get_u64(slice, "bundle_offset", &entry.bundle_offset);
            json_get_u64(slice, "bundle_length", &entry.bundle_length);
            std::string part_path;
            if (json_get_string(slice, "part_path", &part_path) || json_get_string(slice, "file", &part_path)) {
                entry.part_path = part_path;
            }
            assign_projection_blocks(&entry, projection_name, blocks);
        }
        pos = object_end;
    }
}

bool ManifestLookup::load(const char* manifest_path) {
    expert_cache_.clear();
    if (!manifest_path) {
        return false;
    }
    if (!read_text_file(manifest_path, &text_)) {
        return false;
    }
    load_l_dot_e_entries(text_, &expert_cache_);
    load_tensor_alias_entries(text_, &expert_cache_);
    return true;
}

bool ManifestLookup::find_expert(
    uint32_t layer,
    uint32_t expert,
    ExpertManifestEntry* out
) const {
    if (!out || text_.empty()) {
        return false;
    }
    auto cached = expert_cache_.find(manifest_expert_cache_key(layer, expert));
    if (cached != expert_cache_.end() && expert_entry_has_all_projections(cached->second)) {
        *out = cached->second;
        return true;
    }
    return false;
}

}  // namespace storagellm
