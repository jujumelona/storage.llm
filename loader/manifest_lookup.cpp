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

bool ManifestLookup::load(const char* manifest_path) {
    expert_cache_.clear();
    if (!read_text_file(manifest_path, &text_)) {
        return false;
    }
    size_t pos = 0;
    while ((pos = text_.find("\"L", pos)) != std::string::npos) {
        const size_t key_begin = pos + 2;
        char* endptr = nullptr;
        const uint32_t layer = static_cast<uint32_t>(
            std::strtoul(text_.c_str() + key_begin, &endptr, 10));
        if (!endptr || endptr[0] != '.' || endptr[1] != 'E') {
            ++pos;
            continue;
        }
        char* expert_end = nullptr;
        const uint32_t expert = static_cast<uint32_t>(
            std::strtoul(endptr + 2, &expert_end, 10));
        if (!expert_end || expert_end[0] != '"') {
            ++pos;
            continue;
        }
        const size_t object_begin = text_.find(
            '{', static_cast<size_t>(expert_end - text_.c_str()));
        if (object_begin == std::string::npos) {
            ++pos;
            continue;  // malformed entry, skip and keep scanning
        }
        const size_t object_end = json_match_object(text_, object_begin);
        if (object_end == std::string::npos || object_end > text_.size()) {
            pos = object_begin + 1;
            continue;  // malformed entry, skip and keep scanning
        }
        ExpertManifestEntry entry{};
        JsonSlice slice{&text_, object_begin, object_end};
        if (parse_expert_entry(slice, layer, expert, &entry)) {
            expert_cache_[manifest_expert_cache_key(layer, expert)] = std::move(entry);
        }
        pos = object_end;
    }
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
    if (cached != expert_cache_.end()) {
        *out = cached->second;
        return true;
    }
    JsonSlice expert_obj{};
    if (!json_find_key_object(text_, manifest_expert_key(layer, expert), &expert_obj)) {
        return false;
    }
    return parse_expert_entry(expert_obj, layer, expert, out);
}

}  // namespace storagellm
