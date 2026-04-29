#include "juju_footer.h"
#include "json_scan.h"

namespace storagellm {

static bool block_from_slice(const JsonSlice& slice, JujuFooterBlock* out) {
    uint64_t value = 0;
    if (!json_get_u64(slice, "id", &value) || value > UINT32_MAX) return false;
    out->id = static_cast<uint32_t>(value);
    if (!json_get_u64(slice, "offset", &out->offset)) return false;
    if (!json_get_u64(slice, "length", &out->length)) return false;
    json_get_string(slice, "kind", &out->kind);
    json_get_string(slice, "key", &out->key);
    return true;
}

bool JujuFooter::find_block(uint32_t id, JujuFooterBlock* out) const {
    if (!out) return false;
    auto cached = block_cache_.find(id);
    if (cached != block_cache_.end()) {
        *out = cached->second;
        return true;
    }
    if (text_.empty()) return false;
    size_t pos = text_.find("\"blocks\"");
    if (pos == std::string::npos) return false;
    pos = text_.find('[', pos);
    if (pos == std::string::npos) return false;

    const size_t array_end = text_.find(']', pos);
    if (array_end == std::string::npos) return false;

    // Bug 1: Use forward scan to avoid infinite loop with nested objects
    while ((pos = text_.find('{', pos)) != std::string::npos && pos < array_end) {
        const size_t end = json_match_object(text_, pos);
        if (end == std::string::npos) {
            pos++;
            continue;
        }
        JsonSlice slice{&text_, pos, end};
        JujuFooterBlock block{};
        if (block_from_slice(slice, &block) && block.id == id) {
            *out = block;
            // Bug 2: Cache the result to avoid repeated scans
            block_cache_[id] = block;
            return true;
        }
        pos = end;
    }
    return false;
}

}  // namespace storagellm
