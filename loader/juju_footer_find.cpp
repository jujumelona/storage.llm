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
    while ((pos = text_.find("\"id\"", pos)) != std::string::npos) {
        const size_t begin = text_.rfind('{', pos);
        // Bug 6: Don't abort entire search if one block has no opening brace
        if (begin == std::string::npos) {
            pos += 4;  // Skip this "id" and continue searching
            continue;
        }
        const size_t end = json_match_object(text_, begin);
        if (end == std::string::npos || end > text_.size()) {
            pos = begin + 1;  // Continue from after the brace
            continue;
        }
        JsonSlice slice{&text_, begin, end};
        JujuFooterBlock block{};
        if (block_from_slice(slice, &block) && block.id == id) {
            *out = block;
            return true;
        }
        pos = end;
    }
    return false;
}

}  // namespace storagellm
