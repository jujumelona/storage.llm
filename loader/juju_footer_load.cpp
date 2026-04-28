#include "juju_footer.h"

#include <string>

#include "file_read.h"
#include "json_scan.h"

namespace storagellm {

bool JujuFooter::load(const char* path) {
    text_.clear();
    block_cache_.clear();
    if (!read_text_file(path, &text_)) {
        return false;
    }
    size_t pos = text_.find("\"blocks\"");
    if (pos == std::string::npos) {
        return false;  // no "blocks" key — footer is malformed
    }
    // Advance past "blocks" key to the '[' array start so we only parse
    // "id" entries inside the blocks array, not unrelated earlier keys.
    pos = text_.find('[', pos);
    if (pos == std::string::npos) {
        return false;
    }
    while ((pos = text_.find("\"id\"", pos)) != std::string::npos) {
        const size_t begin = text_.rfind('{', pos);
        if (begin == std::string::npos) {
            pos += 4;  // skip this "id" and continue searching
            continue;  // malformed block, don't abort entire load
        }
        const size_t end = json_match_object(text_, begin);
        if (end == std::string::npos || end > text_.size()) {
            pos = begin + 1;  // resume after opening brace
            continue;  // malformed block, don't abort entire load
        }
        JsonSlice slice{&text_, begin, end};
        JujuFooterBlock block{};
        uint64_t value = 0;
        if (json_get_u64(slice, "id", &value) && value <= UINT32_MAX &&
            json_get_u64(slice, "offset", &block.offset) &&
            json_get_u64(slice, "length", &block.length)) {
            block.id = static_cast<uint32_t>(value);
            json_get_string(slice, "kind", &block.kind);
            json_get_string(slice, "key", &block.key);
            block_cache_[block.id] = block;
        }
        pos = end;
    }
    return true;
}

}  // namespace storagellm
