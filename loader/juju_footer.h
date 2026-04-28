#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace storagellm {

struct JujuFooterBlock {
    uint32_t id = UINT32_MAX;
    uint64_t offset = 0;
    uint64_t length = 0;
    std::string kind;
    std::string key;
};

class JujuFooter {
public:
    bool load(const char* path);
    bool find_block(uint32_t id, JujuFooterBlock* out) const;
    bool empty() const { return text_.empty(); }

private:
    std::string text_;
    std::unordered_map<uint32_t, JujuFooterBlock> block_cache_;
};

std::string juju_footer_name(uint32_t part);

}  // namespace storagellm
