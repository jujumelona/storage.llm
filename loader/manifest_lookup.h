#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace storagellm {

struct ProjectionBlocks {
    uint32_t weight_block = UINT32_MAX;
    uint32_t scale_block = UINT32_MAX;
    uint32_t scale2_block = UINT32_MAX;
    uint32_t aux0_block = UINT32_MAX;
    uint32_t aux1_block = UINT32_MAX;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 0;
    uint32_t group_size = 0;
    std::string scale_mode;
};

struct ExpertManifestEntry {
    uint32_t part = 0;
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t bundle_offset = 0;
    uint64_t bundle_length = 0;
    std::string part_path;
    ProjectionBlocks gate;
    ProjectionBlocks up;
    ProjectionBlocks down;
};

class ManifestLookup {
public:
    bool load(const char* manifest_path);
    bool find_expert(uint32_t layer, uint32_t expert, ExpertManifestEntry* out) const;
private:
    std::string text_;
    std::unordered_map<uint64_t, ExpertManifestEntry> expert_cache_;
};

}  // namespace storagellm
