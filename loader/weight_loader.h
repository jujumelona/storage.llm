#pragma once

#include "juju_footer.h"
#include "juju_part.h"
#include "manifest_lookup.h"

#include <cstdint>
#include <string>
#include <mutex>

namespace storagellm {

struct StorageBlockSlice {
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;
    uint32_t block_id = UINT32_MAX;
    std::string kind;
    std::string key;
};

struct ProjectionWeights {
    StorageBlockSlice weight;
    StorageBlockSlice scale;
    StorageBlockSlice scale2;
    StorageBlockSlice aux0;
    StorageBlockSlice aux1;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 0;
    uint32_t group_size = 0;
    std::string scale_mode;
};

struct ExpertWeights {
    uint32_t layer = 0;
    uint32_t expert = 0;
    ProjectionWeights gate;
    ProjectionWeights up;
    ProjectionWeights down;
};

class WeightLoader {
public:
    bool open(const char* model_root, const char* manifest_path, const char* footer_root);
    void close();
    bool load_expert(uint32_t layer, uint32_t expert, ExpertWeights* out);

private:
    bool open_part(uint32_t part, const std::string& rel_path);
    bool fill_block(uint32_t block_id, StorageBlockSlice* out);
    bool fill_projection(const ProjectionBlocks& spec, ProjectionWeights* out);
    ManifestLookup manifest_;
    JujuPart part_;
    JujuFooter footer_;
    std::string model_root_;
    std::string footer_root_;
    uint32_t active_part_ = UINT32_MAX;
    // Critical Fix 1: Thread safety for concurrent io_worker access
    // Multiple io_workers calling load_expert() simultaneously cause race on active_part_,
    // leading to wrong weights loaded into VRAM (data corruption).
    std::mutex mtx_;
};

}  // namespace storagellm
