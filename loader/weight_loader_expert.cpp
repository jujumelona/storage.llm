#include "weight_loader.h"

namespace storagellm {

bool WeightLoader::load_expert(uint32_t layer, uint32_t expert, ExpertWeights* out) {
    if (!out) {
        return false;
    }
    // Critical Fix 1: Lock to prevent concurrent access race on active_part_
    std::lock_guard<std::mutex> lock(mtx_);

    ExpertManifestEntry entry{};
    if (!manifest_.find_expert(layer, expert, &entry)) {
        return false;
    }
    if (!open_part(entry.part, entry.part_path)) {
        return false;
    }
    *out = ExpertWeights{};
    out->layer = layer;
    out->expert = expert;
    return fill_projection(entry.gate, &out->gate) &&
           fill_projection(entry.up, &out->up) &&
           fill_projection(entry.down, &out->down);
}

}  // namespace storagellm
