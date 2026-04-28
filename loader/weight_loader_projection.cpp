#include "weight_loader.h"

namespace storagellm {

bool WeightLoader::fill_projection(
    const ProjectionBlocks& spec,
    ProjectionWeights* out
) {
    if (!out) {
        return false;
    }
    *out = ProjectionWeights{};
    out->rows = spec.rows;
    out->cols = spec.cols;
    out->groups = spec.groups;
    out->group_size = spec.group_size;
    out->scale_mode = spec.scale_mode;
    return fill_block(spec.weight_block, &out->weight) &&
           fill_block(spec.scale_block, &out->scale) &&
           fill_block(spec.scale2_block, &out->scale2) &&
           fill_block(spec.aux0_block, &out->aux0) &&
           fill_block(spec.aux1_block, &out->aux1);
}

}  // namespace storagellm
