#include "manifest_projection.h"
#include "json_scan.h"

namespace storagellm {

bool parse_projection_blocks(const JsonSlice& proj, ProjectionBlocks* out) {
    uint64_t value = 0;
    if (!out || !json_get_u64(proj, "weight_block", &value)) {
        return false;
    }
    out->weight_block = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "raw_scale_block", &value)) out->scale_block = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "raw_scale2_block", &value)) out->scale2_block = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "aux0_block", &value)) out->aux0_block = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "aux1_block", &value)) out->aux1_block = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "rows", &value)) out->rows = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "cols", &value)) out->cols = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "groups", &value)) out->groups = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    if (json_get_u64(proj, "group_size", &value)) out->group_size = value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
    json_get_string(proj, "scale_mode", &out->scale_mode);
    return true;
}

}  // namespace storagellm
