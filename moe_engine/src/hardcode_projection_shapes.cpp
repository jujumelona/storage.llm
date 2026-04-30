#include "moe_pc_engine.h"

static const moe_projection_shape_spec_t kMoeProjectionShapes[] = {
    {moe_PROJ_GATE, 2048, 6144, 384, 16},
    {moe_PROJ_UP, 2048, 6144, 384, 16},
    {moe_PROJ_DOWN, 6144, 2048, 128, 16}
};

const moe_projection_shape_spec_t* moe_storage_projection_shape(moe_projection_t proj) {
    if ((uint32_t)proj >= 3) {
        return 0;
    }
    return &kMoeProjectionShapes[(uint32_t)proj];
}
