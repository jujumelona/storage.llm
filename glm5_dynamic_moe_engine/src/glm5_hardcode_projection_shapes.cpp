#include "glm5_pc_engine.h"

static const glm5_projection_shape_spec_t kGlm5ProjectionShapes[] = {
    {GLM5_PROJ_GATE, 2048, 6144, 384, 16},
    {GLM5_PROJ_UP, 2048, 6144, 384, 16},
    {GLM5_PROJ_DOWN, 6144, 2048, 128, 16}
};

const glm5_projection_shape_spec_t* glm5_storage_projection_shape(glm5_projection_t proj) {
    if ((uint32_t)proj >= 3) {
        return 0;
    }
    return &kGlm5ProjectionShapes[(uint32_t)proj];
}
