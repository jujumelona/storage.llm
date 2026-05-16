#include "moe_pc_engine.h"

// Model-agnostic: this function returns a zeroed shape.
// All model shape information is populated dynamically from the
// Graph IR runtime contract via engine->dynamic_shape.
// The moe_engine_model_shape_or_static() function in engine_state.cpp.inc
// handles the dynamic override path.
moe_model_shape_t moe_pc_Moe1_model_shape(void) {
    moe_model_shape_t shape{};
    return shape;
}
