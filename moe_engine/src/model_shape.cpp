#include "moe_pc_engine.h"

moe_model_shape_t moe_pc_Moe1_model_shape(void) {
    moe_model_shape_t shape{};
    const moe_storage_constants_t* c = moe_storage_constants();
    // BUGFIX 446: c null 체크 강화
    if (!c) {
        return shape;
    }
    // BUGFIX 447: 값 범위 체크
    if (c->num_hidden_layers > 10000 || c->experts_per_moe_layer > 10000 ||
        c->hidden_size > 1000000 || c->expert_intermediate_size > 1000000 ||
        c->vocab_size > 10000000) {
        return shape;
    }
    shape.num_hidden_layers = c->num_hidden_layers;
    shape.first_moe_layer = c->first_moe_layer;
    shape.last_moe_layer = c->last_moe_layer;
    shape.experts_per_moe_layer = c->experts_per_moe_layer;
    shape.hidden_size = c->hidden_size;
    shape.expert_intermediate_size = c->expert_intermediate_size;
    shape.vocab_size = c->vocab_size;
    shape.projection_count = c->expert_projection_count;
    return shape;
}

