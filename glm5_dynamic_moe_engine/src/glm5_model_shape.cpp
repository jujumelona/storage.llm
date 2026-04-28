#include "glm5_pc_engine.h"

glm5_model_shape_t glm5_pc_glm51_model_shape(void) {
    glm5_model_shape_t shape{};
    const glm5_storage_constants_t* c = glm5_storage_constants();
    if (!c) {
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

