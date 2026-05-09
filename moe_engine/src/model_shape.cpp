#include "moe_pc_engine.h"

namespace {

bool storage_constants_shape_fields_valid(const moe_storage_constants_t* c) {
    if (!c) {
        return false;
    }
    if (c->num_hidden_layers == 0 || c->hidden_size == 0 ||
        c->expert_intermediate_size == 0 || c->vocab_size == 0) {
        return false;
    }
    if (c->first_moe_layer > c->last_moe_layer ||
        c->last_moe_layer >= c->num_hidden_layers) {
        return false;
    }
    if (c->experts_per_moe_layer == 0 || c->expert_projection_count == 0) {
        return false;
    }
    if (c->num_hidden_layers > 10000 || c->experts_per_moe_layer > 10000 ||
        c->hidden_size > 1000000 || c->expert_intermediate_size > 1000000 ||
        c->vocab_size > 10000000 || c->expert_projection_count > 1024) {
        return false;
    }

    const uint64_t moe_layer_count =
        static_cast<uint64_t>(c->last_moe_layer - c->first_moe_layer + 1u);
    const uint64_t expected_experts =
        moe_layer_count * static_cast<uint64_t>(c->experts_per_moe_layer);
    if (c->total_expert_count != 0 && c->total_expert_count != expected_experts) {
        return false;
    }
    return true;
}

}  // namespace

moe_model_shape_t moe_model_shape_from_storage_constants(const moe_storage_constants_t* c) {
    moe_model_shape_t shape{};
    if (!storage_constants_shape_fields_valid(c)) {
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

moe_model_shape_t moe_pc_Moe1_model_shape(void) {
    return moe_model_shape_from_storage_constants(moe_storage_constants());
}
