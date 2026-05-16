#include "moe_pc_engine.h"

// Model-agnostic: no hardcoded layer table.
// Layer information is discovered at runtime from the Graph IR tensor index.
// The JUJU parser populates engine->dynamic_shape with layer/expert counts.
// IO prefetch uses the tensor record index (find_tensor_record) instead of
// this static table.

uint32_t moe_storage_layer_count(void) {
    return 0;
}

const moe_storage_layer_spec_t* moe_storage_layer_at(uint32_t layer) {
    (void)layer;
    return 0;
}
