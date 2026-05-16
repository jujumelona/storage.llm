#include "moe_pc_engine.h"

// Model-agnostic: no hardcoded projection shapes.
// Projection dimensions are discovered from Graph IR tensor records at runtime.
// The expert_triplet_cpu path uses require_expert_projection() which validates
// actual tensor dimensions from the JUJU index, not from this static table.

const moe_projection_shape_spec_t* moe_storage_projection_shape(moe_projection_t proj) {
    (void)proj;
    return 0;
}
