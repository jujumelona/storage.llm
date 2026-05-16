#include "moe_pc_engine.h"

// Model-agnostic: no hardcoded raw span table.
// Raw spans (file offsets for tensor regions) are discovered at runtime
// from the JUJU index file and mmap'd tensor records.
// The Graph IR forward path uses find_raw_role_span() which iterates
// this table — with 0 entries it returns null, causing the forward path
// to use the mmap-based tensor access path instead (which is the correct
// path for JUJU format models).

uint32_t moe_storage_raw_span_count(void) {
    return 0;
}

const moe_storage_raw_span_spec_t* moe_storage_raw_span_at(uint32_t index) {
    (void)index;
    return 0;
}

uint32_t moe_storage_raw_span_find_first(int32_t layer) {
    (void)layer;
    return 0;
}

uint32_t moe_storage_raw_span_find_end(int32_t layer) {
    (void)layer;
    return 0;
}

uint32_t moe_storage_attention_span_index(int32_t layer, uint32_t kind_index) {
    (void)layer;
    (void)kind_index;
    return 0xFFFFFFFFu;
}

uint32_t moe_storage_router_span_index(int32_t layer) {
    (void)layer;
    return 0xFFFFFFFFu;
}
