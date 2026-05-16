#include "moe_pc_engine.h"

#include <mutex>
#include <string.h>

static moe_storage_model_state_t kMoeModelState;
static std::once_flag s_init_flag;

static void moe_build_model_state_impl(void) {
    // Model-agnostic: with no hardcoded parts or layers, the summary
    // simply reports zero counts.  The engine uses the dynamic Graph IR
    // contract and JUJU tensor index for actual model validation.
    memset(&kMoeModelState, 0, sizeof(kMoeModelState));
    kMoeModelState.matches_constants = 1;  // No static data to mismatch
}

static void moe_build_model_state(void) {
    std::call_once(s_init_flag, moe_build_model_state_impl);
}

const moe_storage_model_state_t* moe_storage_model_state_summary(void) {
    moe_build_model_state();
    return &kMoeModelState;
}

int moe_storage_model_state_valid(void) {
    return moe_storage_model_state_summary()->matches_constants;
}
