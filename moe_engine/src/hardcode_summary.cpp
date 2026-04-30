#include "moe_pc_engine.h"

#include <mutex>
#include <string.h>

static moe_storage_model_state_t kMoeModelState;
static std::once_flag s_init_flag;

static void moe_build_model_state_impl(void) {
    const moe_storage_constants_t* c = moe_storage_constants();
    uint8_t seen[85];
    memset(seen, 0, sizeof(seen));
    kMoeModelState.part_count = moe_storage_part_count();
    kMoeModelState.shard_first = 1;
    kMoeModelState.shard_last = c->source_shard_count;
    kMoeModelState.scale4_count = c->scale4_count;
    kMoeModelState.raw_scale_count = c->raw_scale_count;
    kMoeModelState.raw_expert_scale_count = c->raw_expert_scale_count;
    for (uint32_t i = 0; i < kMoeModelState.part_count; ++i) {
        const moe_storage_part_spec_t* part = moe_storage_part_at(i);
        kMoeModelState.total_bytes += part->bytes;
        kMoeModelState.block_count += part->block_count;
        kMoeModelState.expert_bundle_count += part->expert_bundle_count;
        kMoeModelState.raw_expert_bundle_count += part->raw_expert_bundle_count;
        kMoeModelState.raw_tensor_count += part->raw_tensor_count;
        kMoeModelState.raw_tensor_bytes += part->raw_tensor_bytes;
        for (uint32_t n = 0; n < part->primary_count; ++n) {
            const uint32_t shard = part->primary_first + n;
            if (shard < 1 || shard > c->source_shard_count) {
                continue;
            }
            if (seen[shard]) {
                ++kMoeModelState.duplicate_primary_shard_count;
                if (!kMoeModelState.first_duplicate_primary_shard) {
                    kMoeModelState.first_duplicate_primary_shard = shard;
                }
            } else {
                seen[shard] = 1;
                ++kMoeModelState.shard_covered_count;
            }
        }
    }
    for (uint32_t shard = 1; shard <= c->source_shard_count; ++shard) {
        if (!seen[shard]) {
            ++kMoeModelState.missing_shard_count;
            if (!kMoeModelState.first_missing_shard) {
                kMoeModelState.first_missing_shard = shard;
            }
        }
    }
    kMoeModelState.matches_constants =
        kMoeModelState.part_count == c->physical_part_count &&
        c->logical_part_count >= c->physical_part_count &&
        kMoeModelState.shard_covered_count == c->source_shard_count &&
        kMoeModelState.total_bytes == c->file_bytes &&
        kMoeModelState.block_count == c->block_count &&
        kMoeModelState.expert_bundle_count == c->expert_bundle_count &&
        kMoeModelState.raw_expert_bundle_count == c->raw_expert_bundle_count &&
        kMoeModelState.raw_tensor_count == c->raw_tensor_count &&
        kMoeModelState.raw_tensor_bytes == c->raw_tensor_total_bytes &&
        kMoeModelState.missing_shard_count == 0 &&
        kMoeModelState.duplicate_primary_shard_count == 0;
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
