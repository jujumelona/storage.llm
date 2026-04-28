#include "glm5_pc_engine.h"

#include <mutex>
#include <string.h>

static glm5_storage_model_state_t kGlm5ModelState;
static std::once_flag s_init_flag;

static void glm5_build_model_state_impl(void) {
    const glm5_storage_constants_t* c = glm5_storage_constants();
    uint8_t seen[85];
    memset(seen, 0, sizeof(seen));
    kGlm5ModelState.part_count = glm5_storage_part_count();
    kGlm5ModelState.shard_first = 1;
    kGlm5ModelState.shard_last = c->source_shard_count;
    kGlm5ModelState.scale4_count = c->scale4_count;
    kGlm5ModelState.raw_scale_count = c->raw_scale_count;
    kGlm5ModelState.raw_expert_scale_count = c->raw_expert_scale_count;
    for (uint32_t i = 0; i < kGlm5ModelState.part_count; ++i) {
        const glm5_storage_part_spec_t* part = glm5_storage_part_at(i);
        kGlm5ModelState.total_bytes += part->bytes;
        kGlm5ModelState.block_count += part->block_count;
        kGlm5ModelState.expert_bundle_count += part->expert_bundle_count;
        kGlm5ModelState.raw_expert_bundle_count += part->raw_expert_bundle_count;
        kGlm5ModelState.raw_tensor_count += part->raw_tensor_count;
        kGlm5ModelState.raw_tensor_bytes += part->raw_tensor_bytes;
        for (uint32_t n = 0; n < part->primary_count; ++n) {
            const uint32_t shard = part->primary_first + n;
            if (shard < 1 || shard > c->source_shard_count) {
                continue;
            }
            if (seen[shard]) {
                ++kGlm5ModelState.duplicate_primary_shard_count;
                if (!kGlm5ModelState.first_duplicate_primary_shard) {
                    kGlm5ModelState.first_duplicate_primary_shard = shard;
                }
            } else {
                seen[shard] = 1;
                ++kGlm5ModelState.shard_covered_count;
            }
        }
    }
    for (uint32_t shard = 1; shard <= c->source_shard_count; ++shard) {
        if (!seen[shard]) {
            ++kGlm5ModelState.missing_shard_count;
            if (!kGlm5ModelState.first_missing_shard) {
                kGlm5ModelState.first_missing_shard = shard;
            }
        }
    }
    kGlm5ModelState.matches_constants =
        kGlm5ModelState.part_count == c->physical_part_count &&
        c->logical_part_count >= c->physical_part_count &&
        kGlm5ModelState.shard_covered_count == c->source_shard_count &&
        kGlm5ModelState.total_bytes == c->file_bytes &&
        kGlm5ModelState.block_count == c->block_count &&
        kGlm5ModelState.expert_bundle_count == c->expert_bundle_count &&
        kGlm5ModelState.raw_expert_bundle_count == c->raw_expert_bundle_count &&
        kGlm5ModelState.raw_tensor_count == c->raw_tensor_count &&
        kGlm5ModelState.raw_tensor_bytes == c->raw_tensor_total_bytes &&
        kGlm5ModelState.missing_shard_count == 0 &&
        kGlm5ModelState.duplicate_primary_shard_count == 0;
}

static void glm5_build_model_state(void) {
    std::call_once(s_init_flag, glm5_build_model_state_impl);
}

const glm5_storage_model_state_t* glm5_storage_model_state_summary(void) {
    glm5_build_model_state();
    return &kGlm5ModelState;
}

int glm5_storage_model_state_valid(void) {
    return glm5_storage_model_state_summary()->matches_constants;
}
