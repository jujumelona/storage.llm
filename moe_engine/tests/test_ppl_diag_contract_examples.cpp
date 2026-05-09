// PPL diagnostics contract tests
// These tests compile the diagnostics hooks in enabled mode and pin the
// production call signatures used by eval/generation/QKV paths.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

#include "../../engine_core/kv/kv_qkv.h"

#undef STORAGELLM_PPL_DIAG
#define STORAGELLM_PPL_DIAG 1
#ifndef STORAGELLM_PPL_DIAG_ENABLED
#define STORAGELLM_PPL_DIAG_ENABLED 1
#endif

#include "../src/parts/ppl_diag_hooks.cpp.inc"

namespace {

template <typename Actual, typename Expected>
constexpr bool same_signature = std::is_same<Actual, Expected>::value;

}  // namespace

TEST(PPLDiagContractExamples, EnabledBuildExportsCallSiteGate) {
#ifdef STORAGELLM_PPL_DIAG_ENABLED
    EXPECT_EQ(STORAGELLM_PPL_DIAG_ENABLED, 1);
#else
    FAIL() << "STORAGELLM_PPL_DIAG_ENABLED must be defined when diagnostics are enabled";
#endif
}

TEST(PPLDiagContractExamples, HookSignaturesMatchProductionCallSites) {
    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_forward_vec),
        void (*)(const char*, uint32_t, uint32_t, const float*, uint32_t)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_lm_head_begin),
        void (*)(uint32_t, uint32_t, uint32_t, uint8_t, uint8_t, uint8_t,
                 uint64_t, uint64_t, float, const float*)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_lm_head_end),
        void (*)(uint32_t, float, uint32_t, float, uint64_t, double,
                 float, double, double, float)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_attention_config),
        void (*)(const char*, int32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t, int, float, float)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_qkv_block),
        void (*)(const char*, int32_t, uint32_t, const float*, uint32_t,
                 const float*, uint32_t, const float*, uint32_t, uint32_t,
                 uint32_t, uint32_t, uint32_t)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_qkv_append_probe),
        void (*)(uint32_t, uint32_t, uint32_t, uint32_t, const float*,
                 const float*, const qkv_state_t*, const char*)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_qkv_decode_enter),
        void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 float, float, const float*, const qkv_state_t*, const qkv_config_t*)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_qkv_decode_failed),
        void (*)(uint32_t, uint32_t, uint32_t, const float*, const qkv_state_t*,
                 const qkv_config_t*, uint32_t, uint32_t, uint32_t, float, float)>);

    static_assert(same_signature<
        decltype(&storagellm_ppl_diag_attention_heads_failed),
        void (*)(int32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                 uint32_t, int, float, float)>);

    SUCCEED();
}

TEST(PPLDiagContractExamples, EnabledHooksAcceptRepresentativeFiniteInputs) {
    float q[] = {1.0f, -2.0f, 3.0f, -4.0f};
    float k[] = {0.5f, 0.25f, -0.75f, 1.25f};
    float v[] = {2.0f, 0.0f, -1.0f, 0.5f};

    qkv_state_t state{};
    state.head_dim = 4;
    state.n_tokens = 1;
    state.sink_tokens = 1;
    state.k_sink = k;
    state.v_sink = v;

    qkv_config_t cfg = qkv_config_default(4);

    storagellm_ppl_diag_forward_vec("unit", 0u, 0u, q, 4u);
    storagellm_ppl_diag_lm_head_begin(1u, 8u, 4u, 0u, 0u, 0u, 16u, 128u, 0.0f, q);
    storagellm_ppl_diag_lm_head_end(1u, 0.5f, 2u, 1.0f, 2ull, 0.5, 1.0f, 2.0, -1.5, 0.0f);
    storagellm_ppl_diag_attention_config(
        "unit", 0, 0u, 4u, 1u, 1u, 4u, 4u, 4u, 4u, 4u, 1u, 1u, 4u, 4u, 1, 0.5f, 0.0f);
    storagellm_ppl_diag_qkv_block("unit", 0, 0u, q, 4u, k, 4u, v, 4u, 1u, 1u, 4u, 4u);
    storagellm_ppl_diag_qkv_append_probe(0u, 0u, 0u, 4u, k, v, &state, "unit");
    storagellm_ppl_diag_qkv_decode_enter(0u, 0u, 0u, 1u, 1u, 4u, 0.5f, 0.0f, q, &state, &cfg);
    storagellm_ppl_diag_qkv_decode_failed(0u, 0u, 4u, q, &state, &cfg, 1u, 1u, 4u, 0.5f, 0.0f);
    storagellm_ppl_diag_attention_heads_failed(0, 0u, 1u, 1u, 1u, 4u, 4u, 1, 0.5f, 0.0f);

    SUCCEED();
}
