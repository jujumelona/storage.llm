// QKV contract example tests
// These tests use small deterministic numbers to prove that example GGUF QKV
// fields, packed indices, quantized KV rows, and attention decode fail closed
// instead of crashing or silently accepting invalid contracts.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../../engine_core/kv/kv_qkv.h"
#include "../../engine_core/kv/qkv_codebook.h"
#include "../../engine_core/kv/qkv_helpers.h"
#include "../../engine_core/kv/qkv_matrix.h"
#include "../../engine_core/kv/qkv_packing.h"

namespace {

constexpr int kHeadDim = 8;
constexpr int kTokens = 4;

std::vector<float> make_example_keys() {
    return {
        0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f, 0.70f, -0.80f,
        0.15f, -0.25f, 0.35f, -0.45f, 0.55f, -0.65f, 0.75f, -0.85f,
        0.05f, 0.10f, -0.15f, -0.20f, 0.25f, 0.30f, -0.35f, -0.40f,
        0.90f, 0.70f, 0.50f, 0.30f, 0.10f, -0.10f, -0.30f, -0.50f,
    };
}

std::vector<float> make_example_values() {
    return {
        -0.05f, 0.10f, -0.15f, 0.20f, -0.25f, 0.30f, -0.35f, 0.40f,
        0.45f, -0.40f, 0.35f, -0.30f, 0.25f, -0.20f, 0.15f, -0.10f,
        0.12f, 0.24f, 0.36f, 0.48f, -0.12f, -0.24f, -0.36f, -0.48f,
        -0.60f, -0.30f, 0.00f, 0.30f, 0.60f, 0.90f, 1.20f, 1.50f,
    };
}

void expect_all_finite(const std::vector<float>& values) {
    for (float v : values) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

double mse(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return acc / static_cast<double>(a.size());
}

double max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    double out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        out = std::max(out, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return out;
}

void expect_roundtrip_within(
    const std::vector<float>& original,
    const std::vector<float>& restored,
    double max_mse,
    double max_error
) {
    expect_all_finite(restored);
    EXPECT_LE(mse(original, restored), max_mse);
    EXPECT_LE(max_abs_error(original, restored), max_error);
}

}  // namespace

TEST(QKVContractExamples, OffloadKvExampleNumbersMapIntoConfig) {
    // Representative values from the offload.* GGUF KV namespace.
    qkv_config_t cfg = qkv_config_default(128);
    cfg.k_bits = 3;
    cfg.v_bits = 2;
    cfg.group_size = 64;
    cfg.page_size_tokens = 16;
    cfg.sink_tokens = 4;
    cfg.rotation_seed = 42;
    cfg.qjl_seed = 43;
    cfg.enable_qjl = true;
    cfg.enable_rotation = true;
    cfg.outlier_channels = 32;
    cfg.key_outlier_bits = 3;
    cfg.key_normal_bits = 2;
    cfg.value_outlier_bits = 3;
    cfg.value_normal_bits = 2;
    cfg.plain_kv_persistent_storage = false;

    EXPECT_EQ(cfg.k_bits, 3);
    EXPECT_EQ(cfg.v_bits, 2);
    EXPECT_EQ(cfg.group_size, 64u);
    EXPECT_EQ(cfg.page_size_tokens, 16u);
    EXPECT_EQ(cfg.sink_tokens, 4u);
    EXPECT_EQ(cfg.rotation_seed, 42u);
    EXPECT_EQ(cfg.qjl_seed, 43u);
    EXPECT_FALSE(cfg.plain_kv_persistent_storage);

    // 32 channels @ 3 bits + 96 channels @ 2 bits over dim=128 = 2.25.
    EXPECT_FLOAT_EQ(qkv_effective_bits_for_values(128, 32, 3, 2), 2.25f);
    EXPECT_FLOAT_EQ(qkv_effective_bits(&cfg), 2.25f);
}

TEST(QKVContractExamples, ValidAndInvalidBitContractsAreExplicit) {
    for (int bits = 1; bits <= 8; ++bits) {
        EXPECT_TRUE(qkv_bits_valid(bits));
        EXPECT_TRUE(qkv_bits_codebook(bits));
        EXPECT_FALSE(qkv_bits_raw(bits));
    }
    EXPECT_TRUE(qkv_bits_valid(16));
    EXPECT_TRUE(qkv_bits_valid(32));
    EXPECT_TRUE(qkv_bits_raw(16));
    EXPECT_TRUE(qkv_bits_raw(32));

    for (int bits : {0, 9, 15, 31, 33}) {
        EXPECT_FALSE(qkv_bits_valid(bits));
        EXPECT_FALSE(qkv_bits_codebook(bits));
        EXPECT_FALSE(qkv_bits_raw(bits));
    }
}

TEST(QKVContractExamples, PackedIndicesRoundTripAllSmallBitWidths) {
    for (int bits = 1; bits <= 8; ++bits) {
        const int levels = 1 << bits;
        for (int n : {1, 2, 3, 7, 8, 9, 16, 31}) {
            std::vector<int> input(n);
            for (int i = 0; i < n; ++i) {
                input[i] = (i * 3 + bits) % levels;
            }

            std::vector<uint8_t> packed((static_cast<size_t>(n) * bits + 7u) / 8u, 0xA5u);
            std::vector<int> output(n, -1);
            qkv_pack_indices(input.data(), packed.data(), n, bits);
            qkv_unpack_indices(packed.data(), output.data(), n, bits);
            EXPECT_EQ(output, input) << "bits=" << bits << " n=" << n;
        }
    }
}

TEST(QKVContractExamples, SignPackingRoundTripHandlesTails) {
    std::vector<float> signs = {-1.0f, 1.0f, -0.0f, 0.25f, -3.0f, 2.0f, 5.0f, -8.0f, 1.0f};
    std::vector<uint8_t> packed((signs.size() + 7u) / 8u, 0);
    std::vector<float> unpacked(signs.size(), 0.0f);

    qkv_pack_signs(signs.data(), packed.data(), static_cast<int>(signs.size()));
    qkv_unpack_signs(packed.data(), unpacked.data(), static_cast<int>(unpacked.size()));

    for (size_t i = 0; i < signs.size(); ++i) {
        const float expected = signs[i] >= 0.0f ? 1.0f : -1.0f;
        EXPECT_FLOAT_EQ(unpacked[i], expected);
    }
}

TEST(QKVContractExamples, Fp16ExamplesRemainFiniteAndClose) {
    for (float v : {0.0f, -0.0f, 1.0f, -2.5f, 0.33325f, 65504.0f}) {
        const uint16_t bits = qkv_float_to_fp16_bits(v);
        const float roundtrip = qkv_fp16_bits_to_float(bits);
        EXPECT_TRUE(std::isfinite(roundtrip));
        if (std::fabs(v) < 1000.0f) {
            EXPECT_NEAR(roundtrip, v, 0.0025f);
        }
    }
}

TEST(QKVContractExamples, CodebooksAreOrderedAndNearestIndexIsInRange) {
    for (int bits : {1, 2, 3, 4}) {
        const int levels = 1 << bits;
        std::vector<float> centroids(levels);
        std::vector<float> thresholds(std::max(1, levels - 1));
        qkv_compute_lloyd_max_codebook(
            centroids.data(),
            thresholds.data(),
            bits,
            128
        );

        for (int i = 1; i < levels; ++i) {
            EXPECT_LE(centroids[i - 1], centroids[i]);
        }
        for (int i = 1; i < levels - 1; ++i) {
            EXPECT_LE(thresholds[i - 1], thresholds[i]);
        }

        for (float sample : {-2.0f, -0.25f, 0.0f, 0.25f, 2.0f}) {
            const int idx = qkv_find_nearest_centroid(
                sample,
                centroids.data(),
                thresholds.data(),
                levels
            );
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, levels);
        }
    }
}

TEST(QKVContractExamples, HadamardRotationRoundTripIsStableForPowerOfTwoDim) {
    std::vector<float> signs(kHeadDim);
    std::vector<float> input(kHeadDim);
    std::vector<float> rotated(kHeadDim, 0.0f);
    std::vector<float> restored(kHeadDim, 0.0f);

    for (int i = 0; i < kHeadDim; ++i) {
        signs[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        input[i] = static_cast<float>(i + 1) / 10.0f;
    }

    ASSERT_TRUE(qkv_dim_is_power_of_two(kHeadDim));
    ASSERT_TRUE(qkv_apply_hadamard_rotation_forward(input.data(), signs.data(), rotated.data(), kHeadDim));
    ASSERT_TRUE(qkv_apply_hadamard_rotation_inverse(rotated.data(), signs.data(), restored.data(), kHeadDim));

    for (int i = 0; i < kHeadDim; ++i) {
        EXPECT_NEAR(restored[i], input[i], 1e-5f);
    }
}

TEST(QKVContractExamples, RawFp32KvRoundTripIsLosslessWithoutCompression) {
    qkv_config_t cfg = qkv_config_default(kHeadDim);
    cfg.k_bits = 32;
    cfg.v_bits = 32;
    cfg.enable_qjl = false;
    cfg.enable_rotation = false;
    cfg.sink_tokens = 0;
    cfg.plain_kv_persistent_storage = true;

    qkv_state_t state{};
    ASSERT_TRUE(qkv_init(&state, &cfg, kTokens));

    const std::vector<float> keys = make_example_keys();
    const std::vector<float> values = make_example_values();
    ASSERT_TRUE(qkv_quantize(&state, &cfg, keys.data(), values.data(), kTokens));

    std::vector<float> key_out(keys.size(), 0.0f);
    std::vector<float> value_out(values.size(), 0.0f);
    ASSERT_TRUE(qkv_dequantize(&state, &cfg, key_out.data(), value_out.data(), kTokens));

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_FLOAT_EQ(key_out[i], keys[i]);
        EXPECT_FLOAT_EQ(value_out[i], values[i]);
    }
    qkv_free(&state);
}

TEST(QKVContractExamples, RawFp16KvRoundTripIsCloseWithoutCodebookCompression) {
    qkv_config_t cfg = qkv_config_default(kHeadDim);
    cfg.k_bits = 16;
    cfg.v_bits = 16;
    cfg.enable_qjl = false;
    cfg.enable_rotation = false;
    cfg.sink_tokens = 0;
    cfg.plain_kv_persistent_storage = true;

    qkv_state_t state{};
    ASSERT_TRUE(qkv_init(&state, &cfg, kTokens));

    const std::vector<float> keys = make_example_keys();
    const std::vector<float> values = make_example_values();
    ASSERT_TRUE(qkv_quantize(&state, &cfg, keys.data(), values.data(), kTokens));

    std::vector<float> key_out(keys.size(), 0.0f);
    std::vector<float> value_out(values.size(), 0.0f);
    ASSERT_TRUE(qkv_dequantize(&state, &cfg, key_out.data(), value_out.data(), kTokens));
    expect_roundtrip_within(keys, key_out, 1e-6, 0.001);
    expect_roundtrip_within(values, value_out, 1e-6, 0.001);
    qkv_free(&state);
}

TEST(QKVContractExamples, QuantizedKvRoundTripHasBoundedReconstructionError) {
    for (int bits : {2, 3, 4, 8}) {
        qkv_config_t cfg = qkv_config_default(kHeadDim);
        cfg.k_bits = bits;
        cfg.v_bits = bits;
        cfg.enable_qjl = false;
        cfg.enable_rotation = false;
        cfg.sink_tokens = 0;
        cfg.plain_kv_persistent_storage = false;

        qkv_state_t state{};
        ASSERT_TRUE(qkv_init(&state, &cfg, kTokens));

        const std::vector<float> keys = make_example_keys();
        const std::vector<float> values = make_example_values();
        ASSERT_TRUE(qkv_quantize(&state, &cfg, keys.data(), values.data(), kTokens));

        std::vector<float> key_out(keys.size(), 0.0f);
        std::vector<float> value_out(values.size(), 0.0f);
        ASSERT_TRUE(qkv_dequantize(&state, &cfg, key_out.data(), value_out.data(), kTokens));

        const double max_mse = bits <= 2 ? 0.08 : bits == 3 ? 0.04 : bits == 4 ? 0.02 : 0.01;
        const double max_err = bits <= 2 ? 0.90 : bits == 3 ? 0.70 : bits == 4 ? 0.45 : 0.25;
        expect_roundtrip_within(keys, key_out, max_mse, max_err);
        expect_roundtrip_within(values, value_out, max_mse, max_err);
        qkv_free(&state);
    }
}

TEST(QKVContractExamples, QuantizeDequantizeAndAttentionDoNotBreakOnExampleKvRows) {
    qkv_config_t cfg = qkv_config_default(kHeadDim);
    cfg.k_bits = 4;
    cfg.v_bits = 4;
    cfg.enable_qjl = false;
    cfg.enable_rotation = true;
    cfg.rotation_backend = QKV_ROTATION_BACKEND_HADAMARD_SIGN_FAST;
    cfg.group_size = 64;
    cfg.page_size_tokens = 16;
    cfg.sink_tokens = 0;
    cfg.plain_kv_persistent_storage = false;

    qkv_state_t state{};
    ASSERT_TRUE(qkv_init(&state, &cfg, kTokens));

    const std::vector<float> keys = make_example_keys();
    const std::vector<float> values = make_example_values();
    ASSERT_TRUE(qkv_quantize(&state, &cfg, keys.data(), values.data(), kTokens));

    std::vector<float> key_out(keys.size(), 0.0f);
    std::vector<float> value_out(values.size(), 0.0f);
    ASSERT_TRUE(qkv_dequantize(&state, &cfg, key_out.data(), value_out.data(), kTokens));
    expect_all_finite(key_out);
    expect_all_finite(value_out);

    std::vector<float> query = {0.20f, -0.10f, 0.05f, 0.40f, -0.30f, 0.25f, -0.15f, 0.10f};
    std::vector<float> attention_out(kHeadDim, 0.0f);
    ASSERT_TRUE(qkv_attention_decode(query.data(), &state, &cfg, kTokens, kHeadDim, attention_out.data()));
    expect_all_finite(attention_out);

    qkv_free(&state);
}

TEST(QKVContractExamples, QuantizeTokenMatchesWholeCacheSmokePath) {
    qkv_config_t cfg = qkv_config_default(kHeadDim);
    cfg.k_bits = 3;
    cfg.v_bits = 2;
    cfg.enable_qjl = false;
    cfg.enable_rotation = false;
    cfg.sink_tokens = 0;

    qkv_state_t whole{};
    qkv_state_t appended{};
    ASSERT_TRUE(qkv_init(&whole, &cfg, kTokens));
    ASSERT_TRUE(qkv_init(&appended, &cfg, kTokens));

    const std::vector<float> keys = make_example_keys();
    const std::vector<float> values = make_example_values();
    ASSERT_TRUE(qkv_quantize(&whole, &cfg, keys.data(), values.data(), kTokens));
    for (int t = 0; t < kTokens; ++t) {
        ASSERT_TRUE(qkv_quantize_token(
            &appended,
            &cfg,
            keys.data() + static_cast<size_t>(t) * kHeadDim,
            values.data() + static_cast<size_t>(t) * kHeadDim,
            t
        ));
    }

    std::vector<float> whole_k(keys.size());
    std::vector<float> whole_v(values.size());
    std::vector<float> appended_k(keys.size());
    std::vector<float> appended_v(values.size());
    ASSERT_TRUE(qkv_dequantize(&whole, &cfg, whole_k.data(), whole_v.data(), kTokens));
    ASSERT_TRUE(qkv_dequantize(&appended, &cfg, appended_k.data(), appended_v.data(), kTokens));

    for (size_t i = 0; i < whole_k.size(); ++i) {
        EXPECT_FLOAT_EQ(appended_k[i], whole_k[i]);
        EXPECT_FLOAT_EQ(appended_v[i], whole_v[i]);
    }

    qkv_free(&whole);
    qkv_free(&appended);
}

TEST(QKVContractExamples, InvalidConfigurationsFailClosed) {
    qkv_state_t state{};

    qkv_config_t bad_bits = qkv_config_default(kHeadDim);
    bad_bits.k_bits = 9;
    EXPECT_FALSE(qkv_init(&state, &bad_bits, kTokens));

    qkv_config_t bad_dim = qkv_config_default(0);
    EXPECT_FALSE(qkv_init(&state, &bad_dim, kTokens));

    qkv_config_t bad_backend = qkv_config_default(kHeadDim);
    bad_backend.rotation_backend = 999u;
    EXPECT_FALSE(qkv_init(&state, &bad_backend, kTokens));

    qkv_config_t good = qkv_config_default(kHeadDim);
    EXPECT_FALSE(qkv_init(nullptr, &good, kTokens));
    EXPECT_FALSE(qkv_init(&state, &good, 0));
}
