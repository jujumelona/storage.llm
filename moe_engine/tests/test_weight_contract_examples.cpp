// Weight decode example tests
// These are tiny deterministic tests for the native FP4 path used by the
// weight fallback/decode modules. They ensure example packed nibble data does
// not break at odd offsets, tails, scaled dot products, or PPL contract math.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "../../moe_engine/native/fp4_decode.h"

TEST(WeightContractExamples, NvFp4NibbleOrderAndScaledValuesAreStable) {
    const uint8_t packed[] = {0x12u, 0x9Fu};

    EXPECT_EQ(moe_engine::fp4_code_at(packed, 0), 0x1u);  // high nibble
    EXPECT_EQ(moe_engine::fp4_code_at(packed, 1), 0x2u);  // low nibble
    EXPECT_EQ(moe_engine::fp4_code_at(packed, 2), 0x9u);
    EXPECT_EQ(moe_engine::fp4_code_at(packed, 3), 0xFu);

    EXPECT_FLOAT_EQ(moe_engine::fp4_value_at(packed, 0, 2.0f), 1.0f);
    EXPECT_FLOAT_EQ(moe_engine::fp4_value_at(packed, 1, 2.0f), 2.0f);
    EXPECT_FLOAT_EQ(moe_engine::fp4_value_at(packed, 2, 2.0f), -1.0f);
    EXPECT_FLOAT_EQ(moe_engine::fp4_value_at(packed, 3, 2.0f), -12.0f);
}

TEST(WeightContractExamples, DequantPackedFp4HandlesOddElementOffsetAndTail) {
    const uint8_t packed[] = {0x12u, 0x34u, 0xABu};
    std::vector<float> out(4, 0.0f);

    moe_engine::dequant_packed_fp4_scalar(
        packed,
        1,   // odd element offset starts at low nibble of 0x12
        out.size(),
        0.5f,
        out.data()
    );

    EXPECT_FLOAT_EQ(out[0], 0.5f);    // code 2 => 1.0 * 0.5
    EXPECT_FLOAT_EQ(out[1], 0.75f);   // code 3 => 1.5 * 0.5
    EXPECT_FLOAT_EQ(out[2], 1.0f);    // code 4 => 2.0 * 0.5
    EXPECT_FLOAT_EQ(out[3], -0.5f);   // code A => -1.0 * 0.5
}

TEST(WeightContractExamples, DotPackedFp4MatchesManualComputation) {
    const uint8_t packed[] = {0x17u, 0x8Fu};
    const float x[] = {2.0f, -1.0f, 3.0f, 0.5f};

    // codes: 1=>0.5, 7=>6.0, 8=>-0.0, F=>-6.0; scale=0.25
    const float expected =
        (0.5f * 0.25f) * 2.0f +
        (6.0f * 0.25f) * -1.0f +
        (-0.0f * 0.25f) * 3.0f +
        (-6.0f * 0.25f) * 0.5f;

    const float actual = moe_engine::dot_packed_fp4_f32(packed, 0, x, 4, 0.25f);
    EXPECT_NEAR(actual, expected, 1e-6f);
}

TEST(WeightContractExamples, AllNvFp4TableEntriesAreFinite) {
    uint8_t packed[8]{};
    for (int code = 0; code < 16; ++code) {
        packed[code >> 1] |= (code & 1) ? static_cast<uint8_t>(code)
                                        : static_cast<uint8_t>(code << 4);
    }

    for (int i = 0; i < 16; ++i) {
        const float v = moe_engine::fp4_value_at(packed, static_cast<std::size_t>(i), 1.0f);
        EXPECT_TRUE(std::isfinite(v));
    }
}

namespace {

struct PPLStats {
    double nll = 0.0;
    double mean_nll = 0.0;
    double perplexity = 0.0;
    unsigned evaluated = 0;
};

bool compute_reference_ppl(
    const std::vector<double>& token_logprobs,
    unsigned first_target_index,
    PPLStats* out
) {
    if (!out || token_logprobs.empty()) {
        return false;
    }
    if (first_target_index == 0) {
        first_target_index = 1;
    }
    if (first_target_index > token_logprobs.size()) {
        return false;
    }

    PPLStats stats{};
    for (unsigned target_index = 1; target_index <= token_logprobs.size(); ++target_index) {
        if (target_index < first_target_index) {
            continue;
        }
        const double logprob = token_logprobs[target_index - 1u];
        if (!std::isfinite(logprob)) {
            return false;
        }
        const double neg_logprob = -logprob;
        if (stats.nll > std::numeric_limits<double>::max() - neg_logprob) {
            return false;
        }
        stats.nll += neg_logprob;
        ++stats.evaluated;
    }
    if (stats.evaluated == 0) {
        return false;
    }
    stats.mean_nll = stats.nll / static_cast<double>(stats.evaluated);
    stats.perplexity = stats.mean_nll > std::log(std::numeric_limits<double>::max())
        ? std::numeric_limits<double>::max()
        : std::exp(stats.mean_nll);
    *out = stats;
    return std::isfinite(out->mean_nll) && std::isfinite(out->perplexity);
}

}  // namespace

TEST(PPLMathContractExamples, FirstTargetIndexSkipsUnweightedPromptPrefix) {
    // Logprobs correspond to target token indices 1, 2, 3, 4.
    const std::vector<double> logprobs = {-0.2, -0.5, -1.0, -2.0};

    PPLStats stats{};
    ASSERT_TRUE(compute_reference_ppl(logprobs, 3u, &stats));

    EXPECT_EQ(stats.evaluated, 2u);
    EXPECT_DOUBLE_EQ(stats.nll, 3.0);
    EXPECT_DOUBLE_EQ(stats.mean_nll, 1.5);
    EXPECT_DOUBLE_EQ(stats.perplexity, std::exp(1.5));
}

TEST(PPLMathContractExamples, ZeroFirstTargetIndexMatchesPublicEvalDefault) {
    const std::vector<double> logprobs = {-0.25, -0.75};

    PPLStats stats_default{};
    PPLStats stats_explicit{};
    ASSERT_TRUE(compute_reference_ppl(logprobs, 0u, &stats_default));
    ASSERT_TRUE(compute_reference_ppl(logprobs, 1u, &stats_explicit));

    EXPECT_EQ(stats_default.evaluated, stats_explicit.evaluated);
    EXPECT_DOUBLE_EQ(stats_default.nll, stats_explicit.nll);
    EXPECT_DOUBLE_EQ(stats_default.mean_nll, stats_explicit.mean_nll);
    EXPECT_DOUBLE_EQ(stats_default.perplexity, stats_explicit.perplexity);
}

TEST(PPLMathContractExamples, RejectsEmptyTargetWindow) {
    const std::vector<double> logprobs = {-0.25, -0.75};

    PPLStats stats{};
    EXPECT_FALSE(compute_reference_ppl(logprobs, 3u, &stats));
}

TEST(PPLMathContractExamples, RejectsNonFiniteTokenLogprob) {
    PPLStats stats{};
    EXPECT_FALSE(compute_reference_ppl({-0.25, -std::numeric_limits<double>::infinity()}, 1u, &stats));
    EXPECT_FALSE(compute_reference_ppl({-0.25, std::numeric_limits<double>::quiet_NaN()}, 1u, &stats));
}

TEST(PPLMathContractExamples, SaturatesPerplexityInsteadOfOverflowing) {
    const double huge_nll = std::log(std::numeric_limits<double>::max()) + 1.0;

    PPLStats stats{};
    ASSERT_TRUE(compute_reference_ppl({-huge_nll}, 1u, &stats));

    EXPECT_EQ(stats.evaluated, 1u);
    EXPECT_DOUBLE_EQ(stats.nll, huge_nll);
    EXPECT_DOUBLE_EQ(stats.mean_nll, huge_nll);
    EXPECT_EQ(stats.perplexity, std::numeric_limits<double>::max());
}
