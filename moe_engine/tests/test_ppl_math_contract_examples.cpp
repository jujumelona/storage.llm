// PPL math contract tests
// These pin the exact PPL/NLL arithmetic used by eval without requiring a full
// model fixture. They cover target-window weighting via first_target_index and
// edge cases that can otherwise silently perturb regression metrics.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

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
