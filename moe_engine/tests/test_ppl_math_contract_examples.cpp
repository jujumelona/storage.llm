// PPL math contract example tests
// Verifies numerically stable negative-log-likelihood and perplexity math
// without requiring a real model artifact.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

float stable_log_softmax_at(const std::vector<float>& logits, size_t target) {
    if (logits.empty() || target >= logits.size()) {
        return -std::numeric_limits<float>::infinity();
    }
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        if (!std::isfinite(v)) {
            return -std::numeric_limits<float>::infinity();
        }
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (float v : logits) {
        sum_exp += std::exp(static_cast<double>(v - max_logit));
    }
    if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) {
        return -std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(
        static_cast<double>(logits[target] - max_logit) - std::log(sum_exp));
}

double sequence_nll(
    const std::vector<std::vector<float>>& logits_by_pos,
    const std::vector<uint32_t>& targets
) {
    if (logits_by_pos.size() != targets.size() || targets.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double nll = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        const float logp = stable_log_softmax_at(logits_by_pos[i], targets[i]);
        if (!std::isfinite(logp)) {
            return std::numeric_limits<double>::infinity();
        }
        nll -= static_cast<double>(logp);
    }
    return nll;
}

double perplexity_from_nll(double nll, size_t token_count) {
    if (token_count == 0 || !std::isfinite(nll)) {
        return std::numeric_limits<double>::infinity();
    }
    const double mean_nll = nll / static_cast<double>(token_count);
    if (mean_nll > 700.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(mean_nll);
}

}  // namespace

TEST(PplMathContractExamples, UniformDistributionHasVocabSizePerplexity) {
    const std::vector<std::vector<float>> logits = {
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
    };
    const std::vector<uint32_t> targets = {0, 1, 3};
    const double nll = sequence_nll(logits, targets);
    EXPECT_NEAR(nll, 3.0 * std::log(4.0), 1e-6);
    EXPECT_NEAR(perplexity_from_nll(nll, targets.size()), 4.0, 1e-6);
}

TEST(PplMathContractExamples, StableLogSoftmaxHandlesLargeLogits) {
    const std::vector<float> logits = {10000.0f, 9999.0f, 9998.0f};
    const float logp0 = stable_log_softmax_at(logits, 0);
    const float logp1 = stable_log_softmax_at(logits, 1);
    EXPECT_TRUE(std::isfinite(logp0));
    EXPECT_TRUE(std::isfinite(logp1));
    EXPECT_GT(logp0, logp1);
    EXPECT_NEAR(std::exp(logp0) + std::exp(logp1) + std::exp(stable_log_softmax_at(logits, 2)), 1.0, 1e-5);
}

TEST(PplMathContractExamples, ImpossibleInputsFailClosed) {
    EXPECT_FALSE(std::isfinite(stable_log_softmax_at({}, 0)));
    EXPECT_FALSE(std::isfinite(stable_log_softmax_at({0.0f, 1.0f}, 2)));
    EXPECT_FALSE(std::isfinite(sequence_nll({{0.0f, 1.0f}}, {0, 1})));
    EXPECT_FALSE(std::isfinite(perplexity_from_nll(1.0, 0)));
    EXPECT_FALSE(std::isfinite(perplexity_from_nll(std::numeric_limits<double>::infinity(), 1)));
}

TEST(PplMathContractExamples, BetterTargetLogitsLowerPerplexity) {
    const std::vector<std::vector<float>> weak = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    const std::vector<std::vector<float>> strong = {
        {4.0f, 0.0f, 0.0f},
        {0.0f, 4.0f, 0.0f},
    };
    const std::vector<uint32_t> targets = {0, 1};
    const double weak_ppl = perplexity_from_nll(sequence_nll(weak, targets), targets.size());
    const double strong_ppl = perplexity_from_nll(sequence_nll(strong, targets), targets.size());
    EXPECT_TRUE(std::isfinite(weak_ppl));
    EXPECT_TRUE(std::isfinite(strong_ppl));
    EXPECT_LT(strong_ppl, weak_ppl);
}
