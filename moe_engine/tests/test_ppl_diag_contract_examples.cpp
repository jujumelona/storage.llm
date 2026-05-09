// PPL diagnostics hook contract example tests
// Verifies lightweight diagnostics aggregation semantics without depending on
// a real model or enabling heavyweight runtime instrumentation.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

struct PplDiagEvent {
    uint32_t token_index = 0;
    uint32_t target_token = 0;
    float logprob = 0.0f;
    float nll = 0.0f;
    bool accepted = false;
};

class PplDiagRecorder {
public:
    bool record(uint32_t token_index, uint32_t target_token, float logprob) {
        if (!std::isfinite(logprob)) {
            return false;
        }
        PplDiagEvent event{};
        event.token_index = token_index;
        event.target_token = target_token;
        event.logprob = logprob;
        event.nll = -logprob;
        event.accepted = true;
        events_.push_back(event);
        total_nll_ += static_cast<double>(event.nll);
        return true;
    }

    size_t size() const { return events_.size(); }
    double total_nll() const { return total_nll_; }
    double mean_nll() const {
        return events_.empty() ? std::numeric_limits<double>::infinity() :
            total_nll_ / static_cast<double>(events_.size());
    }
    double perplexity() const {
        const double mean = mean_nll();
        if (!std::isfinite(mean) || mean > 700.0) {
            return std::numeric_limits<double>::infinity();
        }
        return std::exp(mean);
    }
    const PplDiagEvent& at(size_t i) const { return events_.at(i); }

private:
    std::vector<PplDiagEvent> events_;
    double total_nll_ = 0.0;
};

}  // namespace

TEST(PplDiagContractExamples, RecorderAggregatesTokenEvents) {
    PplDiagRecorder recorder;
    ASSERT_TRUE(recorder.record(0, 10, -0.25f));
    ASSERT_TRUE(recorder.record(1, 11, -0.75f));

    EXPECT_EQ(recorder.size(), 2u);
    EXPECT_EQ(recorder.at(0).token_index, 0u);
    EXPECT_EQ(recorder.at(0).target_token, 10u);
    EXPECT_TRUE(recorder.at(0).accepted);
    EXPECT_NEAR(recorder.total_nll(), 1.0, 1e-6);
    EXPECT_NEAR(recorder.mean_nll(), 0.5, 1e-6);
    EXPECT_NEAR(recorder.perplexity(), std::exp(0.5), 1e-6);
}

TEST(PplDiagContractExamples, NonFiniteLogprobIsRejected) {
    PplDiagRecorder recorder;
    EXPECT_FALSE(recorder.record(0, 1, std::numeric_limits<float>::quiet_NaN()));
    EXPECT_FALSE(recorder.record(0, 1, -std::numeric_limits<float>::infinity()));
    EXPECT_EQ(recorder.size(), 0u);
    EXPECT_FALSE(std::isfinite(recorder.mean_nll()));
    EXPECT_FALSE(std::isfinite(recorder.perplexity()));
}

TEST(PplDiagContractExamples, EventOrderingIsStable) {
    PplDiagRecorder recorder;
    for (uint32_t i = 0; i < 8; ++i) {
        ASSERT_TRUE(recorder.record(i, 100u + i, -0.1f * static_cast<float>(i + 1u)));
    }
    ASSERT_EQ(recorder.size(), 8u);
    for (uint32_t i = 0; i < 8; ++i) {
        EXPECT_EQ(recorder.at(i).token_index, i);
        EXPECT_EQ(recorder.at(i).target_token, 100u + i);
        EXPECT_GT(recorder.at(i).nll, 0.0f);
    }
}

TEST(PplDiagContractExamples, HugeMeanNllFailsClosedInsteadOfOverflowing) {
    PplDiagRecorder recorder;
    ASSERT_TRUE(recorder.record(0, 1, -1000.0f));
    EXPECT_FALSE(std::isfinite(recorder.perplexity()));
}
