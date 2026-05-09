// Prefetch Semantic Validation Tests (Bug B2)
// Tests for request validation with proper range checking

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <algorithm>

// Mock model shape
struct MockModelShape {
    uint32_t num_hidden_layers = 32;
    uint32_t experts_per_moe_layer = 8;
    uint32_t first_moe_layer = 0;
};

static MockModelShape mock_shape;

MockModelShape moe_pc_Moe1_model_shape() {
    return mock_shape;
}

// Mock engine type
typedef void* moe_pc_engine_t;

// Validation function (same as in pc_server_prefetch.inc)
static bool validate_prefetch_request(
    moe_pc_engine_t* engine,
    int current_layer,
    std::vector<int>& selected_experts,
    std::vector<int>& next_experts,
    std::string* err
) {
    const auto shape = moe_pc_Moe1_model_shape();
    const int layer_count = static_cast<int>(shape.num_hidden_layers);

    if (layer_count <= 0 || current_layer < 0 || current_layer >= layer_count) {
        if (err) *err = "current_layer out of range [0, " + std::to_string(layer_count) + ")";
        return false;
    }

    const int expert_count = static_cast<int>(shape.experts_per_moe_layer);
    if (expert_count <= 0) {
        if (err) *err = "expert_count is not configured";
        return false;
    }

    auto normalize = [&](std::vector<int>& ids, const char* field) -> bool {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

        if (ids.size() > 64) {
            if (err) *err = std::string(field) + " too long (max 64)";
            return false;
        }

        for (int id : ids) {
            if (id < 0 || id >= expert_count) {
                if (err) *err = std::string(field) + " contains invalid expert id " +
                                std::to_string(id) + " (valid range: [0, " +
                                std::to_string(expert_count) + "))";
                return false;
            }
        }
        return true;
    };

    return normalize(selected_experts, "selected_experts") &&
           normalize(next_experts, "next_experts");
}

// ============================================================================
// Test 1: Valid Request
// ============================================================================
TEST(PrefetchValidation, ValidRequest) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {0, 1, 2};
    std::vector<int> next = {3, 4};
    std::string err;

    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_TRUE(err.empty());
}

// ============================================================================
// Test 2: Negative Layer
// ============================================================================
TEST(PrefetchValidation, NegativeLayer) {
    moe_pc_engine_t* engine = nullptr;
    int layer = -1;
    std::vector<int> selected = {0, 1};
    std::vector<int> next = {2};
    std::string err;

    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("out of range"), std::string::npos);
}

// ============================================================================
// Test 3: Layer Out of Range (Too High)
// ============================================================================
TEST(PrefetchValidation, LayerTooHigh) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 999999;
    std::vector<int> selected = {0, 1};
    std::vector<int> next = {2};
    std::string err;

    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("out of range"), std::string::npos);
}

// ============================================================================
// Test 4: Negative Expert ID
// ============================================================================
TEST(PrefetchValidation, NegativeExpertID) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {-1, 0, 1};
    std::vector<int> next = {2};
    std::string err;

    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("invalid expert id"), std::string::npos);
}

// ============================================================================
// Test 5: Expert ID Out of Range
// ============================================================================
TEST(PrefetchValidation, ExpertIDTooHigh) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {0, 1, 999999};
    std::vector<int> next = {2};
    std::string err;

    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("invalid expert id"), std::string::npos);
}

// ============================================================================
// Test 6: Duplicate Expert IDs (Should be deduplicated)
// ============================================================================
TEST(PrefetchValidation, DuplicateExpertIDs) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {3, 3, 3, 1, 1};
    std::vector<int> next = {4, 4};
    std::string err;

    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_TRUE(err.empty());

    // Check deduplication
    EXPECT_EQ(selected.size(), 2);  // Should be [1, 3]
    EXPECT_EQ(next.size(), 1);      // Should be [4]
}

// ============================================================================
// Test 7: Too Many Experts (> 64)
// ============================================================================
TEST(PrefetchValidation, TooManyExperts) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected;
    for (int i = 0; i < 65; ++i) {
        selected.push_back(i);
    }
    std::vector<int> next;
    std::string err;

    const MockModelShape saved = mock_shape;
    mock_shape.experts_per_moe_layer = 128;  // Make IDs valid so the length cap is tested.
    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_NE(err.find("too long"), std::string::npos);
    mock_shape = saved;
}

// ============================================================================
// Test 8: Empty Expert Lists
// ============================================================================
TEST(PrefetchValidation, EmptyExpertLists) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected;
    std::vector<int> next;
    std::string err;

    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_TRUE(err.empty());
}

// ============================================================================
// Test 9: Next Experts Invalid
// ============================================================================
TEST(PrefetchValidation, NextExpertsInvalid) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {0, 1};
    std::vector<int> next = {-1, 999999};
    std::string err;

    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected, next, &err));
    EXPECT_FALSE(err.empty());
    EXPECT_NE(err.find("next_experts"), std::string::npos);
}

// ============================================================================
// Test 10: Boundary Layer (First and Last)
// ============================================================================
TEST(PrefetchValidation, BoundaryLayers) {
    moe_pc_engine_t* engine = nullptr;
    std::vector<int> selected = {0, 1};
    std::vector<int> next = {2};
    std::string err;

    // First layer (0)
    EXPECT_TRUE(validate_prefetch_request(engine, 0, selected, next, &err));

    // Last layer (31)
    EXPECT_TRUE(validate_prefetch_request(engine, 31, selected, next, &err));

    // Just beyond last layer (32)
    EXPECT_FALSE(validate_prefetch_request(engine, 32, selected, next, &err));
}

// ============================================================================
// Test 11: Boundary Expert IDs
// ============================================================================
TEST(PrefetchValidation, BoundaryExpertIDs) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::string err;

    // First expert (0)
    std::vector<int> selected1 = {0};
    std::vector<int> next1 = {1};
    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected1, next1, &err));

    // Last expert (7)
    std::vector<int> selected2 = {7};
    std::vector<int> next2 = {6};
    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected2, next2, &err));

    // Just beyond last expert (8)
    std::vector<int> selected3 = {8};
    std::vector<int> next3 = {0};
    EXPECT_FALSE(validate_prefetch_request(engine, layer, selected3, next3, &err));
}

// ============================================================================
// Test 12: Sorting Verification
// ============================================================================
TEST(PrefetchValidation, SortingVerification) {
    moe_pc_engine_t* engine = nullptr;
    int layer = 5;
    std::vector<int> selected = {5, 2, 7, 1, 3};
    std::vector<int> next = {6, 0, 4};
    std::string err;

    EXPECT_TRUE(validate_prefetch_request(engine, layer, selected, next, &err));

    // Verify sorting
    EXPECT_TRUE(std::is_sorted(selected.begin(), selected.end()));
    EXPECT_TRUE(std::is_sorted(next.begin(), next.end()));
}


// ============================================================================
// Test 13: Zero Shape Fails Closed
// ============================================================================
TEST(PrefetchValidation, ZeroShapeFailsClosed) {
    moe_pc_engine_t* engine = nullptr;
    std::vector<int> selected = {0};
    std::vector<int> next = {1};
    std::string err;

    const MockModelShape saved = mock_shape;
    mock_shape.num_hidden_layers = 0;
    EXPECT_FALSE(validate_prefetch_request(engine, 0, selected, next, &err));
    EXPECT_NE(err.find("out of range"), std::string::npos);

    err.clear();
    mock_shape = saved;
    mock_shape.experts_per_moe_layer = 0;
    EXPECT_FALSE(validate_prefetch_request(engine, 0, selected, next, &err));
    EXPECT_NE(err.find("expert_count"), std::string::npos);
    mock_shape = saved;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
