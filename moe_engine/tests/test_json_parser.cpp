// JSON Parser Tests (Bug B1)
// Tests for safe JSON parsing with proper string context handling

#include <gtest/gtest.h>
#include "../../json_request.h"

using namespace storagellm;

// ============================================================================
// Test 1: Key Collision - String value contains key name
// ============================================================================
TEST(JSONParser, KeyCollisionInString) {
    std::string payload = R"({"note":"payload says \"selected_experts\":[9]","current_layer":2,"selected_experts":[1,2]})";

    int layer = -1;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, 2);

    auto experts = json_read_int_array(payload, "selected_experts");
    ASSERT_EQ(experts.size(), 2);
    EXPECT_EQ(experts[0], 1);
    EXPECT_EQ(experts[1], 2);
}

// ============================================================================
// Test 2: Nested Array Rejection
// ============================================================================
TEST(JSONParser, NestedArrayRejection) {
    std::string payload = R"({"current_layer":2,"selected_experts":[1,[2]],"next_experts":[3]})";

    auto experts = json_read_int_array(payload, "selected_experts");
    EXPECT_TRUE(experts.empty());  // Should reject nested array
}

// ============================================================================
// Test 3: Type Mismatch - String instead of int
// ============================================================================
TEST(JSONParser, TypeMismatchString) {
    std::string payload = R"({"current_layer":"2","selected_experts":[1,2]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
}

// ============================================================================
// Test 4: Type Mismatch - Object instead of int
// ============================================================================
TEST(JSONParser, TypeMismatchObject) {
    std::string payload = R"({"current_layer":{"value":2},"selected_experts":[1,2]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
}

// ============================================================================
// Test 5: Type Mismatch - Array instead of int
// ============================================================================
TEST(JSONParser, TypeMismatchArray) {
    std::string payload = R"({"current_layer":[2],"selected_experts":[1,2]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
}

// ============================================================================
// Test 6: Malformed JSON - Missing closing bracket
// ============================================================================
TEST(JSONParser, MalformedMissingBracket) {
    std::string payload = R"({"current_layer":2,"selected_experts":[1,2)";

    auto experts = json_read_int_array(payload, "selected_experts");
    EXPECT_TRUE(experts.empty());
}

// ============================================================================
// Test 7: Malformed JSON - Missing closing quote
// ============================================================================
TEST(JSONParser, MalformedMissingQuote) {
    std::string payload = R"({"current_layer:2,"selected_experts":[1,2]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
}

// ============================================================================
// Test 8: Valid Complex Payload
// ============================================================================
TEST(JSONParser, ValidComplexPayload) {
    std::string payload = R"({
        "note": "This is a test with \"escaped quotes\" and [brackets]",
        "current_layer": 5,
        "selected_experts": [0, 1, 2, 3],
        "next_experts": [4, 5, 6],
        "metadata": {
            "nested": "value"
        }
    })";

    int layer = -1;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, 5);

    auto selected = json_read_int_array(payload, "selected_experts");
    ASSERT_EQ(selected.size(), 4);
    EXPECT_EQ(selected[0], 0);
    EXPECT_EQ(selected[3], 3);

    auto next = json_read_int_array(payload, "next_experts");
    ASSERT_EQ(next.size(), 3);
    EXPECT_EQ(next[0], 4);
    EXPECT_EQ(next[2], 6);
}

// ============================================================================
// Test 9: Empty Array
// ============================================================================
TEST(JSONParser, EmptyArray) {
    std::string payload = R"({"current_layer":2,"selected_experts":[]})";

    auto experts = json_read_int_array(payload, "selected_experts");
    EXPECT_TRUE(experts.empty());
}

// ============================================================================
// Test 10: Negative Numbers
// ============================================================================
TEST(JSONParser, NegativeNumbers) {
    std::string payload = R"({"current_layer":-1,"selected_experts":[-1,-2]})";

    int layer = 0;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, -1);

    auto experts = json_read_int_array(payload, "selected_experts");
    ASSERT_EQ(experts.size(), 2);
    EXPECT_EQ(experts[0], -1);
    EXPECT_EQ(experts[1], -2);
}

// ============================================================================
// Test 11: Large Numbers
// ============================================================================
TEST(JSONParser, LargeNumbers) {
    std::string payload = R"({"current_layer":999999,"selected_experts":[100000,200000]})";

    int layer = 0;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, 999999);

    auto experts = json_read_int_array(payload, "selected_experts");
    ASSERT_EQ(experts.size(), 2);
    EXPECT_EQ(experts[0], 100000);
    EXPECT_EQ(experts[1], 200000);
}

// ============================================================================
// Test 12: Whitespace Handling
// ============================================================================
TEST(JSONParser, WhitespaceHandling) {
    std::string payload = R"({  "current_layer"  :  2  ,  "selected_experts"  :  [  1  ,  2  ,  3  ]  })";

    int layer = -1;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, 2);

    auto experts = json_read_int_array(payload, "selected_experts");
    ASSERT_EQ(experts.size(), 3);
}

// ============================================================================
// Test 13: Key Not Found
// ============================================================================
TEST(JSONParser, KeyNotFound) {
    std::string payload = R"({"current_layer":2})";

    int value = -1;
    EXPECT_FALSE(json_read_int(payload, "nonexistent_key", &value));

    auto experts = json_read_int_array(payload, "nonexistent_array");
    EXPECT_TRUE(experts.empty());
}

// ============================================================================
// Test 14: Duplicate Keys (should use first occurrence)
// ============================================================================
TEST(JSONParser, DuplicateKeys) {
    std::string payload = R"({"current_layer":2,"current_layer":5})";

    int layer = -1;
    ASSERT_TRUE(json_read_int(payload, "current_layer", &layer));
    EXPECT_EQ(layer, 2);  // Should use first occurrence
}

// ============================================================================
// Test 15: Reject trailing junk after integers
// ============================================================================
TEST(JSONParser, RejectsTrailingJunkAfterInt) {
    std::string payload = R"({"current_layer":2abc,"selected_experts":[1,2]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
}

// ============================================================================
// Test 16: Reject invalid array tokens and trailing commas
// ============================================================================
TEST(JSONParser, RejectsInvalidArrayTokens) {
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":[1,2abc]})", "selected_experts").empty());
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":[1,]})", "selected_experts").empty());
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":[,1]})", "selected_experts").empty());
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":[1,,2]})", "selected_experts").empty());
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":["1",2]})", "selected_experts").empty());
}

// ============================================================================
// Test 17: Reject integer overflow
// ============================================================================
TEST(JSONParser, RejectsIntegerOverflow) {
    std::string payload = R"({"current_layer":2147483648,"selected_experts":[1]})";

    int layer = -1;
    EXPECT_FALSE(json_read_int(payload, "current_layer", &layer));
    EXPECT_TRUE(json_read_int_array(R"({"selected_experts":[2147483648]})", "selected_experts").empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
