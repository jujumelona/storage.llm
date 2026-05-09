// Weight decode example tests
// These are tiny deterministic tests for the native FP4 path used by the
// weight fallback/decode modules. They ensure example packed nibble data does
// not break at odd offsets, tails, or scaled dot products.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
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
