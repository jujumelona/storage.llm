// Loader and manifest contract example tests
// Uses tiny on-disk manifests with representative block numbers to make sure
// example weight-block metadata and JSON KV-like values are parsed safely.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>

#include "../../loader/json_scan.h"
#include "../../loader/manifest_lookup.h"
#include "../../loader/manifest_projection.h"

namespace {

std::string write_temp_manifest(const std::string& name, const std::string& body) {
    const std::string path = std::string(::testing::TempDir()) + name;
    std::ofstream out(path, std::ios::binary);
    out << body;
    out.close();
    return path;
}

std::string projection_json(uint64_t base, uint32_t rows = 6144, uint32_t cols = 2048) {
    return std::string("{") +
        "\"weight_block\":" + std::to_string(base + 0) + "," +
        "\"raw_scale_block\":" + std::to_string(base + 1) + "," +
        "\"raw_scale2_block\":" + std::to_string(base + 2) + "," +
        "\"aux0_block\":" + std::to_string(base + 3) + "," +
        "\"aux1_block\":" + std::to_string(base + 4) + "," +
        "\"rows\":" + std::to_string(rows) + "," +
        "\"cols\":" + std::to_string(cols) + "," +
        "\"groups\":96," +
        "\"group_size\":64," +
        "\"scale_mode\":\"grouped\"" +
        "}";
}

std::string tensor_projection_json(
    uint64_t base,
    uint32_t rows,
    uint32_t cols,
    const std::string& file
) {
    return std::string("{") +
        "\"part\":4," +
        "\"part_path\":\"" + file + "\"," +
        "\"bundle_offset\":" + std::to_string(base * 4096) + "," +
        "\"bundle_length\":4096," +
        "\"weight_block\":" + std::to_string(base) + "," +
        "\"rows\":" + std::to_string(rows) + "," +
        "\"cols\":" + std::to_string(cols) + "," +
        "\"groups\":8," +
        "\"group_size\":64," +
        "\"scale_mode\":\"grouped\"" +
        "}";
}

}  // namespace

TEST(LoaderManifestExamples, ProjectionExampleNumbersParseExactly) {
    const std::string text = projection_json(100);
    storagellm::JsonSlice slice{&text, 0, text.size()};

    storagellm::ProjectionBlocks blocks{};
    ASSERT_TRUE(storagellm::parse_projection_blocks(slice, &blocks));
    EXPECT_EQ(blocks.weight_block, 100u);
    EXPECT_EQ(blocks.scale_block, 101u);
    EXPECT_EQ(blocks.scale2_block, 102u);
    EXPECT_EQ(blocks.aux0_block, 103u);
    EXPECT_EQ(blocks.aux1_block, 104u);
    EXPECT_EQ(blocks.rows, 6144u);
    EXPECT_EQ(blocks.cols, 2048u);
    EXPECT_EQ(blocks.groups, 96u);
    EXPECT_EQ(blocks.group_size, 64u);
    EXPECT_EQ(blocks.scale_mode, "grouped");
}

TEST(LoaderManifestExamples, ProjectionOverflowClampsInsteadOfWrapping) {
    const uint64_t huge = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 77ull;
    const std::string text = std::string("{") +
        "\"weight_block\":" + std::to_string(huge) + "," +
        "\"raw_scale_block\":" + std::to_string(huge) + "," +
        "\"rows\":" + std::to_string(huge) + "," +
        "\"cols\":" + std::to_string(huge) +
        "}";
    storagellm::JsonSlice slice{&text, 0, text.size()};

    storagellm::ProjectionBlocks blocks{};
    ASSERT_TRUE(storagellm::parse_projection_blocks(slice, &blocks));
    EXPECT_EQ(blocks.weight_block, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(blocks.scale_block, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(blocks.rows, std::numeric_limits<uint32_t>::max());
    EXPECT_EQ(blocks.cols, std::numeric_limits<uint32_t>::max());
}

TEST(LoaderManifestExamples, FullManifestFindsRepresentativeExpertTriplet) {
    const std::string manifest = std::string("{") +
        "\"metadata\":{\"note\":\"example manifest with qkv_k_bits text that must not matter\"}," +
        "\"experts\":{" +
        "\"L21.E7\":{" +
            "\"part\":3," +
            "\"bundle_offset\":4096," +
            "\"bundle_length\":8192," +
            "\"part_path\":\"model-00004-of-00021.gguf\"," +
            "\"projections\":{" +
                "\"gate_proj\":" + projection_json(10) + "," +
                "\"up_proj\":" + projection_json(20) + "," +
                "\"down_proj\":" + projection_json(30) +
            "}" +
        "}" +
        "}" +
    "}";

    const std::string path = write_temp_manifest("manifest_contract_example.json", manifest);
    storagellm::ManifestLookup lookup;
    ASSERT_TRUE(lookup.load(path.c_str()));

    storagellm::ExpertManifestEntry entry{};
    ASSERT_TRUE(lookup.find_expert(21, 7, &entry));
    EXPECT_EQ(entry.layer, 21u);
    EXPECT_EQ(entry.expert, 7u);
    EXPECT_EQ(entry.part, 3u);
    EXPECT_EQ(entry.bundle_offset, 4096u);
    EXPECT_EQ(entry.bundle_length, 8192u);
    EXPECT_EQ(entry.part_path, "model-00004-of-00021.gguf");
    EXPECT_EQ(entry.gate.weight_block, 10u);
    EXPECT_EQ(entry.up.weight_block, 20u);
    EXPECT_EQ(entry.down.weight_block, 30u);
    EXPECT_FALSE(lookup.find_expert(21, 8, &entry));
}

TEST(LoaderManifestExamples, GenericGgufTensorAliasesAssembleExpertTriplet) {
    const std::string manifest = std::string("{") +
        "\"tensors\":{" +
        "\"blk.2.ffn_gate_exps.weight\":{\"ignored\":true}," +
        "\"layers.12.mlp.experts.5.gate_proj.weight\":" + tensor_projection_json(101, 4096, 1024, "model-00001.gguf") + "," +
        "\"layers.12.mlp.experts.5.up_proj.weight\":" + tensor_projection_json(102, 4096, 1024, "model-00001.gguf") + "," +
        "\"layers.12.mlp.experts.5.down_proj.weight\":" + tensor_projection_json(103, 1024, 4096, "model-00001.gguf") +
        "}" +
    "}";

    const std::string path = write_temp_manifest("manifest_tensor_aliases.json", manifest);
    storagellm::ManifestLookup lookup;
    ASSERT_TRUE(lookup.load(path.c_str()));

    storagellm::ExpertManifestEntry entry{};
    ASSERT_TRUE(lookup.find_expert(12, 5, &entry));
    EXPECT_EQ(entry.layer, 12u);
    EXPECT_EQ(entry.expert, 5u);
    EXPECT_EQ(entry.part, 4u);
    EXPECT_EQ(entry.part_path, "model-00001.gguf");
    EXPECT_EQ(entry.gate.weight_block, 101u);
    EXPECT_EQ(entry.up.weight_block, 102u);
    EXPECT_EQ(entry.down.weight_block, 103u);
    EXPECT_EQ(entry.gate.rows, 4096u);
    EXPECT_EQ(entry.down.cols, 4096u);
}

TEST(LoaderManifestExamples, W1W2W3TensorAliasesMapToGateDownUp) {
    const std::string manifest = std::string("{") +
        "\"model.layers.3.block_sparse_moe.experts.11.w1.weight\":" + tensor_projection_json(201, 2048, 512, "moe.gguf") + "," +
        "\"model.layers.3.block_sparse_moe.experts.11.w3.weight\":" + tensor_projection_json(202, 2048, 512, "moe.gguf") + "," +
        "\"model.layers.3.block_sparse_moe.experts.11.w2.weight\":" + tensor_projection_json(203, 512, 2048, "moe.gguf") +
    "}";

    const std::string path = write_temp_manifest("manifest_w_aliases.json", manifest);
    storagellm::ManifestLookup lookup;
    ASSERT_TRUE(lookup.load(path.c_str()));

    storagellm::ExpertManifestEntry entry{};
    ASSERT_TRUE(lookup.find_expert(3, 11, &entry));
    EXPECT_EQ(entry.gate.weight_block, 201u);
    EXPECT_EQ(entry.up.weight_block, 202u);
    EXPECT_EQ(entry.down.weight_block, 203u);
}

TEST(LoaderManifestExamples, IncompleteTensorAliasTripletFailsClosed) {
    const std::string manifest = std::string("{") +
        "\"layers.9.mlp.experts.2.gate_proj.weight\":" + tensor_projection_json(301, 1024, 256, "partial.gguf") + "," +
        "\"layers.9.mlp.experts.2.up_proj.weight\":" + tensor_projection_json(302, 1024, 256, "partial.gguf") +
    "}";

    const std::string path = write_temp_manifest("manifest_partial_aliases.json", manifest);
    storagellm::ManifestLookup lookup;
    ASSERT_TRUE(lookup.load(path.c_str()));

    storagellm::ExpertManifestEntry entry{};
    EXPECT_FALSE(lookup.find_expert(9, 2, &entry));
}

TEST(LoaderManifestExamples, MalformedExpertIsSkippedButLaterValidEntryStillLoads) {
    const std::string manifest = std::string("{") +
        "\"L1.E1\":{\"part\":1,\"bundle_offset\":0,\"bundle_length\":1}," +
        "\"L1.E2\":{" +
            "\"part\":2," +
            "\"bundle_offset\":64," +
            "\"bundle_length\":128," +
            "\"part_path\":\"ok.gguf\"," +
            "\"projections\":{" +
                "\"gate_proj\":" + projection_json(1000) + "," +
                "\"up_proj\":" + projection_json(2000) + "," +
                "\"down_proj\":" + projection_json(3000) +
            "}" +
        "}" +
    "}";

    const std::string path = write_temp_manifest("manifest_skip_malformed_example.json", manifest);
    storagellm::ManifestLookup lookup;
    ASSERT_TRUE(lookup.load(path.c_str()));

    storagellm::ExpertManifestEntry entry{};
    EXPECT_FALSE(lookup.find_expert(1, 1, &entry));
    ASSERT_TRUE(lookup.find_expert(1, 2, &entry));
    EXPECT_EQ(entry.bundle_offset, 64u);
    EXPECT_EQ(entry.down.group_size, 64u);
}

TEST(LoaderManifestExamples, NullAndMissingManifestInputsFailClosed) {
    storagellm::ManifestLookup lookup;
    EXPECT_FALSE(lookup.load(nullptr));
    EXPECT_FALSE(lookup.load("/path/that/does/not/exist/storage_llm_manifest.json"));

    storagellm::ExpertManifestEntry entry{};
    EXPECT_FALSE(lookup.find_expert(0, 0, &entry));
    EXPECT_FALSE(lookup.find_expert(0, 0, nullptr));
}
