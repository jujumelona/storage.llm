#include <gtest/gtest.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "../include/moe_pc_engine.h"

namespace {

constexpr uint32_t Layer = 10;
constexpr uint32_t Expert = 0;
constexpr uint32_t Hidden = 6144;
constexpr uint32_t Inter = 2048;
constexpr uint64_t GateBytes = uint64_t(Inter) * Hidden * sizeof(float);
constexpr uint64_t UpBytes = GateBytes;
constexpr uint64_t DownBytes = uint64_t(Hidden) * Inter * sizeof(float);
constexpr uint64_t GateOff = 0;
constexpr uint64_t UpOff = GateOff + GateBytes;
constexpr uint64_t DownOff = UpOff + UpBytes;
constexpr uint64_t TotalBytes = DownOff + DownBytes;

struct Fix {
    bool ok = false;
    std::string error;
    std::string root;
    std::string part_name;
    std::string part;
    std::string index;
};

bool mkdir_ok(const std::string& p) {
#ifdef _WIN32
    return _mkdir(p.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

void u16(std::ofstream& o, uint16_t v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void u32(std::ofstream& o, uint32_t v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void u64(std::ofstream& o, uint64_t v) { o.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void str16(std::ofstream& o, const std::string& s) {
    u16(o, static_cast<uint16_t>(s.size()));
    o.write(s.data(), static_cast<std::streamsize>(s.size()));
}

bool f32_at(std::fstream& f, uint64_t off, float v) {
    f.seekp(static_cast<std::streamoff>(off), std::ios::beg);
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    return f.good();
}

void record(std::ofstream& o, uint64_t off, uint64_t bytes, uint32_t proj, uint32_t rows, uint32_t cols) {
    u64(o, off);                 // weight offset
    u64(o, bytes);               // weight length
    u64(o, 0); u64(o, 0);        // scale
    u64(o, 0); u64(o, 0);        // scale2
    u32(o, 1); u32(o, 1);        // part/shard
    u32(o, Layer); u32(o, Expert); u32(o, proj);
    u32(o, rows); u32(o, cols);
    u32(o, cols >= 16 ? cols / 16 : 1);
    u32(o, cols >= 16 ? 16 : cols);
    u16(o, 0); u16(o, 0); u32(o, 0);
    u32(o, 0xFFFFFFFFu);
    u32(o, moe_WEIGHT_ENCODING_RAW_FP32);
}

Fix make_fixture(const char* suffix, bool write_part) {
    Fix f;
    f.root = std::string(::testing::TempDir()) + suffix;
    f.part_name = "tiny-part.bin";
    f.part = f.root + "/" + f.part_name;
    f.index = f.root + "/tiny.sltidx3";

    if (!mkdir_ok(f.root)) {
        f.error = "mkdir failed: " + f.root;
        return f;
    }

    if (write_part) {
        std::fstream p(f.part, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
        if (!p.good()) {
            f.error = "open part failed: " + f.part;
            return f;
        }
        p.seekp(static_cast<std::streamoff>(TotalBytes - 1), std::ios::beg);
        char z = 0;
        p.write(&z, 1);
        if (!p.good()) {
            f.error = "resize part failed";
            return f;
        }
        if (!f32_at(p, GateOff, 1.0f) || !f32_at(p, UpOff, 2.0f) || !f32_at(p, DownOff, 3.0f)) {
            f.error = "write sparse weights failed";
            return f;
        }
    }

    std::ofstream idx(f.index, std::ios::binary);
    if (!idx.good()) {
        f.error = "open index failed: " + f.index;
        return f;
    }
    const char magic[8] = {'S','L','T','I','D','X','3','\0'};
    idx.write(magic, sizeof(magic));
    u32(idx, 3); u32(idx, 1); u32(idx, 3); u32(idx, 0);
    str16(idx, f.part_name);
    record(idx, GateOff, GateBytes, moe_PROJ_GATE, Inter, Hidden);
    record(idx, UpOff, UpBytes, moe_PROJ_UP, Inter, Hidden);
    record(idx, DownOff, DownBytes, moe_PROJ_DOWN, Hidden, Inter);
    if (!idx.good()) {
        f.error = "write index failed";
        return f;
    }
    f.ok = true;
    return f;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

moe_pc_engine_config_t cpu_cfg() {
    moe_pc_engine_config_t c = moe_pc_default_config();
    c.preferred_backend = moe_BACKEND_CPU;
    c.platform = moe_PLATFORM_CPU_ONLY;
    c.vram_budget_bytes = 64ull * 1024ull * 1024ull;
    c.ram_budget_bytes = 512ull * 1024ull * 1024ull;
    return c;
}

}  // namespace

TEST(EngineFixtureLoading, LoadsTensorIndexAndExecutesExpertTriplet) {
    const Fix fx = make_fixture("storagellm_engine_fixture_root", true);
    ASSERT_TRUE(fx.ok) << fx.error;

    moe_pc_engine_config_t cfg = cpu_cfg();
    moe_pc_engine_t* e = moe_pc_engine_create(&cfg);
    ASSERT_NE(e, nullptr);

    ASSERT_TRUE(moe_pc_engine_set_model_root(e, fx.root.c_str())) << "set model root failed";
    ASSERT_TRUE(moe_pc_engine_load_codec_table(e, fx.index.c_str(), fx.root.c_str(), nullptr)) << "load index failed";
    ASSERT_EQ(moe_pc_engine_tensor_count(e), 3u);

    moe_tensor_info_t info{};
    ASSERT_TRUE(moe_pc_engine_find_tensor(e, Layer, Expert, moe_PROJ_GATE, &info)) << "gate missing";
    EXPECT_EQ(info.rows, Inter);
    EXPECT_EQ(info.cols, Hidden);
    EXPECT_EQ(info.weight_encoding, moe_WEIGHT_ENCODING_RAW_FP32);
    ASSERT_TRUE(moe_pc_engine_find_tensor(e, Layer, Expert, moe_PROJ_UP, &info)) << "up missing";
    ASSERT_TRUE(moe_pc_engine_find_tensor(e, Layer, Expert, moe_PROJ_DOWN, &info)) << "down missing";

    std::vector<float> hidden(Hidden, 0.0f), gate(Inter, 0.0f), up(Inter, 0.0f), out(Hidden, -1.0f);
    hidden[0] = 4.0f;
    ASSERT_TRUE(moe_pc_engine_run_expert_triplet_f32(
        e, Layer, Expert, hidden.data(), Hidden, gate.data(), up.data(), Inter, out.data(), Hidden))
        << "expert triplet execution failed";

    EXPECT_NEAR(out[0], 3.0f * (silu(4.0f) * 8.0f), 1e-4f);
    for (uint32_t i = 1; i < 16; ++i) EXPECT_FLOAT_EQ(out[i], 0.0f);
    moe_pc_engine_destroy(e);
}

TEST(EngineFixtureLoading, MissingWeightPartFailsBeforeExecution) {
    const Fix fx = make_fixture("storagellm_engine_fixture_missing_part", false);
    ASSERT_TRUE(fx.ok) << fx.error;

    moe_pc_engine_config_t cfg = cpu_cfg();
    moe_pc_engine_t* e = moe_pc_engine_create(&cfg);
    ASSERT_NE(e, nullptr);
    ASSERT_TRUE(moe_pc_engine_set_model_root(e, fx.root.c_str()));
    ASSERT_TRUE(moe_pc_engine_load_codec_table(e, fx.index.c_str(), fx.root.c_str(), nullptr));

    std::vector<float> hidden(Hidden, 0.0f), gate(Inter, 0.0f), up(Inter, 0.0f), out(Hidden, 0.0f);
    hidden[0] = 4.0f;
    EXPECT_FALSE(moe_pc_engine_run_expert_triplet_f32(
        e, Layer, Expert, hidden.data(), Hidden, gate.data(), up.data(), Inter, out.data(), Hidden));
    moe_pc_engine_destroy(e);
}
