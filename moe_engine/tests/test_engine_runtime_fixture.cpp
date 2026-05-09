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
#endif
#include "../include/moe_pc_engine.h"

namespace {
constexpr uint32_t L = 10, E = 0, H = 6144, I = 2048;
constexpr uint64_t GateBytes = uint64_t(I) * H * sizeof(float);
constexpr uint64_t UpBytes = GateBytes;
constexpr uint64_t DownBytes = uint64_t(H) * I * sizeof(float);
constexpr uint64_t GateOff = 0;
constexpr uint64_t UpOff = GateOff + GateBytes;
constexpr uint64_t DownOff = UpOff + UpBytes;
constexpr uint64_t TotalBytes = DownOff + DownBytes;

void u16(std::ofstream& f, uint16_t v){ f.write(reinterpret_cast<const char*>(&v),2); }
void u32(std::ofstream& f, uint32_t v){ f.write(reinterpret_cast<const char*>(&v),4); }
void u64(std::ofstream& f, uint64_t v){ f.write(reinterpret_cast<const char*>(&v),8); }
void str16(std::ofstream& f, const std::string& s){ u16(f, uint16_t(s.size())); f.write(s.data(), std::streamsize(s.size())); }
void f32_at(std::fstream& f, uint64_t off, float v){ f.seekp(std::streamoff(off)); f.write(reinterpret_cast<const char*>(&v),4); }

bool mkdir_once(const std::string& p){
#ifdef _WIN32
  return _mkdir(p.c_str()) == 0 || errno == EEXIST;
#else
  return mkdir(p.c_str(),0755) == 0 || errno == EEXIST;
#endif
}

void rec(std::ofstream& out, uint64_t off, uint64_t bytes, uint32_t proj, uint32_t rows, uint32_t cols){
  u64(out,off); u64(out,bytes); u64(out,0); u64(out,0); u64(out,0); u64(out,0);
  u32(out,1); u32(out,1); u32(out,L); u32(out,E); u32(out,proj); u32(out,rows); u32(out,cols);
  u32(out, cols >= 16 ? cols / 16 : 1); u32(out, cols >= 16 ? 16 : cols);
  u16(out,0); u16(out,0); u32(out,0); u32(out,0xFFFFFFFFu); u32(out,moe_WEIGHT_ENCODING_RAW_FP32);
}

struct Fix { std::string root, part, index; bool ok = true; };
Fix make_fixture(const char* name, bool with_part){
  Fix x; x.root = std::string(::testing::TempDir()) + name; x.part = x.root + "/tiny-part.bin"; x.index = x.root + "/tiny.sltidx3";
  x.ok = mkdir_once(x.root);
  if(with_part){
    std::fstream p(x.part, std::ios::in|std::ios::out|std::ios::binary|std::ios::trunc);
    x.ok = x.ok && p.good();
    if (p.good()) {
      p.seekp(std::streamoff(TotalBytes - 1)); char z=0; p.write(&z,1);
      f32_at(p,GateOff,1.0f); f32_at(p,UpOff,2.0f); f32_at(p,DownOff,3.0f);
      x.ok = x.ok && p.good();
    }
  }
  std::ofstream idx(x.index, std::ios::binary); const char magic[8]={'S','L','T','I','D','X','3','\0'}; idx.write(magic,8);
  u32(idx,3); u32(idx,1); u32(idx,3); u32(idx,0); str16(idx,"tiny-part.bin");
  rec(idx,GateOff,GateBytes,moe_PROJ_GATE,I,H); rec(idx,UpOff,UpBytes,moe_PROJ_UP,I,H); rec(idx,DownOff,DownBytes,moe_PROJ_DOWN,H,I);
  x.ok = x.ok && idx.good(); return x;
}

moe_pc_engine_config_t cfg_cpu(){ auto c=moe_pc_default_config(); c.preferred_backend=moe_BACKEND_CPU; c.platform=moe_PLATFORM_CPU_ONLY; c.vram_budget_bytes=64ull<<20; c.ram_budget_bytes=512ull<<20; return c; }
float silu(float v){ return v/(1.0f+std::exp(-v)); }
}

TEST(EngineRuntimeFixture, LoadsSparseIndexAndExecutesCpuExpert){
  Fix fx = make_fixture("storagellm_runtime_fixture_ok", true); ASSERT_TRUE(fx.ok);
  auto cfg = cfg_cpu(); moe_pc_engine_t* e = moe_pc_engine_create(&cfg); ASSERT_NE(e,nullptr);
  ASSERT_TRUE(moe_pc_engine_set_model_root(e, fx.root.c_str()));
  ASSERT_TRUE(moe_pc_engine_load_codec_table(e, fx.index.c_str(), fx.root.c_str(), nullptr));
  EXPECT_EQ(moe_pc_engine_tensor_count(e), 3u);
  moe_forward_status_t st{}; ASSERT_TRUE(moe_pc_engine_get_forward_status(e,&st)); EXPECT_TRUE(st.tensor_table_loaded); EXPECT_TRUE(st.expert_triplet_available);
  std::vector<float> h(H,0), g(I,0), u(I,0), o(H,-1); h[0]=4.0f;
  ASSERT_TRUE(moe_pc_engine_run_expert_triplet_f32(e,L,E,h.data(),H,g.data(),u.data(),I,o.data(),H));
  EXPECT_NEAR(o[0], 3.0f * (silu(4.0f) * 8.0f), 1e-4f);
  for(uint32_t i=1;i<16;i++) EXPECT_FLOAT_EQ(o[i],0.0f);
  moe_pc_engine_destroy(e);
}

TEST(EngineRuntimeFixture, MissingPartFailsAtExecution){
  Fix fx = make_fixture("storagellm_runtime_fixture_missing", false); ASSERT_TRUE(fx.ok);
  auto cfg = cfg_cpu(); moe_pc_engine_t* e = moe_pc_engine_create(&cfg); ASSERT_NE(e,nullptr);
  ASSERT_TRUE(moe_pc_engine_set_model_root(e, fx.root.c_str()));
  ASSERT_TRUE(moe_pc_engine_load_codec_table(e, fx.index.c_str(), fx.root.c_str(), nullptr));
  std::vector<float> h(H,0), g(I,0), u(I,0), o(H,0); h[0]=4.0f;
  EXPECT_FALSE(moe_pc_engine_run_expert_triplet_f32(e,L,E,h.data(),H,g.data(),u.data(),I,o.data(),H));
  moe_pc_engine_destroy(e);
}
