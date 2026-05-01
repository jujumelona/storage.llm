// MLA constants for the legacy metadata fallback path.
static constexpr uint32_t moe_MLA_KV_COMPRESSED_DIM = 512u;
static constexpr uint32_t moe_MLA_K_ROPE_DIM = 64u;
static constexpr uint32_t moe_MLA_KV_A_PROJ_DIM = moe_MLA_KV_COMPRESSED_DIM + moe_MLA_K_ROPE_DIM; // 576
static constexpr uint32_t moe_MLA_KV_ENTRY_DIM = moe_MLA_KV_A_PROJ_DIM;
static constexpr uint32_t moe_MLA_MAX_SEQ_LEN = 8192u;
static constexpr uint32_t moe_MLA_ROPE_TABLE_LEN = 16384u;
// Bonus: Enforce consistency between ROPE table and KV cache max length
static_assert(moe_MLA_ROPE_TABLE_LEN >= moe_MLA_MAX_SEQ_LEN,
              "ROPE table must cover at least MAX_SEQ_LEN positions");
static constexpr uint32_t moe_MLA_KV_INITIAL_CHUNK = 1024u;
static constexpr uint32_t moe_MLA_NUM_HEADS = 64u;
static constexpr uint32_t moe_MLA_HEAD_DIM = 256u;
static constexpr uint32_t moe_MLA_NOPE_DIM = 192u;
static constexpr uint32_t moe_MLA_ROPE_DIM = 64u;
static constexpr uint32_t moe_MLA_V_HEAD_DIM = 256u;
static constexpr uint32_t moe_MLA_Q_B_DIM = moe_MLA_NUM_HEADS * moe_MLA_HEAD_DIM; // 16384
static constexpr uint32_t moe_MLA_K_NOPE_TOTAL_DIM = moe_MLA_NUM_HEADS * moe_MLA_NOPE_DIM; // 12288
static constexpr uint32_t moe_MLA_V_TOTAL_DIM = moe_MLA_NUM_HEADS * moe_MLA_V_HEAD_DIM; // 16384
static_assert(moe_MLA_ROPE_DIM == moe_MLA_K_ROPE_DIM, "Q/K RoPE dims must match");
static_assert(moe_MLA_Q_B_DIM == 16384u, "Moe q_b projection width mismatch");
static_assert(moe_MLA_K_NOPE_TOTAL_DIM == 12288u, "Moe K_nope width mismatch");
static_assert(moe_MLA_V_TOTAL_DIM == 16384u, "Moe V width mismatch");
static constexpr float moe_ROPE_THETA = 500000.0f;
static constexpr uint32_t moe_MOE_BASE_TOP_EXPERTS = 8u;
static constexpr uint32_t moe_MOE_TOP_EXPERTS = 12u;
