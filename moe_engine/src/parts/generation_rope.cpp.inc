// Bug #1 Fix: Full MLA attention with Q projection, RoPE, scaled dot-product.
// GLM4/Moe Multi-head Latent Attention (MLA):
//   KV path: hidden -> kv_a_proj(576) -> [kv_a_norm(512), mqa_rope(64)]
//            kv_a_norm -> kv_b_proj(28672) -> [K_nope(12288), V(16384)]
//   Q path:  hidden -> q_a_proj(2048) -> q_a_norm(2048) -> q_b_proj(12288+4096=16384)
//            -> [Q_nope(12288), Q_rope(4096)] split by heads
//   Score = softmax(Q_nope*K_nope + Q_rope*K_rope) * V -> o_proj -> attn output

// Keep this tied to the converted model metadata; tensor shapes alone do not determine theta.
static void moe_rope_inplace(
    float* data,
    uint32_t dim,
    uint32_t position,
    const std::vector<float>& cos_table,
    const std::vector<float>& sin_table,
    float base = moe_ROPE_THETA
) {
    // BUGFIX 224: Validate dim > 0 to prevent division by zero in frequency calculation ★★
    // Problem: dim=0 causes division by zero in freq = 1.0f / std::pow(base, (float)i / (float)dim)
    // Solution: Return early if dim == 0 or data == nullptr
    // Impact: Prevents crash in RoPE application with invalid config
    if (!data || dim == 0) {
        return;
    }
    const uint32_t half_dim = dim / 2u;
    const size_t row_offset = (size_t)position * (size_t)half_dim;
    const size_t table_len = half_dim ? cos_table.size() / half_dim : 0;
    if ((size_t)position < table_len &&
        cos_table.size() >= row_offset + half_dim &&
        sin_table.size() >= row_offset + half_dim) {
        const float* cos_row = cos_table.data() + row_offset;
        const float* sin_row = sin_table.data() + row_offset;
        for (uint32_t j = 0; j < half_dim; ++j) {
            const uint32_t i = j * 2u;
            const float cos_t = cos_row[j];
            const float sin_t = sin_row[j];
            const float x0 = data[i];
            const float x1 = data[i + 1u];
            data[i] = x0 * cos_t - x1 * sin_t;
            data[i + 1u] = x0 * sin_t + x1 * cos_t;
        }
        return;
    }
    for (uint32_t i = 0; i + 1u < dim; i += 2u) {
        const float freq = 1.0f / std::pow(base, (float)i / (float)dim);
        const float theta = (float)position * freq;
        const float cos_t = std::cos(theta);
        const float sin_t = std::sin(theta);
        const float x0 = data[i];
        const float x1 = data[i + 1];
        data[i]     = x0 * cos_t - x1 * sin_t;
        data[i + 1] = x0 * sin_t + x1 * cos_t;
    }
}
