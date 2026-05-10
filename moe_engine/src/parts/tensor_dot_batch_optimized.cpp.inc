// ============================================================================
// Optimized Batch Dot Product Kernels (Bug B4)
// Implements true batched micro-kernels with multi-way accumulation
// ============================================================================

// 4-way batched FP32 dot product
// Processes 4 tokens simultaneously for better register utilization
static void moe_dot_raw_fp32_batch4(
    const float* weights,  // [in_dim]
    const float* x0,       // [in_dim]
    const float* x1,       // [in_dim]
    const float* x2,       // [in_dim]
    const float* x3,       // [in_dim]
    uint32_t in_dim,
    float* out             // [4]
) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Main loop: process 4 elements at a time for better ILP
    uint32_t i = 0;
    for (; i + 3 < in_dim; i += 4) {
        const float w0 = weights[i + 0];
        const float w1 = weights[i + 1];
        const float w2 = weights[i + 2];
        const float w3 = weights[i + 3];

        acc0 += x0[i + 0] * w0 + x0[i + 1] * w1 + x0[i + 2] * w2 + x0[i + 3] * w3;
        acc1 += x1[i + 0] * w0 + x1[i + 1] * w1 + x1[i + 2] * w2 + x1[i + 3] * w3;
        acc2 += x2[i + 0] * w0 + x2[i + 1] * w1 + x2[i + 2] * w2 + x2[i + 3] * w3;
        acc3 += x3[i + 0] * w0 + x3[i + 1] * w1 + x3[i + 2] * w2 + x3[i + 3] * w3;
    }

    // Tail loop: process remaining elements
    for (; i < in_dim; ++i) {
        const float w = weights[i];
        acc0 += x0[i] * w;
        acc1 += x1[i] * w;
        acc2 += x2[i] * w;
        acc3 += x3[i] * w;
    }

    out[0] = acc0;
    out[1] = acc1;
    out[2] = acc2;
    out[3] = acc3;
}

// 8-way batched FP32 dot product
// For larger batches, processes 8 tokens simultaneously
static void moe_dot_raw_fp32_batch8(
    const float* weights,  // [in_dim]
    const float* const* batch_x,  // [8][in_dim]
    uint32_t in_dim,
    float* out             // [8]
) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const float* x0 = batch_x[0];
    const float* x1 = batch_x[1];
    const float* x2 = batch_x[2];
    const float* x3 = batch_x[3];
    const float* x4 = batch_x[4];
    const float* x5 = batch_x[5];
    const float* x6 = batch_x[6];
    const float* x7 = batch_x[7];

    // Main loop: process 4 elements at a time
    uint32_t i = 0;
    for (; i + 3 < in_dim; i += 4) {
        const float w0 = weights[i + 0];
        const float w1 = weights[i + 1];
        const float w2 = weights[i + 2];
        const float w3 = weights[i + 3];

        acc0 += x0[i + 0] * w0 + x0[i + 1] * w1 + x0[i + 2] * w2 + x0[i + 3] * w3;
        acc1 += x1[i + 0] * w0 + x1[i + 1] * w1 + x1[i + 2] * w2 + x1[i + 3] * w3;
        acc2 += x2[i + 0] * w0 + x2[i + 1] * w1 + x2[i + 2] * w2 + x2[i + 3] * w3;
        acc3 += x3[i + 0] * w0 + x3[i + 1] * w1 + x3[i + 2] * w2 + x3[i + 3] * w3;
        acc4 += x4[i + 0] * w0 + x4[i + 1] * w1 + x4[i + 2] * w2 + x4[i + 3] * w3;
        acc5 += x5[i + 0] * w0 + x5[i + 1] * w1 + x5[i + 2] * w2 + x5[i + 3] * w3;
        acc6 += x6[i + 0] * w0 + x6[i + 1] * w1 + x6[i + 2] * w2 + x6[i + 3] * w3;
        acc7 += x7[i + 0] * w0 + x7[i + 1] * w1 + x7[i + 2] * w2 + x7[i + 3] * w3;
    }

    // Tail loop
    for (; i < in_dim; ++i) {
        const float w = weights[i];
        acc0 += x0[i] * w;
        acc1 += x1[i] * w;
        acc2 += x2[i] * w;
        acc3 += x3[i] * w;
        acc4 += x4[i] * w;
        acc5 += x5[i] * w;
        acc6 += x6[i] * w;
        acc7 += x7[i] * w;
    }

    out[0] = acc0;
    out[1] = acc1;
    out[2] = acc2;
    out[3] = acc3;
    out[4] = acc4;
    out[5] = acc5;
    out[6] = acc6;
    out[7] = acc7;
}

// Optimized batch dispatcher for FP32
// Chooses best kernel based on batch size
static void moe_dot_raw_fp32_batch_optimized(
    const float* weights,
    const float** batch_x,
    uint32_t batch_size,
    uint32_t in_dim,
    float* out_values
) {
    uint32_t b = 0;

    // Process in chunks of 8
    while (b + 8 <= batch_size) {
        // Check all 8 pointers are valid
        bool all_valid = true;
        for (uint32_t i = 0; i < 8; ++i) {
            if (!batch_x[b + i]) {
                all_valid = false;
                break;
            }
        }

        if (all_valid) {
            moe_dot_raw_fp32_batch8(weights, batch_x + b, in_dim, out_values + b);
            b += 8;
        } else {
            // Fallback to scalar for invalid pointers
            for (uint32_t i = 0; i < 8; ++i) {
                if (batch_x[b]) {
                    out_values[b] = moe_dot_raw_fp32_row(
                        reinterpret_cast<const uint8_t*>(weights),
                        batch_x[b],
                        in_dim
                    );
                } else {
                    out_values[b] = 0.0f;
                }
                ++b;
            }
        }
    }

    // Process in chunks of 4
    while (b + 4 <= batch_size) {
        // Check all 4 pointers are valid
        if (batch_x[b] && batch_x[b+1] && batch_x[b+2] && batch_x[b+3]) {
            float temp[4];
            moe_dot_raw_fp32_batch4(
                weights,
                batch_x[b], batch_x[b+1], batch_x[b+2], batch_x[b+3],
                in_dim,
                temp
            );
            out_values[b] = temp[0];
            out_values[b+1] = temp[1];
            out_values[b+2] = temp[2];
            out_values[b+3] = temp[3];
            b += 4;
        } else {
            // Fallback to scalar
            for (uint32_t i = 0; i < 4; ++i) {
                if (batch_x[b]) {
                    out_values[b] = moe_dot_raw_fp32_row(
                        reinterpret_cast<const uint8_t*>(weights),
                        batch_x[b],
                        in_dim
                    );
                } else {
                    out_values[b] = 0.0f;
                }
                ++b;
            }
        }
    }

    // Process remaining elements (< 4)
    while (b < batch_size) {
        if (batch_x[b]) {
            out_values[b] = moe_dot_raw_fp32_row(
                reinterpret_cast<const uint8_t*>(weights),
                batch_x[b],
                in_dim
            );
        } else {
            out_values[b] = 0.0f;
        }
        ++b;
    }
}
