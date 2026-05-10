#pragma once

#include "cpu_activation.h"
#include "cpu_tensor_view.h"
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace storagellm::cpu_moe {

#if defined(__AVX2__)
inline float horizontal_sum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

#if defined(__AVX512F__)
inline float horizontal_sum_ps(__m512 v) {
#if defined(__AVX512DQ__)
    return _mm512_reduce_add_ps(v);
#else
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    float sum = 0.0f;
    for (float x : tmp) sum += x;
    return sum;
#endif
}
#endif

inline void dot_gate_up_f32(
    const float* input,
    const float* gate_row,
    const float* up_row,
    uint32_t H,
    float& g,
    float& u
) {
    g = 0.0f;
    u = 0.0f;
    uint32_t h = 0;
#if defined(__AVX512F__)
    __m512 g16 = _mm512_setzero_ps();
    __m512 u16 = _mm512_setzero_ps();
    for (; h + 15 < H; h += 16) {
        const __m512 x = _mm512_loadu_ps(input + h);
        const __m512 gw = _mm512_loadu_ps(gate_row + h);
        const __m512 uw = _mm512_loadu_ps(up_row + h);
    #if defined(__FMA__) || defined(_MSC_VER)
        g16 = _mm512_fmadd_ps(gw, x, g16);
        u16 = _mm512_fmadd_ps(uw, x, u16);
    #else
        g16 = _mm512_add_ps(g16, _mm512_mul_ps(gw, x));
        u16 = _mm512_add_ps(u16, _mm512_mul_ps(uw, x));
    #endif
    }
    g = horizontal_sum_ps(g16);
    u = horizontal_sum_ps(u16);
#elif defined(__AVX2__)
    __m256 g8 = _mm256_setzero_ps();
    __m256 u8 = _mm256_setzero_ps();
    for (; h + 7 < H; h += 8) {
        const __m256 x = _mm256_loadu_ps(input + h);
        const __m256 gw = _mm256_loadu_ps(gate_row + h);
        const __m256 uw = _mm256_loadu_ps(up_row + h);
    #if defined(__FMA__) || defined(_MSC_VER)
        g8 = _mm256_fmadd_ps(gw, x, g8);
        u8 = _mm256_fmadd_ps(uw, x, u8);
    #else
        g8 = _mm256_add_ps(g8, _mm256_mul_ps(gw, x));
        u8 = _mm256_add_ps(u8, _mm256_mul_ps(uw, x));
    #endif
    }
    g = horizontal_sum_ps(g8);
    u = horizontal_sum_ps(u8);
#endif
    for (; h < H; ++h) {
        const float x = input[h];
        g += gate_row[h] * x;
        u += up_row[h] * x;
    }
}

inline float dot_down_f32(const float* down_row, const float* mid, uint32_t I) {
    float y = 0.0f;
    uint32_t r = 0;
#if defined(__AVX512F__)
    __m512 y16 = _mm512_setzero_ps();
    for (; r + 15 < I; r += 16) {
        const __m512 dw = _mm512_loadu_ps(down_row + r);
        const __m512 mv = _mm512_loadu_ps(mid + r);
    #if defined(__FMA__) || defined(_MSC_VER)
        y16 = _mm512_fmadd_ps(dw, mv, y16);
    #else
        y16 = _mm512_add_ps(y16, _mm512_mul_ps(dw, mv));
    #endif
    }
    y = horizontal_sum_ps(y16);
#elif defined(__AVX2__)
    __m256 y8 = _mm256_setzero_ps();
    for (; r + 7 < I; r += 8) {
        const __m256 dw = _mm256_loadu_ps(down_row + r);
        const __m256 mv = _mm256_loadu_ps(mid + r);
    #if defined(__FMA__) || defined(_MSC_VER)
        y8 = _mm256_fmadd_ps(dw, mv, y8);
    #else
        y8 = _mm256_add_ps(y8, _mm256_mul_ps(dw, mv));
    #endif
    }
    y = horizontal_sum_ps(y8);
#endif
    for (; r < I; ++r) {
        y += down_row[r] * mid[r];
    }
    return y;
}


inline void dot_gate_up_weighted(
    const float* input,
    const WeightMatrixView& gate,
    const WeightMatrixView& up,
    uint32_t row,
    uint32_t H,
    float& g,
    float& u
) {
    const float* gate_row = weight_ptr_fp32(gate, row);
    const float* up_row = weight_ptr_fp32(up, row);
    if (gate_row && up_row) {
        dot_gate_up_f32(input, gate_row, up_row, H, g, u);
        return;
    }
    g = 0.0f;
    u = 0.0f;
    for (uint32_t h = 0; h < H; ++h) {
        const float x = input[h];
        g += weight_value(gate, row, h) * x;
        u += weight_value(up, row, h) * x;
    }
}

inline float dot_down_weighted(
    const WeightMatrixView& down,
    uint32_t row,
    const float* mid,
    uint32_t I
) {
    const float* down_row = weight_ptr_fp32(down, row);
    if (down_row) {
        return dot_down_f32(down_row, mid, I);
    }
    float y = 0.0f;
    for (uint32_t r = 0; r < I; ++r) {
        y += weight_value(down, row, r) * mid[r];
    }
    return y;
}


inline void compute_assignment_output_f32(
    const ValidatedTask& ctx,
    uint32_t local_row,
    std::vector<float>& mid,
    float* out_hidden
) {
    const auto& task = *ctx.task;
    const uint32_t H = task.hidden_size;
    const uint32_t I = task.intermediate_size;
    if (!out_hidden) return;
    std::fill(out_hidden, out_hidden + H, 0.0f);
    mid.resize(I);

    const uint32_t global_row = task.assignment_offset + local_row;
    const uint32_t token = task.d_token_indices[global_row];
    const float route_weight = task.d_token_weights[global_row];
    if (!std::isfinite(route_weight)) {
        return;
    }

    const auto* input = static_cast<const float*>(task.d_input) +
        static_cast<uint64_t>(token) * task.input_stride;

    for (uint32_t r = 0; r < I; ++r) {
        float g = 0.0f;
        float u = 0.0f;
        dot_gate_up_weighted(input, ctx.gate, ctx.up, r, H, g, u);
        mid[r] = activation(task.activation_mode, g, u);
    }

    for (uint32_t h = 0; h < H; ++h) {
        out_hidden[h] = dot_down_weighted(ctx.down, h, mid.data(), I) * route_weight;
    }
}

inline void run_assignment_range(
    const ValidatedTask& ctx,
    uint32_t begin,
    uint32_t end,
    std::vector<float>& mid
) {
    const auto& task = *ctx.task;
    const uint32_t H = task.hidden_size;
    const uint32_t I = task.intermediate_size;
    mid.resize(I);

    for (uint32_t local_row = begin; local_row < end; ++local_row) {
        const uint32_t global_row = task.assignment_offset + local_row;
        const uint32_t token = task.d_token_indices[global_row];
        const float route_weight = task.d_token_weights[global_row];
        if (!std::isfinite(route_weight)) {
            continue;
        }

        const auto* input = static_cast<const float*>(task.d_input) +
            static_cast<uint64_t>(token) * task.input_stride;
        auto* accum = static_cast<float*>(task.d_accum) +
            static_cast<uint64_t>(token) * task.accum_stride;

        for (uint32_t r = 0; r < I; ++r) {
            float g = 0.0f;
            float u = 0.0f;
            dot_gate_up_weighted(input, ctx.gate, ctx.up, r, H, g, u);
            mid[r] = activation(task.activation_mode, g, u);
        }

        for (uint32_t h = 0; h < H; ++h) {
            const float y = dot_down_weighted(ctx.down, h, mid.data(), I);
            accum[h] += y * route_weight;
        }
    }
}

} // namespace storagellm::cpu_moe
