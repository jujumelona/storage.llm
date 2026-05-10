#pragma once

#include "cpu_activation.h"
#include "cpu_tensor_view.h"
#include <cstdint>
#include <vector>

namespace storagellm::cpu_moe {

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
            const float* gate_row = ctx.gate + static_cast<uint64_t>(r) * H;
            const float* up_row = ctx.up + static_cast<uint64_t>(r) * H;
            float g = 0.0f;
            float u = 0.0f;
            uint32_t h = 0;
            for (; h + 3 < H; h += 4) {
                const float x0 = input[h + 0];
                const float x1 = input[h + 1];
                const float x2 = input[h + 2];
                const float x3 = input[h + 3];
                g += gate_row[h + 0] * x0 + gate_row[h + 1] * x1 +
                     gate_row[h + 2] * x2 + gate_row[h + 3] * x3;
                u += up_row[h + 0] * x0 + up_row[h + 1] * x1 +
                     up_row[h + 2] * x2 + up_row[h + 3] * x3;
            }
            for (; h < H; ++h) {
                const float x = input[h];
                g += gate_row[h] * x;
                u += up_row[h] * x;
            }
            mid[r] = activation(task.activation_mode, g, u);
        }

        for (uint32_t h = 0; h < H; ++h) {
            const float* down_row = ctx.down + static_cast<uint64_t>(h) * I;
            float y = 0.0f;
            uint32_t r = 0;
            for (; r + 3 < I; r += 4) {
                y += down_row[r + 0] * mid[r + 0] + down_row[r + 1] * mid[r + 1] +
                     down_row[r + 2] * mid[r + 2] + down_row[r + 3] * mid[r + 3];
            }
            for (; r < I; ++r) {
                y += down_row[r] * mid[r];
            }
            accum[h] += y * route_weight;
        }
    }
}

} // namespace storagellm::cpu_moe
