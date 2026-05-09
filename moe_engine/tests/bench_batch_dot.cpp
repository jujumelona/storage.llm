// Batch Dot Product Microbenchmark (Bug B4)
// Tests batch scaling performance of dot product kernels

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

// Test configuration
constexpr int WARMUP_ITERS = 200;
constexpr int MEASURE_ITERS = 1000;
constexpr int INPUT_DIM = 4096;  // Typical hidden dimension

// Simple FP32 dot product (baseline)
static float dot_scalar(const float* w, const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += w[i] * x[i];
    }
    return sum;
}

// Naive batch implementation (current)
static void dot_batch_naive(
    const float* w,
    const float** batch_x,
    int batch_size,
    int n,
    float* out
) {
    for (int b = 0; b < batch_size; ++b) {
        out[b] = dot_scalar(w, batch_x[b], n);
    }
}

// Optimized 4-way batch
static void dot_batch4(
    const float* w,
    const float* x0,
    const float* x1,
    const float* x2,
    const float* x3,
    int n,
    float* out
) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int i = 0;
    for (; i + 3 < n; i += 4) {
        const float w0 = w[i + 0];
        const float w1 = w[i + 1];
        const float w2 = w[i + 2];
        const float w3 = w[i + 3];

        acc0 += x0[i + 0] * w0 + x0[i + 1] * w1 + x0[i + 2] * w2 + x0[i + 3] * w3;
        acc1 += x1[i + 0] * w0 + x1[i + 1] * w1 + x1[i + 2] * w2 + x1[i + 3] * w3;
        acc2 += x2[i + 0] * w0 + x2[i + 1] * w1 + x2[i + 2] * w2 + x2[i + 3] * w3;
        acc3 += x3[i + 0] * w0 + x3[i + 1] * w1 + x3[i + 2] * w2 + x3[i + 3] * w3;
    }

    for (; i < n; ++i) {
        const float wv = w[i];
        acc0 += x0[i] * wv;
        acc1 += x1[i] * wv;
        acc2 += x2[i] * wv;
        acc3 += x3[i] * wv;
    }

    out[0] = acc0;
    out[1] = acc1;
    out[2] = acc2;
    out[3] = acc3;
}

// Optimized batch dispatcher
static void dot_batch_optimized(
    const float* w,
    const float** batch_x,
    int batch_size,
    int n,
    float* out
) {
    int b = 0;

    // Process in chunks of 4
    while (b + 4 <= batch_size) {
        float temp[4];
        dot_batch4(w, batch_x[b], batch_x[b+1], batch_x[b+2], batch_x[b+3], n, temp);
        out[b] = temp[0];
        out[b+1] = temp[1];
        out[b+2] = temp[2];
        out[b+3] = temp[3];
        b += 4;
    }

    // Process remaining
    while (b < batch_size) {
        out[b] = dot_scalar(w, batch_x[b], n);
        ++b;
    }
}

// Benchmark runner
struct BenchResult {
    double time_ms;
    double gflops;
    double speedup;
};

BenchResult benchmark(
    const char* name,
    void (*func)(const float*, const float**, int, int, float*),
    const float* weights,
    const float** batch_x,
    int batch_size,
    int input_dim,
    float* output,
    double baseline_time_ms = 0.0
) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        func(weights, batch_x, batch_size, input_dim, output);
    }

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MEASURE_ITERS; ++i) {
        func(weights, batch_x, batch_size, input_dim, output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_call_ms = elapsed_ms / MEASURE_ITERS;

    // Calculate GFLOPS: 2 * batch_size * input_dim operations per call
    double ops = 2.0 * batch_size * input_dim;
    double gflops = (ops / 1e9) / (time_per_call_ms / 1000.0);

    double speedup = baseline_time_ms > 0.0 ? baseline_time_ms / time_per_call_ms : 1.0;

    return {time_per_call_ms, gflops, speedup};
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << "Batch Dot Product Microbenchmark (Bug B4)\n";
    std::cout << "=============================================================\n";
    std::cout << "Input dimension: " << INPUT_DIM << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Measure iterations: " << MEASURE_ITERS << "\n";
    std::cout << "=============================================================\n\n";

    // Initialize random data
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> weights(INPUT_DIM);
    for (auto& w : weights) w = dist(rng);

    // Test different batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};

    for (int batch_size : batch_sizes) {
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "-------------------------------------------------------------\n";

        // Allocate batch data
        std::vector<std::vector<float>> batch_data(batch_size);
        std::vector<const float*> batch_ptrs(batch_size);

        for (int b = 0; b < batch_size; ++b) {
            batch_data[b].resize(INPUT_DIM);
            for (auto& x : batch_data[b]) x = dist(rng);
            batch_ptrs[b] = batch_data[b].data();
        }

        std::vector<float> output_naive(batch_size);
        std::vector<float> output_optimized(batch_size);

        // Benchmark naive implementation
        auto result_naive = benchmark(
            "Naive",
            dot_batch_naive,
            weights.data(),
            batch_ptrs.data(),
            batch_size,
            INPUT_DIM,
            output_naive.data()
        );

        // Benchmark optimized implementation
        auto result_optimized = benchmark(
            "Optimized",
            dot_batch_optimized,
            weights.data(),
            batch_ptrs.data(),
            batch_size,
            INPUT_DIM,
            output_optimized.data(),
            result_naive.time_ms
        );

        // Verify correctness
        float max_diff = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            float diff = std::abs(output_naive[b] - output_optimized[b]);
            if (diff > max_diff) max_diff = diff;
        }

        // Print results
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Naive:      " << std::setw(8) << result_naive.time_ms << " ms  "
                  << std::setw(6) << result_naive.gflops << " GFLOPS\n";
        std::cout << "  Optimized:  " << std::setw(8) << result_optimized.time_ms << " ms  "
                  << std::setw(6) << result_optimized.gflops << " GFLOPS  "
                  << std::setw(5) << result_optimized.speedup << "x speedup\n";
        std::cout << "  Max diff:   " << std::scientific << max_diff << "\n";

        // Check for performance regression
        if (batch_size >= 4 && result_optimized.speedup < 1.1) {
            std::cout << "  WARNING: Expected >1.1x speedup for batch >= 4\n";
        }

        std::cout << "\n";
    }

    std::cout << "=============================================================\n";
    std::cout << "Expected behavior:\n";
    std::cout << "  - Batch 1-2: Similar performance (overhead dominates)\n";
    std::cout << "  - Batch 4:   ~1.5-2x speedup (4-way accumulation)\n";
    std::cout << "  - Batch 8:   ~2-3x speedup (better register utilization)\n";
    std::cout << "  - Batch 16+: ~2-4x speedup (sustained throughput)\n";
    std::cout << "=============================================================\n";

    return 0;
}
