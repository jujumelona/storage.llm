#pragma once

#include <stdint.h>

// Random Matrix Generation for QKV Quantization

// Generate random rotation matrix Pi via QR decomposition
// For power-of-2 dimensions: use fast Hadamard-like construction
// For non-power-of-2: use QR decomposition of Gaussian random matrix
void qkv_generate_rotation_matrix(
    float* Pi,
    int dim,
    uint64_t seed
);

// Generate QJL (Quantized Johnson-Lindenstrauss) matrix
// S_ij ~ N(0, 1) per paper Lemma 4
void qkv_generate_qjl_matrix(
    float* S,
    int dim,
    uint64_t seed
);
