#pragma once

#include <stdint.h>

// Random Matrix Generation for QKV Quantization

bool qkv_dim_is_power_of_two(int dim);

void qkv_fwht_inplace(float* data, int dim);

int qkv_apply_hadamard_rotation_forward(
    const float* input,
    const float* signs,
    float* output,
    int dim
);

int qkv_apply_hadamard_rotation_inverse(
    const float* input,
    const float* signs,
    float* output,
    int dim
);

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
