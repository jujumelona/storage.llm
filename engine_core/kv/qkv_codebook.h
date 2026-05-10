#pragma once

// Lloyd-Max Codebook Generation for Beta Distribution
// Paper: TurboQuant Section 3.1, Equation 4

// Codebook distributions. Exact Beta is the paper-conformant default.
#define QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA 0u
#define QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX 1u

// Compute Lloyd-Max codebook for given bit-width and dimension.
// This wrapper keeps the paper-conformant exact Beta distribution default.
void qkv_compute_lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int bits,
    int dim
);

void qkv_compute_lloyd_max_codebook_ex(
    float* centroids,
    float* thresholds,
    int bits,
    int dim,
    unsigned distribution
);

// Find nearest centroid index for a value using binary search
int qkv_find_nearest_centroid(
    float val,
    const float* centroids,
    const float* thresholds,
    int n_levels
);
