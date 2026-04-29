#pragma once

// Lloyd-Max Codebook Generation for Beta Distribution
// Paper: TurboQuant Section 3.1, Equation 4

// Compute Lloyd-Max codebook for given bit-width and dimension
void qkv_compute_lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int bits,
    int dim
);

// Find nearest centroid index for a value using binary search
int qkv_find_nearest_centroid(
    float val,
    const float* centroids,
    const float* thresholds,
    int n_levels
);
