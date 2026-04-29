#include "qkv_codebook.h"
#include <math.h>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Beta distribution PDF: f(x) proportional to (1 - x^2)^((d - 3) / 2)
// For large d, converges to N(0, 1/d)
static double beta_pdf(double x, int d) {
    if (x <= -1.0 || x >= 1.0) return 0.0;
    // For large d, use Gaussian approximation
    double variance = 1.0 / (double)d;
    double sigma = sqrt(variance);
    return exp(-x * x / (2.0 * variance)) / (sigma * sqrt(2.0 * M_PI));
}

// Lloyd-Max algorithm: solve continuous k-means (Eq. 4)
static void lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int n_levels,
    int dim,
    int max_iters
) {
    // Fix 1: Initialize centroids within N(0, 1/d) distribution range
    double sigma = 1.0 / sqrt((double)dim);
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = (float)((-3.5 + 7.0 * (double)i / (double)(n_levels - 1)) * sigma);
    }

    // Lloyd-Max iteration
    for (int iter = 0; iter < max_iters; iter++) {
        // Update thresholds (midpoints between centroids)
        thresholds[0] = (float)(-5.0 * sigma);
        for (int i = 1; i < n_levels; i++) {
            thresholds[i] = (centroids[i-1] + centroids[i]) / 2.0f;
        }
        thresholds[n_levels] = (float)(5.0 * sigma);

        // Update centroids (weighted mean in each region)
        bool converged = true;
        for (int i = 0; i < n_levels; i++) {
            double sum_x = 0.0, sum_w = 0.0;
            int n_samples = 1000;
            double step = (thresholds[i+1] - thresholds[i]) / n_samples;

            for (int j = 0; j <= n_samples; j++) {
                double x = thresholds[i] + j * step;
                double w = beta_pdf(x, dim);
                sum_x += x * w;
                sum_w += w;
            }

            if (sum_w > 1e-10) {
                float new_centroid = (float)(sum_x / sum_w);
                if (fabsf(new_centroid - centroids[i]) > 1e-6f) {
                    converged = false;
                }
                centroids[i] = new_centroid;
            }
        }

        if (converged) break;
    }
}

void qkv_compute_lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int bits,
    int dim
) {
    int n_levels = 1 << bits;

    // For 2-bit, paper gives explicit values
    if (bits == 2) {
        double scale = 1.0 / sqrt((double)dim);
        centroids[0] = (float)(-1.51 * scale);
        centroids[1] = (float)(-0.453 * scale);
        centroids[2] = (float)(0.453 * scale);
        centroids[3] = (float)(1.51 * scale);

        thresholds[0] = -std::numeric_limits<float>::infinity();
        thresholds[1] = (float)((-1.51 - 0.453) / 2.0 * scale);
        thresholds[2] = 0.0f;
        thresholds[3] = (float)((0.453 + 1.51) / 2.0 * scale);
        thresholds[4] = std::numeric_limits<float>::infinity();
        return;
    }

    lloyd_max_codebook(centroids, thresholds, n_levels, dim, 100);
}

int qkv_find_nearest_centroid(
    float val,
    const float* centroids,
    const float* thresholds,
    int n_levels
) {
    // Binary search through thresholds
    int lo = 0, hi = n_levels;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val < thresholds[mid + 1]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    // Fix: clamp to valid range [0, n_levels-1]
    return lo < n_levels ? lo : n_levels - 1;
}
