#include "qkv_codebook.h"
#include <math.h>
#include <limits>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double qkv_gaussian_approx_pdf(double x, int d) {
    if (d <= 0) return 0.0;
    const double variance = 1.0 / (double)d;
    const double sigma = sqrt(variance);
    return exp(-x * x / (2.0 * variance)) / (sigma * sqrt(2.0 * M_PI));
}

static double qkv_exact_beta_pdf(double x, int d) {
    if (d <= 1 || x <= -1.0 || x >= 1.0) return 0.0;
    const double one_minus = std::max(0.0, 1.0 - x * x);
    if (one_minus <= 0.0) return 0.0;
    const double log_c = lgamma(0.5 * (double)d) -
        (0.5 * log(M_PI) + lgamma(0.5 * (double)(d - 1)));
    const double log_p = log_c + 0.5 * (double)(d - 3) * log(one_minus);
    return exp(log_p);
}

static double qkv_codebook_pdf(double x, int d, unsigned distribution) {
    if (distribution == QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX) {
        return qkv_gaussian_approx_pdf(x, d);
    }
    return qkv_exact_beta_pdf(x, d);
}

static void lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int n_levels,
    int dim,
    int max_iters,
    unsigned distribution
) {
    if (!centroids || !thresholds || n_levels <= 1 || dim <= 0 || max_iters <= 0) {
        return;
    }

    const double sigma = 1.0 / sqrt((double)dim);
    for (int i = 0; i < n_levels; i++) {
        centroids[i] = (float)((-3.5 + 7.0 * (double)i / (double)(n_levels - 1)) * sigma);
    }

    std::vector<float> bounds((size_t)n_levels + 1u);
    for (int iter = 0; iter < max_iters; iter++) {
        bounds[0] = -std::numeric_limits<float>::infinity();
        for (int i = 1; i < n_levels; i++) {
            bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0f;
        }
        bounds[n_levels] = std::numeric_limits<float>::infinity();

        bool converged = true;
        for (int i = 0; i < n_levels; i++) {
            const double lo = (i == 0) ? -1.0 : std::max(-1.0, (double)bounds[i]);
            const double hi = (i + 1 == n_levels) ? 1.0 : std::min(1.0, (double)bounds[i + 1]);
            if (!(hi > lo)) {
                continue;
            }
            const int n_samples = distribution == QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX ? 768 : 1536;
            const double step = (hi - lo) / (double)n_samples;
            double sum_x = 0.0;
            double sum_w = 0.0;

            for (int j = 0; j <= n_samples; j++) {
                const double x = lo + (double)j * step;
                const double w = qkv_codebook_pdf(x, dim, distribution);
                sum_x += x * w;
                sum_w += w;
            }

            if (sum_w > 1e-10) {
                const float new_centroid = (float)(sum_x / sum_w);
                if (fabsf(new_centroid - centroids[i]) > 1e-6f) {
                    converged = false;
                }
                centroids[i] = new_centroid;
            }
        }

        if (converged) break;
    }

    for (int i = 0; i + 1 < n_levels; ++i) {
        thresholds[i] = (centroids[i] + centroids[i + 1]) / 2.0f;
    }
}

void qkv_compute_lloyd_max_codebook_ex(
    float* centroids,
    float* thresholds,
    int bits,
    int dim,
    unsigned distribution
) {
    if (!centroids || !thresholds || bits <= 0 || bits > 8 || dim <= 0) {
        return;
    }
    const int n_levels = 1 << bits;

    if (bits == 2 && distribution == QKV_CODEBOOK_DISTRIBUTION_GAUSSIAN_APPROX) {
        const double scale = 1.0 / sqrt((double)dim);
        centroids[0] = (float)(-1.51 * scale);
        centroids[1] = (float)(-0.453 * scale);
        centroids[2] = (float)(0.453 * scale);
        centroids[3] = (float)(1.51 * scale);

        thresholds[0] = (float)((-1.51 - 0.453) / 2.0 * scale);
        thresholds[1] = 0.0f;
        thresholds[2] = (float)((0.453 + 1.51) / 2.0 * scale);
        return;
    }

    lloyd_max_codebook(centroids, thresholds, n_levels, dim, 100, distribution);
}

void qkv_compute_lloyd_max_codebook(
    float* centroids,
    float* thresholds,
    int bits,
    int dim
) {
    qkv_compute_lloyd_max_codebook_ex(
        centroids,
        thresholds,
        bits,
        dim,
        QKV_CODEBOOK_DISTRIBUTION_EXACT_BETA);
}

int qkv_find_nearest_centroid(
    float val,
    const float* centroids,
    const float* thresholds,
    int n_levels
) {
    if (!centroids || !thresholds || n_levels <= 0) {
        return 0;
    }
    if (n_levels == 1) {
        return 0;
    }
    int lo = 0;
    int hi = n_levels - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (val < thresholds[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
