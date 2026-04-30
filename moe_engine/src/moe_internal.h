#pragma once

// Common header for all moe_pc_engine compilation units
// This replaces the monolithic moe_pc_engine.cpp approach

#include "moe_pc_engine.h"

#include "../../system_memory.h"
#include "../native/fp4_decode.h"
#include "../native/scale4.h"
#include "../../engine_core/kv/kv_qkv.h"
#include "../../engine_core/core/mmap_loader.h"
#include "../../loader/path_join.h"
#include "../../loader/wide_path.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <memory>
#include <queue>
#include <limits>
#include <sstream>
#include <shared_mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(__AVX2__)
#if defined(__FMA__) || defined(_MSC_VER)
#define moe_HAVE_FMA_INTRINSICS 1
#else
#define moe_HAVE_FMA_INTRINSICS 0
#endif

static inline __m256 moe_madd_ps(__m256 a, __m256 b, __m256 c) {
#if moe_HAVE_FMA_INTRINSICS
    return _mm256_fmadd_ps(a, b, c);
#else
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}
#endif

#include "storage_llm_tools.h"
