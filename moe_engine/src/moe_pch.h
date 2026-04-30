#pragma once
// Precompiled header for fast compilation

// Standard library
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

// Platform-specific
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

// Project headers
#include "moe_pc_engine.h"
#include "../../system_memory.h"
#include "../native/fp4_decode.h"
#include "../native/scale4.h"
#include "../../engine_core/kv/kv_qkv.h"
#include "../../engine_core/core/mmap_loader.h"
#include "../../loader/path_join.h"
#include "../../loader/wide_path.h"
#include "storage_llm_tools.h"
