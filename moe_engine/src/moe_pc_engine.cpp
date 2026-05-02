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
#include <cctype>
#include <cstdarg>
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

#if defined(__AVX2__) || defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
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
#include "parts/io_ring_adapter.cpp.inc"
#include "parts/io_atomic_stats_state.cpp.inc"
#include "parts/engine_types.cpp.inc"
#include "parts/device_types.cpp.inc"
#include "parts/generation_paged_kv.cpp.inc"
#include "parts/generation_batch_scheduler.cpp.inc"
#include "parts/engine_state.cpp.inc"
#include "parts/csv_numbers.cpp.inc"
#include "parts/projection_parse.cpp.inc"
#include "parts/tensor_slots.cpp.inc"
#include "parts/tensor_paths.cpp.inc"
#include "parts/platform_utils.cpp.inc"
#include "parts/platform_dir.cpp.inc"
#include "parts/binary_reader.cpp.inc"
#include "parts/file_collector.cpp.inc"
#include "parts/gguf_types.cpp.inc"
#include "parts/json_parser.cpp.inc"
#include "parts/model_contract_types.cpp.inc"
#include "parts/gguf_tensor_reader.cpp.inc"
#include "parts/juju_parser.cpp.inc"
#include "parts/metadata_parser.cpp.inc"
#include "parts/model_file_readers.cpp.inc"
#include "parts/model_scan.cpp.inc"
#include "parts/model_helpers.cpp.inc"
#include "parts/model_engine_state.cpp.inc"
#include "parts/model_engine_contract.cpp.inc"
#include "parts/model_validation_api.cpp.inc"
#include "parts/device_cuda_driver.cpp.inc"
#include "parts/device_hip_driver.cpp.inc"
#include "parts/tensor_layout.cpp.inc"
#include "parts/residency_helpers.cpp.inc"
#include "parts/residency_usage.cpp.inc"
#include "parts/backend_compiled_flags.cpp.inc"
#include "parts/backend_kernel_flags.cpp.inc"
#include "parts/backend_choose.cpp.inc"
#include "parts/backend_common_cache.cpp.inc"
#include "parts/backend_caps_helpers.cpp.inc"
#include "parts/backend_caps.cpp.inc"
#include "parts/model_library_paths.cpp.inc"
#include "parts/model_library_source.cpp.inc"
#include "parts/model_library_load.cpp.inc"
#include "parts/io_path.cpp.inc"
#include "parts/juju_format.cpp.inc"
#include "parts/common_residency.cpp.inc"
#include "parts/prefetch_bytes.cpp.inc"
#include "parts/staging_alloc.cpp.inc"
#include "parts/worker_caps.cpp.inc"
#include "parts/juju_runtime_strategy.cpp.inc"
#include "parts/staging_slots.cpp.inc"
#include "parts/io_stats_helpers.cpp.inc"
#include "parts/residency_eviction.cpp.inc"
#include "parts/device_allocator.cpp.inc"
#include "parts/device_tensor_copy.cpp.inc"
#include "parts/device_raw_copy.cpp.inc"
#include "parts/prefetch_model_bytes.cpp.inc"
#include "parts/prefetch_disk.cpp.inc"
#include "parts/io_iocp_adapter.cpp.inc"
#include "parts/prefetch_pinned.cpp.inc"
#include "parts/prefetch_devices.cpp.inc"
#include "parts/queue_helpers.cpp.inc"
#include "parts/io_worker_loop.cpp.inc"
#include "parts/pinned_worker_loop.cpp.inc"
#include "parts/gpu_worker_loop.cpp.inc"
#include "parts/runtime_orchestrator_workers.cpp.inc"
#include "parts/io_worker_lifecycle.cpp.inc"
#include "parts/enqueue_tensor_prefetch.cpp.inc"
#include "parts/default_configs.cpp.inc"
#include "parts/profile_trace.cpp.inc"
#include "parts/parallel_rows_pool.cpp.inc"
#include "parts/parallel_rows.cpp.inc"
#include "parts/lifecycle_backend_api.cpp.inc"
#include "parts/optimization_plan.cpp.inc"
#include "parts/kv_residency_api.cpp.inc"
#include "parts/residency_stats.cpp.inc"
#include "parts/io_config_api.cpp.inc"
#include "parts/common_raw_prefetch.cpp.inc"
#include "parts/moe_activation_api.cpp.inc"
#include "parts/prefetch_plan_helpers.cpp.inc"
#include "parts/expert_prefetch_api.cpp.inc"
#include "parts/prefetch_plan_topology.cpp.inc"
#include "parts/prefetch_plan.cpp.inc"
#include "parts/prefetch_admission.cpp.inc"
#include "parts/prefetch_plan_execute.cpp.inc"
#include "parts/runtime_orchestrator_items.cpp.inc"
#include "parts/topology_load.cpp.inc"
#include "parts/topology_save.cpp.inc"
#include "parts/topology_record.cpp.inc"
#include "parts/runtime_orchestrator.cpp.inc"
#include "parts/codec_table_columns.cpp.inc"
#include "parts/codec_table.cpp.inc"
#include "parts/tensor_fp16.cpp.inc"
#include "parts/tensor_scale_lookup.cpp.inc"
#include "parts/tensor_query.cpp.inc"
#include "parts/gguf_iq_tables.cpp.inc"
#include "parts/tensor_dot_cpu_kernels.cpp.inc"
#include "parts/tensor_dot.cpp.inc"
#include "parts/raw_forward_cpu.cpp.inc"
#include "parts/generation_api.cpp.inc"
#include "parts/generation_api_batch.cpp.inc"
#include "parts/forward_status.cpp.inc"
#include "parts/expert_triplet_cpu.cpp.inc"
#include "parts/token_grouping.cpp.inc"
#include "parts/names_core.cpp.inc"
#include "parts/names_backend.cpp.inc"
