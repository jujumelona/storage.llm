static std::string model_lib_source_text(const moe_pc_engine* e) {
    const moe_storage_constants_t* c = moe_storage_constants();
    const moe_backend_t backend = e ? e->backend_caps.backend : moe_BACKEND_CPU;
    const int generated_expert_batch_kernel =
        (backend == moe_BACKEND_CUDA || backend == moe_BACKEND_HIP) ? 1 : 0;
    const uint32_t scratch_slots = std::max<uint32_t>(
        16u,
        e ? e->io_config.gpu_expert_scratch_slots : 16u);
    std::ostringstream s;
    s << "#include <stdint.h>\n";
    s << "#include <stddef.h>\n";
    s << "#include <string.h>\n";
    s << "#include <stdlib.h>\n";
    s << "#include <math.h>\n";
    s << "#include <atomic>\n";
    s << "#include <chrono>\n";
    s << "#include <new>\n";
    s << "#if defined(_WIN32)\n#include <windows.h>\n";
    s << "#else\n#include <dlfcn.h>\n#endif\n";
    s << "#if defined(_WIN32)\n#define moe_EXPORT __declspec(dllexport)\n";
    s << "#else\n#define moe_EXPORT __attribute__((visibility(\"default\")))\n#endif\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_RAW_FP32 = "
      << (uint32_t)moe_WEIGHT_ENCODING_RAW_FP32 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_RAW_FP16 = "
      << (uint32_t)moe_WEIGHT_ENCODING_RAW_FP16 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_RAW_BF16 = "
      << (uint32_t)moe_WEIGHT_ENCODING_RAW_BF16 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_MXFP4 = "
      << (uint32_t)moe_WEIGHT_ENCODING_MXFP4 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_GGUF_Q4_0 = "
      << (uint32_t)moe_WEIGHT_ENCODING_GGUF_Q4_0 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_GGUF_Q4_1 = "
      << (uint32_t)moe_WEIGHT_ENCODING_GGUF_Q4_1 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_GGUF_Q8_0 = "
      << (uint32_t)moe_WEIGHT_ENCODING_GGUF_Q8_0 << "u;\n";
    s << "static const uint32_t moe_MODEL_WEIGHT_GGUF_Q8_1 = "
      << (uint32_t)moe_WEIGHT_ENCODING_GGUF_Q8_1 << "u;\n";
    s << "static const uint32_t moe_MODEL_ALLOC_NATIVE_PTR = "
      << (uint32_t)moe_DEVICE_ALLOC_NATIVE_PTR << "u;\n";
    s << "static const int32_t moe_MODEL_BACKEND_CUDA = "
      << (int32_t)moe_BACKEND_CUDA << ";\n";
    s << "static const int32_t moe_MODEL_BACKEND_HIP = "
      << (int32_t)moe_BACKEND_HIP << ";\n";
    s << "static const uint32_t moe_MODEL_EXPERT_SCRATCH_SLOTS = "
      << scratch_slots << "u;\n";
    s << R"MOE_MODEL_LIB(
struct moe_model_device_tensor_batch_view {
    const void* rec;
    uint64_t key;
    uint64_t ptr;
    uint64_t bytes;
    uint32_t weight_format;
    uint32_t backend_kind;
    uint64_t backend_aux;
    uint32_t rows;
    uint32_t cols;
    uint64_t weight_row_bytes;
    uint64_t weight_bytes;
    uint64_t stream_bytes;
    uint32_t expert_gpu_layout_kind;
    uint64_t expert_gpu_layout_offset;
    uint64_t expert_gpu_layout_size;
    uint64_t expert_gpu_layout_row_bytes;
};

typedef int (*moe_gpu_malloc_fn)(void**, size_t);
typedef int (*moe_gpu_free_fn)(void*);
typedef int (*moe_gpu_memcpy_fn)(void*, const void*, size_t, int);
typedef int (*moe_blas_create_fn)(void**);
typedef int (*moe_blas_destroy_fn)(void*);
typedef int (*moe_blas_sgemm_fn)(
    void*, int, int, int, int, int,
    const float*, const float*, int,
    const float*, int, const float*, float*, int);
typedef int (*moe_blas_sgemm_batched_fn)(
    void*, int, int, int, int, int,
    const float*, const float* const*, int,
    const float* const*, int, const float*, float* const*, int, int);
typedef int (*moe_blas_sgemm_grouped_batched_fn)(
    void*,
    const int*, const int*,
    const int*, const int*, const int*,
    const float*,
    const float* const*, const int*,
    const float* const*, const int*,
    const float*,
    float* const*, const int*,
    int, const int*);
typedef int (*moe_cuda_cuInit_fn)(unsigned int);
typedef int (*moe_cuda_cuModuleLoadData_fn)(void**, const void*);
typedef int (*moe_cuda_cuModuleGetFunction_fn)(void**, void*, const char*);
typedef int (*moe_cuda_cuLaunchKernel_fn)(
    void*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, void*, void**, void**);
typedef int (*moe_nvrtcCreateProgram_fn)(
    void**, const char*, const char*, int, const char* const*, const char* const*);
typedef int (*moe_nvrtcCompileProgram_fn)(void*, int, const char* const*);
typedef int (*moe_nvrtcGetPTXSize_fn)(void*, size_t*);
typedef int (*moe_nvrtcGetPTX_fn)(void*, char*);
typedef int (*moe_nvrtcDestroyProgram_fn)(void**);
typedef int (*moe_hiprtcGetCodeSize_fn)(void*, size_t*);
typedef int (*moe_hiprtcGetCode_fn)(void*, char*);
typedef int (*moe_hipModuleLoadData_fn)(void**, const void*);
typedef int (*moe_hipModuleGetFunction_fn)(void**, void*, const char*);
typedef int (*moe_hipModuleLaunchKernel_fn)(
    void*, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, void*, void**, void**);

struct moe_gpu_expert_scratch {
    std::atomic<int> in_use;
    void* d_x;
    void* d_gate;
    void* d_up;
    void* d_mid;
    void* d_out;
    void* d_batch_input;
    void* d_batch_accum;
    void* d_token_indices;
    void* d_token_weights;
    void* h_x;
    size_t x_bytes;
    size_t gate_bytes;
    size_t up_bytes;
    size_t mid_bytes;
    size_t out_bytes;
    size_t batch_input_bytes;
    size_t batch_accum_bytes;
    size_t token_indices_bytes;
    size_t token_weights_bytes;
    size_t h_x_bytes;
};

struct moe_dyn_gpu_blas {
    int loaded;
    int ok;
    int is_hip;
    void* runtime_lib;
    void* blas_lib;
    moe_gpu_malloc_fn gpu_malloc;
    moe_gpu_free_fn gpu_free;
    moe_gpu_memcpy_fn gpu_memcpy;
    moe_blas_create_fn blas_create;
    moe_blas_destroy_fn blas_destroy;
    moe_blas_sgemm_fn blas_sgemm;
    moe_blas_sgemm_batched_fn blas_sgemm_batched;
    moe_blas_sgemm_grouped_batched_fn blas_sgemm_grouped_batched;
    void* blas_handle;
    void* driver_lib;
    void* rtc_lib;
    moe_cuda_cuInit_fn cu_init;
    moe_cuda_cuModuleLoadData_fn cu_module_load_data;
    moe_cuda_cuModuleGetFunction_fn cu_module_get_function;
    moe_cuda_cuLaunchKernel_fn cu_launch_kernel;
    moe_nvrtcCreateProgram_fn nvrtc_create_program;
    moe_nvrtcCompileProgram_fn nvrtc_compile_program;
    moe_nvrtcGetPTXSize_fn nvrtc_get_ptx_size;
    moe_nvrtcGetPTX_fn nvrtc_get_ptx;
    moe_nvrtcDestroyProgram_fn nvrtc_destroy_program;
    moe_hiprtcGetCodeSize_fn hiprtc_get_code_size;
    moe_hiprtcGetCode_fn hiprtc_get_code;
    moe_hipModuleLoadData_fn hip_module_load_data;
    moe_hipModuleGetFunction_fn hip_module_get_function;
    moe_hipModuleLaunchKernel_fn hip_module_launch_kernel;
    int activation_jit_attempted;
    int activation_jit_ready;
    void* activation_module;
    void* activation_function;
    void* activation_grouped_function;
    void* activation_gather_function;
    void* activation_accum_function;
    moe_gpu_expert_scratch* scratch_pool;
    uint32_t scratch_pool_count;
};

static std::atomic<uint64_t> moe_model_gpu_expert_scratch_wait_ns{0};
static std::atomic<uint64_t> moe_model_gpu_expert_scratch_backpressure{0};

struct moe_model_grouped_expert_task {
    int32_t layer;
    int32_t expert;
    const void* gate;
    const void* up;
    const void* down;
    const float* x;
    float* out;
    uint32_t batch;
    uint32_t hidden;
    uint32_t intermediate;
};

struct moe_model_grouped_expert_device_task {
    int32_t layer;
    int32_t expert;
    const void* gate;
    const void* up;
    const void* down;
    const void* d_x;
    void* d_out;
    uint32_t batch_stride;
    uint32_t out_stride;
    uint32_t batch;
    uint32_t hidden;
    uint32_t intermediate;
};

struct moe_model_indexed_expert_task {
    int32_t layer;
    int32_t expert;
    const void* gate;
    const void* up;
    const void* down;
    uint32_t token_offset;
    uint32_t batch;
    uint32_t hidden;
    uint32_t intermediate;
};

static void* moe_model_load_one(const char* const* names) {
    if (!names) return 0;
    for (uint32_t i = 0; names[i]; ++i) {
#if defined(_WIN32)
        HMODULE h = LoadLibraryA(names[i]);
        if (h) return (void*)h;
#else
        void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (h) return h;
#endif
    }
    return 0;
}

static void* moe_model_sym(void* lib, const char* name) {
    if (!lib || !name) return 0;
#if defined(_WIN32)
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

static int moe_model_init_scratch_pool(moe_dyn_gpu_blas* b) {
    if (!b || !b->ok) return 0;
    if (b->scratch_pool && b->scratch_pool_count > 0) return 1;
    const uint32_t count = moe_MODEL_EXPERT_SCRATCH_SLOTS < 16u ? 16u : moe_MODEL_EXPERT_SCRATCH_SLOTS;
    b->scratch_pool = new (std::nothrow) moe_gpu_expert_scratch[count];
    if (!b->scratch_pool) {
        return 0;
    }
    b->scratch_pool_count = count;
    return 1;
}

static void moe_model_try_load_cuda(moe_dyn_gpu_blas* b) {
    if (!b || b->loaded) return;
    b->loaded = 1;
    b->is_hip = 0;
#if defined(_WIN32)
    static const char* const runtime_names[] = {
        "cudart64_130.dll", "cudart64_12.dll", "cudart64_110.dll", 0};
    static const char* const blas_names[] = {
        "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll", 0};
    static const char* const driver_names[] = {
        "nvcuda.dll", 0};
    static const char* const rtc_names[] = {
        "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc64_110_0.dll", 0};
#else
    static const char* const runtime_names[] = {
        "libcudart.so.13", "libcudart.so.12", "libcudart.so", 0};
    static const char* const blas_names[] = {
        "libcublas.so.13", "libcublas.so.12", "libcublas.so", 0};
    static const char* const driver_names[] = {
        "libcuda.so.1", "libcuda.so", 0};
    static const char* const rtc_names[] = {
        "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so", 0};
#endif
    b->runtime_lib = moe_model_load_one(runtime_names);
    b->blas_lib = moe_model_load_one(blas_names);
    b->driver_lib = moe_model_load_one(driver_names);
    b->rtc_lib = moe_model_load_one(rtc_names);
    b->gpu_malloc = (moe_gpu_malloc_fn)moe_model_sym(b->runtime_lib, "cudaMalloc");
    b->gpu_free = (moe_gpu_free_fn)moe_model_sym(b->runtime_lib, "cudaFree");
    b->gpu_memcpy = (moe_gpu_memcpy_fn)moe_model_sym(b->runtime_lib, "cudaMemcpy");
    b->blas_create = (moe_blas_create_fn)moe_model_sym(b->blas_lib, "cublasCreate_v2");
    b->blas_destroy = (moe_blas_destroy_fn)moe_model_sym(b->blas_lib, "cublasDestroy_v2");
    b->blas_sgemm = (moe_blas_sgemm_fn)moe_model_sym(b->blas_lib, "cublasSgemm_v2");
    b->blas_sgemm_batched = (moe_blas_sgemm_batched_fn)moe_model_sym(b->blas_lib, "cublasSgemmBatched_v2");
    b->blas_sgemm_grouped_batched =
        (moe_blas_sgemm_grouped_batched_fn)moe_model_sym(b->blas_lib, "cublasSgemmGroupedBatched");
    b->cu_init = (moe_cuda_cuInit_fn)moe_model_sym(b->driver_lib, "cuInit");
    b->cu_module_load_data = (moe_cuda_cuModuleLoadData_fn)moe_model_sym(b->driver_lib, "cuModuleLoadData");
    b->cu_module_get_function = (moe_cuda_cuModuleGetFunction_fn)moe_model_sym(b->driver_lib, "cuModuleGetFunction");
    b->cu_launch_kernel = (moe_cuda_cuLaunchKernel_fn)moe_model_sym(b->driver_lib, "cuLaunchKernel");
    b->nvrtc_create_program = (moe_nvrtcCreateProgram_fn)moe_model_sym(b->rtc_lib, "nvrtcCreateProgram");
    b->nvrtc_compile_program = (moe_nvrtcCompileProgram_fn)moe_model_sym(b->rtc_lib, "nvrtcCompileProgram");
    b->nvrtc_get_ptx_size = (moe_nvrtcGetPTXSize_fn)moe_model_sym(b->rtc_lib, "nvrtcGetPTXSize");
    b->nvrtc_get_ptx = (moe_nvrtcGetPTX_fn)moe_model_sym(b->rtc_lib, "nvrtcGetPTX");
    b->nvrtc_destroy_program = (moe_nvrtcDestroyProgram_fn)moe_model_sym(b->rtc_lib, "nvrtcDestroyProgram");
    b->ok = b->runtime_lib && b->blas_lib &&
            b->gpu_malloc && b->gpu_free && b->gpu_memcpy &&
            b->blas_create && b->blas_destroy && b->blas_sgemm;
    if (b->ok && !moe_model_init_scratch_pool(b)) {
        b->ok = 0;
    }
}

static void moe_model_try_load_hip(moe_dyn_gpu_blas* b) {
    if (!b || b->loaded) return;
    b->loaded = 1;
    b->is_hip = 1;
#if defined(_WIN32)
    static const char* const runtime_names[] = {
        "amdhip64.dll", "hip64.dll", 0};
    static const char* const blas_names[] = {
        "hipblas.dll", "hipblas64.dll", 0};
    static const char* const rtc_names[] = {
        "hiprtc.dll", "hiprtc64.dll", 0};
#else
    static const char* const runtime_names[] = {
        "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so", 0};
    static const char* const blas_names[] = {
        "libhipblas.so.2", "libhipblas.so", 0};
    static const char* const rtc_names[] = {
        "libhiprtc.so.6", "libhiprtc.so.5", "libhiprtc.so", 0};
#endif
    b->runtime_lib = moe_model_load_one(runtime_names);
    b->blas_lib = moe_model_load_one(blas_names);
    b->rtc_lib = moe_model_load_one(rtc_names);
    b->gpu_malloc = (moe_gpu_malloc_fn)moe_model_sym(b->runtime_lib, "hipMalloc");
    b->gpu_free = (moe_gpu_free_fn)moe_model_sym(b->runtime_lib, "hipFree");
    b->gpu_memcpy = (moe_gpu_memcpy_fn)moe_model_sym(b->runtime_lib, "hipMemcpy");
    b->blas_create = (moe_blas_create_fn)moe_model_sym(b->blas_lib, "hipblasCreate");
    b->blas_destroy = (moe_blas_destroy_fn)moe_model_sym(b->blas_lib, "hipblasDestroy");
    b->blas_sgemm = (moe_blas_sgemm_fn)moe_model_sym(b->blas_lib, "hipblasSgemm");
    b->blas_sgemm_batched = (moe_blas_sgemm_batched_fn)moe_model_sym(b->blas_lib, "hipblasSgemmBatched");
    b->blas_sgemm_grouped_batched = 0;
    b->nvrtc_create_program = (moe_nvrtcCreateProgram_fn)moe_model_sym(b->rtc_lib, "hiprtcCreateProgram");
    b->nvrtc_compile_program = (moe_nvrtcCompileProgram_fn)moe_model_sym(b->rtc_lib, "hiprtcCompileProgram");
    b->hiprtc_get_code_size = (moe_hiprtcGetCodeSize_fn)moe_model_sym(b->rtc_lib, "hiprtcGetCodeSize");
    b->hiprtc_get_code = (moe_hiprtcGetCode_fn)moe_model_sym(b->rtc_lib, "hiprtcGetCode");
    b->nvrtc_destroy_program = (moe_nvrtcDestroyProgram_fn)moe_model_sym(b->rtc_lib, "hiprtcDestroyProgram");
    b->hip_module_load_data = (moe_hipModuleLoadData_fn)moe_model_sym(b->runtime_lib, "hipModuleLoadData");
    b->hip_module_get_function = (moe_hipModuleGetFunction_fn)moe_model_sym(b->runtime_lib, "hipModuleGetFunction");
    b->hip_module_launch_kernel = (moe_hipModuleLaunchKernel_fn)moe_model_sym(b->runtime_lib, "hipModuleLaunchKernel");
    b->ok = b->runtime_lib && b->blas_lib &&
            b->gpu_malloc && b->gpu_free && b->gpu_memcpy &&
            b->blas_create && b->blas_destroy && b->blas_sgemm;
    if (b->ok && !moe_model_init_scratch_pool(b)) {
        b->ok = 0;
    }
}

static moe_dyn_gpu_blas* moe_model_gpu_blas(int32_t backend) {
    static moe_dyn_gpu_blas cuda_blas = {};
    static moe_dyn_gpu_blas hip_blas = {};
    if (backend == moe_MODEL_BACKEND_CUDA) {
        moe_model_try_load_cuda(&cuda_blas);
        return cuda_blas.ok ? &cuda_blas : 0;
    }
    if (backend == moe_MODEL_BACKEND_HIP) {
        moe_model_try_load_hip(&hip_blas);
        return hip_blas.ok ? &hip_blas : 0;
    }
    return 0;
}

static void* moe_model_blas_handle(moe_dyn_gpu_blas* gpu) {
    if (!gpu) return 0;
    if (gpu->blas_handle) return gpu->blas_handle;
    void* handle = 0;
    if (gpu->blas_create && gpu->blas_create(&handle) == 0 && handle) {
        gpu->blas_handle = handle;
        return gpu->blas_handle;
    }
    return 0;
}

static int moe_model_grow_device_buffer(
    moe_dyn_gpu_blas* gpu,
    void** ptr,
    size_t* cap,
    size_t need
) {
    if (!gpu || !ptr || !cap || need == 0) return 0;
    if (*ptr && *cap >= need) return 1;
    void* next = 0;
    if (gpu->gpu_malloc(&next, need) != 0 || !next) return 0;
    if (*ptr) gpu->gpu_free(*ptr);
    *ptr = next;
    *cap = need;
    return 1;
}

static int moe_model_grow_host_buffer(void** ptr, size_t* cap, size_t need) {
    if (!ptr || !cap || need == 0) return 0;
    if (*ptr && *cap >= need) return 1;
    void* next = malloc(need);
    if (!next) return 0;
    free(*ptr);
    *ptr = next;
    *cap = need;
    return 1;
}

static moe_gpu_expert_scratch* moe_acquire_expert_scratch(
    moe_dyn_gpu_blas* gpu,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size
) {
    if (!gpu || batch_size == 0 || hidden_size == 0 || intermediate_size == 0) return 0;
    const auto start_ns = std::chrono::steady_clock::now();
    const size_t x_bytes = (size_t)batch_size * hidden_size * sizeof(float);
    const size_t mid_bytes = (size_t)batch_size * intermediate_size * sizeof(float);
    const size_t out_bytes = (size_t)batch_size * hidden_size * sizeof(float);
    if (!gpu->scratch_pool || gpu->scratch_pool_count == 0) {
        moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    for (uint32_t i = 0; i < gpu->scratch_pool_count; ++i) {
        moe_gpu_expert_scratch* s = &gpu->scratch_pool[i];
        int expected = 0;
        if (!s->in_use.compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }
        if (!moe_model_grow_device_buffer(gpu, &s->d_x, &s->x_bytes, x_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_gate, &s->gate_bytes, mid_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_up, &s->up_bytes, mid_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_mid, &s->mid_bytes, mid_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_out, &s->out_bytes, out_bytes) ||
            !moe_model_grow_host_buffer(&s->h_x, &s->h_x_bytes, x_bytes)) {
            s->in_use.store(0, std::memory_order_release);
            moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        return s;
    }
    const auto end_ns = std::chrono::steady_clock::now();
    moe_model_gpu_expert_scratch_wait_ns.fetch_add(
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - start_ns).count(),
        std::memory_order_relaxed);
    moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

static void moe_release_expert_scratch(moe_gpu_expert_scratch* s) {
    if (s) s->in_use.store(0, std::memory_order_release);
}

static moe_gpu_expert_scratch* moe_acquire_prefill_activation_scratch(
    moe_dyn_gpu_blas* gpu,
    size_t input_bytes,
    size_t accum_bytes,
    size_t index_bytes,
    size_t weight_bytes
) {
    if (!gpu || input_bytes == 0 || accum_bytes == 0 || index_bytes == 0 || weight_bytes == 0 ||
        !gpu->scratch_pool || gpu->scratch_pool_count == 0) {
        moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto start_ns = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < gpu->scratch_pool_count; ++i) {
        moe_gpu_expert_scratch* s = &gpu->scratch_pool[i];
        int expected = 0;
        if (!s->in_use.compare_exchange_strong(
                expected, 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            continue;
        }
        if (!moe_model_grow_device_buffer(gpu, &s->d_batch_input, &s->batch_input_bytes, input_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_batch_accum, &s->batch_accum_bytes, accum_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_token_indices, &s->token_indices_bytes, index_bytes) ||
            !moe_model_grow_device_buffer(gpu, &s->d_token_weights, &s->token_weights_bytes, weight_bytes)) {
            s->in_use.store(0, std::memory_order_release);
            moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        return s;
    }
    const auto end_ns = std::chrono::steady_clock::now();
    moe_model_gpu_expert_scratch_wait_ns.fetch_add(
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - start_ns).count(),
        std::memory_order_relaxed);
    moe_model_gpu_expert_scratch_backpressure.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

static float moe_model_silu(float x) {
    if (x > 40.0f) return x;
    if (x < -40.0f) return 0.0f;
    return x / (1.0f + expf(-x));
}

static float moe_model_fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int exp = (int)((h >> 10) & 0x1fu);
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static float moe_model_bf16_to_f32(uint16_t b) {
    uint32_t out = (uint32_t)b << 16;
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static int moe_model_quant_format_supported(uint32_t f) {
    return f == moe_MODEL_WEIGHT_RAW_FP32 ||
           f == moe_MODEL_WEIGHT_RAW_FP16 ||
           f == moe_MODEL_WEIGHT_RAW_BF16 ||
           f == moe_MODEL_WEIGHT_MXFP4 ||
           f == moe_MODEL_WEIGHT_GGUF_Q4_0 ||
           f == moe_MODEL_WEIGHT_GGUF_Q4_1 ||
           f == moe_MODEL_WEIGHT_GGUF_Q8_0 ||
           f == moe_MODEL_WEIGHT_GGUF_Q8_1;
}

static float moe_model_e8m0_to_f32(uint8_t v) {
    uint32_t bits = v ? ((uint32_t)v << 23u) : 0x00400000u;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float moe_model_mxfp4_value(uint8_t code) {
    static const float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return table[code & 0x0fu];
}

static int moe_model_dequant_row_to_fp32(
    const uint8_t* row,
    uint32_t format,
    uint32_t cols,
    float* out
) {
    if (!row || !out || cols == 0) return 0;
    if (format == moe_MODEL_WEIGHT_RAW_FP32) {
        memcpy(out, row, (size_t)cols * sizeof(float));
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_RAW_FP16) {
        const uint16_t* p = (const uint16_t*)row;
        for (uint32_t i = 0; i < cols; ++i) out[i] = moe_model_fp16_to_f32(p[i]);
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_RAW_BF16) {
        const uint16_t* p = (const uint16_t*)row;
        for (uint32_t i = 0; i < cols; ++i) out[i] = moe_model_bf16_to_f32(p[i]);
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_GGUF_Q4_0) {
        const uint32_t blocks = (cols + 31u) / 32u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint8_t* b = row + (size_t)block * 18u;
            const float d = moe_model_fp16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            const uint8_t* qs = b + 2;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = block * 32u + j;
                if (col >= cols) break;
                const uint8_t q = qs[j & 15u];
                const int v = (j < 16u) ? (int)(q & 0x0fu) : (int)(q >> 4);
                out[col] = ((float)v - 8.0f) * d;
            }
        }
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_GGUF_Q4_1) {
        const uint32_t blocks = (cols + 31u) / 32u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint8_t* b = row + (size_t)block * 20u;
            const float d = moe_model_fp16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            const float m = moe_model_fp16_to_f32((uint16_t)b[2] | ((uint16_t)b[3] << 8));
            const uint8_t* qs = b + 4;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = block * 32u + j;
                if (col >= cols) break;
                const uint8_t q = qs[j & 15u];
                const int v = (j < 16u) ? (int)(q & 0x0fu) : (int)(q >> 4);
                out[col] = (float)v * d + m;
            }
        }
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_GGUF_Q8_0) {
        const uint32_t blocks = (cols + 31u) / 32u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint8_t* b = row + (size_t)block * 34u;
            const float d = moe_model_fp16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            const int8_t* qs = (const int8_t*)(b + 2);
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = block * 32u + j;
                if (col >= cols) break;
                out[col] = (float)qs[j] * d;
            }
        }
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_GGUF_Q8_1) {
        const uint32_t blocks = (cols + 31u) / 32u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint8_t* b = row + (size_t)block * 36u;
            const float d = moe_model_fp16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            const int8_t* qs = (const int8_t*)(b + 4);
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = block * 32u + j;
                if (col >= cols) break;
                out[col] = (float)qs[j] * d;
            }
        }
        return 1;
    }
    if (format == moe_MODEL_WEIGHT_MXFP4) {
        const uint32_t blocks = (cols + 31u) / 32u;
        for (uint32_t block = 0; block < blocks; ++block) {
            const uint8_t* b = row + (size_t)block * 17u;
            const float scale = moe_model_e8m0_to_f32(b[0]);
            const uint8_t* qs = b + 1;
            for (uint32_t j = 0; j < 32u; ++j) {
                const uint32_t col = block * 32u + j;
                if (col >= cols) break;
                const uint8_t byte = qs[j & 15u];
                const uint8_t code = (j < 16u) ? (byte & 0x0fu) : (byte >> 4);
                out[col] = moe_model_mxfp4_value(code) * scale;
            }
        }
        return 1;
    }
    return 0;
}

struct moe_model_weight_cache_entry {
    uint64_t key;
    uint64_t ptr;
    uint64_t bytes;
    uint32_t format;
    uint32_t rows;
    uint32_t cols;
    void* fp32_device;
};

static std::atomic_flag moe_model_weight_cache_lock = ATOMIC_FLAG_INIT;
static moe_model_weight_cache_entry moe_model_weight_cache[128] = {};
static uint32_t moe_model_weight_cache_count = 0;

static void moe_model_lock_cache() {
    while (moe_model_weight_cache_lock.test_and_set(std::memory_order_acquire)) {}
}

static void moe_model_unlock_cache() {
    moe_model_weight_cache_lock.clear(std::memory_order_release);
}

static int moe_model_allow_cpu_weight_dequant() {
    const char* v = getenv("STORAGELLM_MODEL_LIB_ALLOW_CPU_WEIGHT_DEQUANT");
    return v && v[0] == '1';
}

static int moe_model_get_weight_fp32_device(
    moe_dyn_gpu_blas* gpu,
    const moe_model_device_tensor_batch_view* v,
    uint32_t expected_rows,
    uint32_t expected_cols,
    void** out_ptr,
    int* out_owned
) {
    if (out_ptr) *out_ptr = 0;
    if (out_owned) *out_owned = 0;
    if (!gpu || !v || !out_ptr || !out_owned ||
        v->ptr == 0 || v->backend_kind != moe_MODEL_ALLOC_NATIVE_PTR ||
        v->rows != expected_rows || v->cols != expected_cols ||
        v->weight_row_bytes == 0 || v->weight_bytes == 0 ||
        !moe_model_quant_format_supported(v->weight_format)) {
        return 0;
    }
    const uint64_t fp32_bytes = (uint64_t)expected_rows * (uint64_t)expected_cols * sizeof(float);
    if (fp32_bytes == 0) return 0;
    if (v->weight_format == moe_MODEL_WEIGHT_RAW_FP32 &&
        v->bytes >= fp32_bytes && v->weight_bytes >= fp32_bytes) {
        *out_ptr = (void*)(uintptr_t)v->ptr;
        return 1;
    }
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= fp32_bytes &&
        v->expert_gpu_layout_row_bytes >= expected_cols * sizeof(float) &&
        v->expert_gpu_layout_offset <= v->bytes &&
        fp32_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        *out_ptr = (void*)(uintptr_t)(v->ptr + v->expert_gpu_layout_offset);
        return 1;
    }
    if (!moe_model_allow_cpu_weight_dequant()) {
        return 0;
    }

    moe_model_lock_cache();
    for (uint32_t i = 0; i < moe_model_weight_cache_count; ++i) {
        const moe_model_weight_cache_entry* e = &moe_model_weight_cache[i];
        if (e->key == v->key && e->ptr == v->ptr && e->bytes == v->weight_bytes &&
            e->format == v->weight_format && e->rows == expected_rows && e->cols == expected_cols &&
            e->fp32_device) {
            *out_ptr = e->fp32_device;
            moe_model_unlock_cache();
            return 1;
        }
    }
    moe_model_unlock_cache();

    uint8_t* host_packed = (uint8_t*)malloc((size_t)v->weight_bytes);
    float* host_fp32 = (float*)malloc((size_t)fp32_bytes);
    void* device_fp32 = 0;
    if (!host_packed || !host_fp32 ||
        gpu->gpu_memcpy(host_packed, (const void*)(uintptr_t)v->ptr, (size_t)v->weight_bytes, 2) != 0) {
        free(host_fp32);
        free(host_packed);
        return 0;
    }
    for (uint32_t r = 0; r < expected_rows; ++r) {
        const uint64_t row_off = (uint64_t)r * v->weight_row_bytes;
        if (row_off >= v->weight_bytes ||
            !moe_model_dequant_row_to_fp32(
                host_packed + row_off,
                v->weight_format,
                expected_cols,
                host_fp32 + (size_t)r * expected_cols)) {
            free(host_fp32);
            free(host_packed);
            return 0;
        }
    }
    if (gpu->gpu_malloc(&device_fp32, (size_t)fp32_bytes) != 0 ||
        !device_fp32 ||
        gpu->gpu_memcpy(device_fp32, host_fp32, (size_t)fp32_bytes, 1) != 0) {
        if (device_fp32) gpu->gpu_free(device_fp32);
        free(host_fp32);
        free(host_packed);
        return 0;
    }
    free(host_fp32);
    free(host_packed);

    moe_model_lock_cache();
    if (moe_model_weight_cache_count < 128u) {
        moe_model_weight_cache_entry* e = &moe_model_weight_cache[moe_model_weight_cache_count++];
        e->key = v->key;
        e->ptr = v->ptr;
        e->bytes = v->weight_bytes;
        e->format = v->weight_format;
        e->rows = expected_rows;
        e->cols = expected_cols;
        e->fp32_device = device_fp32;
        *out_ptr = device_fp32;
        *out_owned = 0;
        moe_model_unlock_cache();
        return 1;
    }
    moe_model_unlock_cache();
    *out_ptr = device_fp32;
    *out_owned = 1;
    return 1;
}

static int moe_model_ensure_cuda_activation_kernel(moe_dyn_gpu_blas* gpu) {
    if (!gpu || gpu->is_hip) return 0;
    if (gpu->activation_jit_ready && gpu->activation_function) return 1;
    if (gpu->activation_jit_attempted) return 0;
    gpu->activation_jit_attempted = 1;
    if (!gpu->driver_lib || !gpu->rtc_lib || !gpu->cu_init ||
        !gpu->cu_module_load_data || !gpu->cu_module_get_function ||
        !gpu->cu_launch_kernel ||
        !gpu->nvrtc_create_program || !gpu->nvrtc_compile_program ||
        !gpu->nvrtc_get_ptx_size || !gpu->nvrtc_get_ptx ||
        !gpu->nvrtc_destroy_program) {
        return 0;
    }
    const char* src =
        "extern \"C\" __global__ void moe_silu_mul_kernel("
        "const float* gate,const float* up,float* mid,unsigned long long n){"
        "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "if(i<n){float x=gate[i];float s=x>40.0f?x:(x<-40.0f?0.0f:x/(1.0f+__expf(-x)));mid[i]=s*up[i];}}"
        "extern \"C\" __global__ void moe_silu_mul_grouped_kernel("
        "const float* const* gate,const float* const* up,float** mid,unsigned long long n,unsigned int task_count){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=n*(unsigned long long)task_count;"
        "if(idx<total){unsigned int t=(unsigned int)(idx/n);unsigned long long i=idx-(unsigned long long)t*n;"
        "float x=gate[t][i];float s=x>40.0f?x:(x<-40.0f?0.0f:x/(1.0f+__expf(-x)));mid[t][i]=s*up[t][i];}}"
        "extern \"C\" __global__ void moe_gather_rows_kernel("
        "const float* input,unsigned int input_stride,const unsigned int* token_indices,"
        "unsigned long long total_rows,unsigned int hidden,float* out){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=total_rows*(unsigned long long)hidden;"
        "if(idx<total){unsigned long long row=idx/hidden;unsigned int col=(unsigned int)(idx-row*(unsigned long long)hidden);"
        "unsigned int token=token_indices[row];out[row*(unsigned long long)hidden+col]=input[token*(unsigned long long)input_stride+col];}}"
        "extern \"C\" __global__ void moe_weighted_accum_rows_kernel("
        "const float* rows,const unsigned int* token_indices,const float* weights,"
        "unsigned long long total_rows,unsigned int hidden,unsigned int accum_stride,float* accum){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=total_rows*(unsigned long long)hidden;"
        "if(idx<total){unsigned long long row=idx/hidden;unsigned int col=(unsigned int)(idx-row*(unsigned long long)hidden);"
        "unsigned int token=token_indices[row];float v=rows[row*(unsigned long long)hidden+col]*weights[row];"
        "atomicAdd(accum+token*(unsigned long long)accum_stride+col,v);}}";
    void* prog = 0;
    if (gpu->cu_init(0) != 0 ||
        gpu->nvrtc_create_program(&prog, src, "moe_silu_mul.cu", 0, 0, 0) != 0 ||
        !prog) {
        return 0;
    }
    const char* opts[] = {"--std=c++11", "--use_fast_math"};
    int ok = 0;
    if (gpu->nvrtc_compile_program(prog, 2, opts) == 0) {
        size_t ptx_size = 0;
        if (gpu->nvrtc_get_ptx_size(prog, &ptx_size) == 0 && ptx_size > 0) {
            char* ptx = (char*)malloc(ptx_size);
            if (ptx && gpu->nvrtc_get_ptx(prog, ptx) == 0 &&
                gpu->cu_module_load_data(&gpu->activation_module, ptx) == 0 &&
                gpu->activation_module &&
                gpu->cu_module_get_function(
                    &gpu->activation_function,
                    gpu->activation_module,
                    "moe_silu_mul_kernel") == 0 &&
                gpu->activation_function &&
                gpu->cu_module_get_function(
                    &gpu->activation_grouped_function,
                    gpu->activation_module,
                    "moe_silu_mul_grouped_kernel") == 0 &&
                gpu->activation_grouped_function &&
                gpu->cu_module_get_function(
                    &gpu->activation_gather_function,
                    gpu->activation_module,
                    "moe_gather_rows_kernel") == 0 &&
                gpu->activation_gather_function &&
                gpu->cu_module_get_function(
                    &gpu->activation_accum_function,
                    gpu->activation_module,
                    "moe_weighted_accum_rows_kernel") == 0 &&
                gpu->activation_accum_function) {
                gpu->activation_jit_ready = 1;
                ok = 1;
            }
            free(ptx);
        }
    }
    gpu->nvrtc_destroy_program(&prog);
    return ok;
}

static int moe_model_ensure_hip_activation_kernel(moe_dyn_gpu_blas* gpu) {
    if (!gpu || !gpu->is_hip) return 0;
    if (gpu->activation_jit_ready && gpu->activation_function) return 1;
    if (gpu->activation_jit_attempted) return 0;
    gpu->activation_jit_attempted = 1;
    if (!gpu->rtc_lib || !gpu->runtime_lib ||
        !gpu->nvrtc_create_program || !gpu->nvrtc_compile_program ||
        !gpu->hiprtc_get_code_size || !gpu->hiprtc_get_code ||
        !gpu->nvrtc_destroy_program ||
        !gpu->hip_module_load_data || !gpu->hip_module_get_function ||
        !gpu->hip_module_launch_kernel) {
        return 0;
    }
    const char* src =
        "extern \"C\" __global__ void moe_silu_mul_kernel("
        "const float* gate,const float* up,float* mid,unsigned long long n){"
        "unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "if(i<n){float x=gate[i];float s=x>40.0f?x:(x<-40.0f?0.0f:x/(1.0f+expf(-x)));mid[i]=s*up[i];}}"
        "extern \"C\" __global__ void moe_silu_mul_grouped_kernel("
        "const float* const* gate,const float* const* up,float** mid,unsigned long long n,unsigned int task_count){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=n*(unsigned long long)task_count;"
        "if(idx<total){unsigned int t=(unsigned int)(idx/n);unsigned long long i=idx-(unsigned long long)t*n;"
        "float x=gate[t][i];float s=x>40.0f?x:(x<-40.0f?0.0f:x/(1.0f+expf(-x)));mid[t][i]=s*up[t][i];}}"
        "extern \"C\" __global__ void moe_gather_rows_kernel("
        "const float* input,unsigned int input_stride,const unsigned int* token_indices,"
        "unsigned long long total_rows,unsigned int hidden,float* out){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=total_rows*(unsigned long long)hidden;"
        "if(idx<total){unsigned long long row=idx/hidden;unsigned int col=(unsigned int)(idx-row*(unsigned long long)hidden);"
        "unsigned int token=token_indices[row];out[row*(unsigned long long)hidden+col]=input[token*(unsigned long long)input_stride+col];}}"
        "extern \"C\" __global__ void moe_weighted_accum_rows_kernel("
        "const float* rows,const unsigned int* token_indices,const float* weights,"
        "unsigned long long total_rows,unsigned int hidden,unsigned int accum_stride,float* accum){"
        "unsigned long long idx=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;"
        "unsigned long long total=total_rows*(unsigned long long)hidden;"
        "if(idx<total){unsigned long long row=idx/hidden;unsigned int col=(unsigned int)(idx-row*(unsigned long long)hidden);"
        "unsigned int token=token_indices[row];float v=rows[row*(unsigned long long)hidden+col]*weights[row];"
        "atomicAdd(accum+token*(unsigned long long)accum_stride+col,v);}}";
    void* prog = 0;
    if (gpu->nvrtc_create_program(&prog, src, "moe_silu_mul.hip", 0, 0, 0) != 0 || !prog) {
        return 0;
    }
    const char* opts[] = {"--std=c++11"};
    int ok = 0;
    if (gpu->nvrtc_compile_program(prog, 1, opts) == 0) {
        size_t code_size = 0;
        if (gpu->hiprtc_get_code_size(prog, &code_size) == 0 && code_size > 0) {
            char* code = (char*)malloc(code_size);
            if (code && gpu->hiprtc_get_code(prog, code) == 0 &&
                gpu->hip_module_load_data(&gpu->activation_module, code) == 0 &&
                gpu->activation_module &&
                gpu->hip_module_get_function(
                    &gpu->activation_function,
                    gpu->activation_module,
                    "moe_silu_mul_kernel") == 0 &&
                gpu->activation_function &&
                gpu->hip_module_get_function(
                    &gpu->activation_grouped_function,
                    gpu->activation_module,
                    "moe_silu_mul_grouped_kernel") == 0 &&
                gpu->activation_grouped_function &&
                gpu->hip_module_get_function(
                    &gpu->activation_gather_function,
                    gpu->activation_module,
                    "moe_gather_rows_kernel") == 0 &&
                gpu->activation_gather_function &&
                gpu->hip_module_get_function(
                    &gpu->activation_accum_function,
                    gpu->activation_module,
                    "moe_weighted_accum_rows_kernel") == 0 &&
                gpu->activation_accum_function) {
                gpu->activation_jit_ready = 1;
                ok = 1;
            }
            free(code);
        }
    }
    gpu->nvrtc_destroy_program(&prog);
    return ok;
}

static int moe_model_device_silu_mul(
    moe_dyn_gpu_blas* gpu,
    const void* d_gate,
    const void* d_up,
    void* d_mid,
    size_t count
) {
    if (!gpu || !d_gate || !d_up || !d_mid || count == 0) {
        return 0;
    }
    const unsigned int block = 256u;
    const unsigned int grid = (unsigned int)((count + block - 1u) / block);
    unsigned long long n = (unsigned long long)count;
    void* args[] = {
        (void*)&d_gate,
        (void*)&d_up,
        (void*)&d_mid,
        (void*)&n
    };
    if (gpu->is_hip) {
        return moe_model_ensure_hip_activation_kernel(gpu) &&
               gpu->hip_module_launch_kernel(
                   gpu->activation_function,
                   grid, 1u, 1u,
                   block, 1u, 1u,
                   0u, 0, args, 0) == 0;
    }
    return moe_model_ensure_cuda_activation_kernel(gpu) &&
           gpu->cu_launch_kernel(
               gpu->activation_function,
               grid, 1u, 1u,
               block, 1u, 1u,
               0u, 0, args, 0) == 0;
}

static int moe_model_device_silu_mul_grouped(
    moe_dyn_gpu_blas* gpu,
    const void* d_gate_ptrs,
    const void* d_up_ptrs,
    void* d_mid_ptrs,
    size_t count,
    uint32_t task_count
) {
    if (!gpu || !d_gate_ptrs || !d_up_ptrs || !d_mid_ptrs || count == 0 || task_count == 0) {
        return 0;
    }
    if (count > UINT64_MAX / (uint64_t)task_count) {
        return 0;
    }
    const uint64_t total = (uint64_t)count * (uint64_t)task_count;
    const unsigned int block = 256u;
    const unsigned int grid = (unsigned int)((total + block - 1u) / block);
    unsigned long long n = (unsigned long long)count;
    unsigned int tc = task_count;
    void* args[] = {
        (void*)&d_gate_ptrs,
        (void*)&d_up_ptrs,
        (void*)&d_mid_ptrs,
        (void*)&n,
        (void*)&tc
    };
    if (gpu->is_hip) {
        return moe_model_ensure_hip_activation_kernel(gpu) &&
               gpu->activation_grouped_function &&
               gpu->hip_module_launch_kernel(
                   gpu->activation_grouped_function,
                   grid, 1u, 1u,
                   block, 1u, 1u,
                   0u, 0, args, 0) == 0;
    }
    return moe_model_ensure_cuda_activation_kernel(gpu) &&
           gpu->activation_grouped_function &&
           gpu->cu_launch_kernel(
               gpu->activation_grouped_function,
               grid, 1u, 1u,
               block, 1u, 1u,
               0u, 0, args, 0) == 0;
}

static int moe_model_device_gather_rows(
    moe_dyn_gpu_blas* gpu,
    const void* d_input,
    uint32_t input_stride,
    const void* d_token_indices,
    uint32_t total_rows,
    uint32_t hidden,
    void* d_out
) {
    if (!gpu || !d_input || !d_token_indices || !d_out ||
        input_stride < hidden || total_rows == 0 || hidden == 0) {
        return 0;
    }
    if (total_rows > UINT64_MAX / (uint64_t)hidden) {
        return 0;
    }
    const uint64_t total = (uint64_t)total_rows * (uint64_t)hidden;
    const unsigned int block = 256u;
    const unsigned int grid = (unsigned int)((total + block - 1u) / block);
    unsigned int stride = input_stride;
    unsigned long long rows = (unsigned long long)total_rows;
    unsigned int h = hidden;
    void* args[] = {
        (void*)&d_input,
        (void*)&stride,
        (void*)&d_token_indices,
        (void*)&rows,
        (void*)&h,
        (void*)&d_out
    };
    if (gpu->is_hip) {
        return moe_model_ensure_hip_activation_kernel(gpu) &&
               gpu->activation_gather_function &&
               gpu->hip_module_launch_kernel(
                   gpu->activation_gather_function,
                   grid, 1u, 1u,
                   block, 1u, 1u,
                   0u, 0, args, 0) == 0;
    }
    return moe_model_ensure_cuda_activation_kernel(gpu) &&
           gpu->activation_gather_function &&
           gpu->cu_launch_kernel(
               gpu->activation_gather_function,
               grid, 1u, 1u,
               block, 1u, 1u,
               0u, 0, args, 0) == 0;
}

static int moe_model_device_weighted_accum_rows(
    moe_dyn_gpu_blas* gpu,
    const void* d_rows,
    const void* d_token_indices,
    const void* d_weights,
    uint32_t total_rows,
    uint32_t hidden,
    uint32_t accum_stride,
    void* d_accum
) {
    if (!gpu || !d_rows || !d_token_indices || !d_weights || !d_accum ||
        accum_stride < hidden || total_rows == 0 || hidden == 0) {
        return 0;
    }
    if (total_rows > UINT64_MAX / (uint64_t)hidden) {
        return 0;
    }
    const uint64_t total = (uint64_t)total_rows * (uint64_t)hidden;
    const unsigned int block = 256u;
    const unsigned int grid = (unsigned int)((total + block - 1u) / block);
    unsigned long long rows = (unsigned long long)total_rows;
    unsigned int h = hidden;
    unsigned int stride = accum_stride;
    void* args[] = {
        (void*)&d_rows,
        (void*)&d_token_indices,
        (void*)&d_weights,
        (void*)&rows,
        (void*)&h,
        (void*)&stride,
        (void*)&d_accum
    };
    if (gpu->is_hip) {
        return moe_model_ensure_hip_activation_kernel(gpu) &&
               gpu->activation_accum_function &&
               gpu->hip_module_launch_kernel(
                   gpu->activation_accum_function,
                   grid, 1u, 1u,
                   block, 1u, 1u,
                   0u, 0, args, 0) == 0;
    }
    return moe_model_ensure_cuda_activation_kernel(gpu) &&
           gpu->activation_accum_function &&
           gpu->cu_launch_kernel(
               gpu->activation_accum_function,
               grid, 1u, 1u,
               block, 1u, 1u,
               0u, 0, args, 0) == 0;
}

static int moe_model_run_expert_triplet_batch_fp32_blas(
    int32_t backend,
    const moe_model_device_tensor_batch_view* gate,
    const moe_model_device_tensor_batch_view* up,
    const moe_model_device_tensor_batch_view* down,
    const float* const* batch_hidden,
    const float* batch_hidden_contiguous,
    const void* d_batch_hidden,
    void* d_out_device,
    uint32_t out_stride,
    uint32_t batch_stride,
    uint32_t batch_size,
    uint32_t hidden_size,
    uint32_t intermediate_size,
    float* out_pool,
    uint32_t out_count,
    int activation_mode
) {
    (void)activation_mode;
    if (!gate || !up || !down || (!out_pool && !d_out_device) ||
        batch_size == 0 || hidden_size == 0 || intermediate_size == 0 ||
        out_count != hidden_size || batch_stride < hidden_size ||
        (!batch_hidden && !batch_hidden_contiguous && !d_batch_hidden) ||
        (d_batch_hidden && batch_stride != hidden_size) ||
        (d_out_device && out_stride != hidden_size) ||
        batch_size > 65535u || hidden_size > 2147483647u ||
        intermediate_size > 2147483647u) {
        return 0;
    }
    moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
    if (!gpu) return 0;

    void* gate_w_device = 0;
    void* up_w_device = 0;
    void* down_w_device = 0;
    int gate_w_owned = 0;
    int up_w_owned = 0;
    int down_w_owned = 0;
    if (!moe_model_get_weight_fp32_device(
            gpu, gate, intermediate_size, hidden_size, &gate_w_device, &gate_w_owned) ||
        !moe_model_get_weight_fp32_device(
            gpu, up, intermediate_size, hidden_size, &up_w_device, &up_w_owned) ||
        !moe_model_get_weight_fp32_device(
            gpu, down, hidden_size, intermediate_size, &down_w_device, &down_w_owned)) {
        if (gate_w_owned && gate_w_device) gpu->gpu_free(gate_w_device);
        if (up_w_owned && up_w_device) gpu->gpu_free(up_w_device);
        if (down_w_owned && down_w_device) gpu->gpu_free(down_w_device);
        return 0;
    }

    const size_t x_count = (size_t)batch_size * (size_t)hidden_size;
    const size_t mid_count = (size_t)batch_size * (size_t)intermediate_size;
    const size_t out_count_total = (size_t)batch_size * (size_t)out_count;
    moe_gpu_expert_scratch* scratch =
        moe_acquire_expert_scratch(gpu, batch_size, hidden_size, intermediate_size);
    if (!scratch) {
        if (gate_w_owned && gate_w_device) gpu->gpu_free(gate_w_device);
        if (up_w_owned && up_w_device) gpu->gpu_free(up_w_device);
        if (down_w_owned && down_w_device) gpu->gpu_free(down_w_device);
        return 0;
    }

    void* handle = 0;
    int ok = 0;
    const int memcpy_h2d = 1;
    const int memcpy_d2h = 2;
    const int op_n = gpu->is_hip ? 111 : 0;
    const int op_t = gpu->is_hip ? 112 : 1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int bsz = (int)batch_size;
    const int hsz = (int)hidden_size;
    const int isz = (int)intermediate_size;
    const void* x_device = d_batch_hidden;
    void* out_device = d_out_device ? d_out_device : scratch->d_out;

    if (x_device) {
        // Device input is already ready.
    } else if (batch_hidden_contiguous && batch_stride == hidden_size) {
        if (gpu->gpu_memcpy(scratch->d_x, batch_hidden_contiguous, x_count * sizeof(float), memcpy_h2d) != 0) {
            goto cleanup;
        }
        x_device = scratch->d_x;
    } else {
        float* host_x = (float*)scratch->h_x;
        if (!host_x) goto cleanup;
        if (batch_hidden_contiguous) {
            for (uint32_t b = 0; b < batch_size; ++b) {
                memcpy(host_x + (size_t)b * hidden_size,
                       batch_hidden_contiguous + (size_t)b * batch_stride,
                       (size_t)hidden_size * sizeof(float));
            }
        } else {
            for (uint32_t b = 0; b < batch_size; ++b) {
                if (!batch_hidden[b]) goto cleanup;
                memcpy(host_x + (size_t)b * hidden_size,
                       batch_hidden[b],
                       (size_t)hidden_size * sizeof(float));
            }
        }
        if (gpu->gpu_memcpy(scratch->d_x, host_x, x_count * sizeof(float), memcpy_h2d) != 0) {
            goto cleanup;
        }
        x_device = scratch->d_x;
    }
    handle = moe_model_blas_handle(gpu);
    if (!handle) {
        goto cleanup;
    }

    if (gpu->blas_sgemm(handle, op_t, op_n, isz, bsz, hsz,
                        &alpha, (const float*)gate_w_device, hsz,
                        (const float*)x_device, hsz,
                        &beta, (float*)scratch->d_gate, isz) != 0) {
        goto cleanup;
    }
    if (gpu->blas_sgemm(handle, op_t, op_n, isz, bsz, hsz,
                        &alpha, (const float*)up_w_device, hsz,
                        (const float*)x_device, hsz,
                        &beta, (float*)scratch->d_up, isz) != 0) {
        goto cleanup;
    }
    if (!moe_model_device_silu_mul(gpu, scratch->d_gate, scratch->d_up, scratch->d_mid, mid_count)) {
        goto cleanup;
    }
    if (gpu->blas_sgemm(handle, op_t, op_n, hsz, bsz, isz,
                        &alpha, (const float*)down_w_device, isz,
                        (const float*)scratch->d_mid, isz,
                        &beta, (float*)out_device, hsz) != 0) {
        goto cleanup;
    }
    if (!d_out_device &&
        gpu->gpu_memcpy(out_pool, scratch->d_out, out_count_total * sizeof(float), memcpy_d2h) != 0) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    if (gate_w_owned && gate_w_device) gpu->gpu_free(gate_w_device);
    if (up_w_owned && up_w_device) gpu->gpu_free(up_w_device);
    if (down_w_owned && down_w_device) gpu->gpu_free(down_w_device);
    moe_release_expert_scratch(scratch);
    return ok;
}

struct moe_model_grouped_expert_runtime_task {
    const void* gate;
    const void* up;
    const void* down;
    const void* x;
    void* out;
    uint32_t batch_stride;
    uint32_t out_stride;
    uint32_t batch;
    uint32_t hidden;
    uint32_t intermediate;
    uint8_t device_io;
};

static int moe_model_run_grouped_expert_triplets_runtime_fp32_blas(
    int32_t backend,
    const moe_model_grouped_expert_runtime_task* tasks,
    uint32_t task_count
) {
    if (!tasks || task_count < 2u || task_count > 4096u) {
        return 0;
    }
    const uint32_t batch = tasks[0].batch;
    const uint32_t hidden = tasks[0].hidden;
    const uint32_t intermediate = tasks[0].intermediate;
    if (batch == 0 || hidden == 0 || intermediate == 0 ||
        hidden > 2147483647u || intermediate > 2147483647u || batch > 65535u) {
        return 0;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        if (!tasks[i].gate || !tasks[i].up || !tasks[i].down || !tasks[i].x || !tasks[i].out ||
            tasks[i].batch != batch ||
            tasks[i].hidden != hidden ||
            tasks[i].intermediate != intermediate ||
            tasks[i].batch_stride != hidden ||
            tasks[i].out_stride != hidden) {
            return 0;
        }
    }
    if (batch > UINT32_MAX / task_count) {
        return 0;
    }
    moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
    if (!gpu || !gpu->blas_sgemm_batched) {
        return 0;
    }
    void* handle = moe_model_blas_handle(gpu);
    if (!handle) {
        return 0;
    }

    const uint32_t n = task_count;
    const size_t ptr_count = (size_t)n * 8u;
    void** host_ptrs = (void**)malloc(ptr_count * sizeof(void*));
    void** owned_weights = (void**)calloc((size_t)n * 3u, sizeof(void*));
    int* owned_flags = (int*)calloc((size_t)n * 3u, sizeof(int));
    moe_gpu_expert_scratch* workspace =
        moe_acquire_expert_scratch(gpu, batch * n, hidden, intermediate);
    void* d_ptrs = 0;
    if (!host_ptrs || !owned_weights || !owned_flags || !workspace) {
        if (workspace) {
            moe_release_expert_scratch(workspace);
        }
        free(owned_flags);
        free(owned_weights);
        free(host_ptrs);
        return 0;
    }

    int ok = 0;
    const int memcpy_h2d = 1;
    const int memcpy_d2h = 2;
    const int op_n = gpu->is_hip ? 111 : 0;
    const int op_t = gpu->is_hip ? 112 : 1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int bsz = (int)batch;
    const int hsz = (int)hidden;
    const int isz = (int)intermediate;
    const size_t x_bytes = (size_t)batch * hidden * sizeof(float);
    const size_t mid_count = (size_t)batch * intermediate;
    const size_t mid_bytes = mid_count * sizeof(float);
    const size_t out_bytes = (size_t)batch * hidden * sizeof(float);

    for (uint32_t i = 0; i < n; ++i) {
        void* gate_w = 0;
        void* up_w = 0;
        void* down_w = 0;
        int gate_owned = 0;
        int up_owned = 0;
        int down_owned = 0;
        if (!moe_model_get_weight_fp32_device(
                gpu,
                (const moe_model_device_tensor_batch_view*)tasks[i].gate,
                intermediate, hidden, &gate_w, &gate_owned) ||
            !moe_model_get_weight_fp32_device(
                gpu,
                (const moe_model_device_tensor_batch_view*)tasks[i].up,
                intermediate, hidden, &up_w, &up_owned) ||
            !moe_model_get_weight_fp32_device(
                gpu,
                (const moe_model_device_tensor_batch_view*)tasks[i].down,
                hidden, intermediate, &down_w, &down_owned)) {
            if (gate_owned && gate_w) gpu->gpu_free(gate_w);
            if (up_owned && up_w) gpu->gpu_free(up_w);
            if (down_owned && down_w) gpu->gpu_free(down_w);
            goto cleanup;
        }
        owned_weights[(size_t)i * 3u + 0u] = gate_w;
        owned_weights[(size_t)i * 3u + 1u] = up_w;
        owned_weights[(size_t)i * 3u + 2u] = down_w;
        owned_flags[(size_t)i * 3u + 0u] = gate_owned;
        owned_flags[(size_t)i * 3u + 1u] = up_owned;
        owned_flags[(size_t)i * 3u + 2u] = down_owned;

        void* x_slot = (void*)((char*)workspace->d_x + (size_t)i * x_bytes);
        void* gate_slot = (void*)((char*)workspace->d_gate + (size_t)i * mid_bytes);
        void* up_slot = (void*)((char*)workspace->d_up + (size_t)i * mid_bytes);
        void* mid_slot = (void*)((char*)workspace->d_mid + (size_t)i * mid_bytes);
        void* out_slot = (void*)((char*)workspace->d_out + (size_t)i * out_bytes);
        const void* x_device = tasks[i].x;
        void* out_device = tasks[i].out;
        if (!tasks[i].device_io) {
            if (gpu->gpu_memcpy(x_slot, tasks[i].x, x_bytes, memcpy_h2d) != 0) {
                goto cleanup;
            }
            x_device = x_slot;
            out_device = out_slot;
        }
        host_ptrs[(size_t)0u * n + i] = gate_w;
        host_ptrs[(size_t)1u * n + i] = up_w;
        host_ptrs[(size_t)2u * n + i] = down_w;
        host_ptrs[(size_t)3u * n + i] = (void*)x_device;
        host_ptrs[(size_t)4u * n + i] = gate_slot;
        host_ptrs[(size_t)5u * n + i] = up_slot;
        host_ptrs[(size_t)6u * n + i] = mid_slot;
        host_ptrs[(size_t)7u * n + i] = out_device;
    }

    if (gpu->gpu_malloc(&d_ptrs, ptr_count * sizeof(void*)) != 0 || !d_ptrs ||
        gpu->gpu_memcpy(d_ptrs, host_ptrs, ptr_count * sizeof(void*), memcpy_h2d) != 0) {
        goto cleanup;
    }
    {
        char* base = (char*)d_ptrs;
        const size_t stride = (size_t)n * sizeof(void*);
        const float* const* d_gate_w = (const float* const*)(base + stride * 0u);
        const float* const* d_up_w = (const float* const*)(base + stride * 1u);
        const float* const* d_down_w = (const float* const*)(base + stride * 2u);
        const float* const* d_x = (const float* const*)(base + stride * 3u);
        float* const* d_gate = (float* const*)(base + stride * 4u);
        float* const* d_up = (float* const*)(base + stride * 5u);
        float* const* d_mid = (float* const*)(base + stride * 6u);
        float* const* d_out = (float* const*)(base + stride * 7u);
        int grouped_gemm_ok = 0;
        if (!gpu->is_hip && gpu->blas_sgemm_grouped_batched && n >= 2u) {
            const int transa_gate[1] = {op_t};
            const int transb_gate[1] = {op_n};
            const int m_gate[1] = {isz};
            const int n_gate[1] = {bsz};
            const int k_gate[1] = {hsz};
            const int lda_gate[1] = {hsz};
            const int ldb_gate[1] = {hsz};
            const int ldc_gate[1] = {isz};
            const int group_size[1] = {(int)n};
            const float alpha_group[1] = {alpha};
            const float beta_group[1] = {beta};
            if (gpu->blas_sgemm_grouped_batched(
                    handle,
                    transa_gate, transb_gate, m_gate, n_gate, k_gate,
                    alpha_group, d_gate_w, lda_gate, d_x, ldb_gate,
                    beta_group, d_gate, ldc_gate, 1, group_size) == 0 &&
                gpu->blas_sgemm_grouped_batched(
                    handle,
                    transa_gate, transb_gate, m_gate, n_gate, k_gate,
                    alpha_group, d_up_w, lda_gate, d_x, ldb_gate,
                    beta_group, d_up, ldc_gate, 1, group_size) == 0 &&
                moe_model_device_silu_mul_grouped(gpu, d_gate, d_up, (void*)d_mid, mid_count, n)) {
                const int m_down[1] = {hsz};
                const int n_down[1] = {bsz};
                const int k_down[1] = {isz};
                const int lda_down[1] = {isz};
                const int ldb_down[1] = {isz};
                const int ldc_down[1] = {hsz};
                if (gpu->blas_sgemm_grouped_batched(
                        handle,
                        transa_gate, transb_gate, m_down, n_down, k_down,
                        alpha_group, d_down_w, lda_down,
                        (const float* const*)d_mid, ldb_down,
                        beta_group, d_out, ldc_down, 1, group_size) == 0) {
                    grouped_gemm_ok = 1;
                }
            }
        }
        if (!grouped_gemm_ok) {
            if (gpu->blas_sgemm_batched(handle, op_t, op_n, isz, bsz, hsz,
                                        &alpha, d_gate_w, hsz,
                                        d_x, hsz,
                                        &beta, d_gate, isz, (int)n) != 0) {
                goto cleanup;
            }
            if (gpu->blas_sgemm_batched(handle, op_t, op_n, isz, bsz, hsz,
                                        &alpha, d_up_w, hsz,
                                        d_x, hsz,
                                        &beta, d_up, isz, (int)n) != 0) {
                goto cleanup;
            }
            if (!moe_model_device_silu_mul_grouped(gpu, d_gate, d_up, (void*)d_mid, mid_count, n)) {
                goto cleanup;
            }
            if (gpu->blas_sgemm_batched(handle, op_t, op_n, hsz, bsz, isz,
                                        &alpha, d_down_w, isz,
                                        (const float* const*)d_mid, isz,
                                        &beta, d_out, hsz, (int)n) != 0) {
                goto cleanup;
            }
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        if (!tasks[i].device_io &&
            gpu->gpu_memcpy(
                tasks[i].out,
                (const void*)((const char*)workspace->d_out + (size_t)i * out_bytes),
                out_bytes,
                memcpy_d2h) != 0) {
            goto cleanup;
        }
    }
    ok = 1;

cleanup:
    if (d_ptrs) gpu->gpu_free(d_ptrs);
    if (workspace) {
        moe_release_expert_scratch(workspace);
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < 3u; ++j) {
            const size_t idx = (size_t)i * 3u + j;
            if (owned_flags[idx] && owned_weights[idx]) {
                gpu->gpu_free(owned_weights[idx]);
            }
        }
    }
    free(owned_flags);
    free(owned_weights);
    free(host_ptrs);
    return ok;
}

static int moe_model_run_grouped_expert_triplets_fp32_blas(
    int32_t backend,
    const moe_model_grouped_expert_task* tasks,
    uint32_t task_count
) {
    if (!tasks || task_count < 2u) {
        return 0;
    }
    moe_model_grouped_expert_runtime_task* rt =
        (moe_model_grouped_expert_runtime_task*)calloc(task_count, sizeof(moe_model_grouped_expert_runtime_task));
    if (!rt) {
        return 0;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        rt[i].gate = tasks[i].gate;
        rt[i].up = tasks[i].up;
        rt[i].down = tasks[i].down;
        rt[i].x = tasks[i].x;
        rt[i].out = tasks[i].out;
        rt[i].batch_stride = tasks[i].hidden;
        rt[i].out_stride = tasks[i].hidden;
        rt[i].batch = tasks[i].batch;
        rt[i].hidden = tasks[i].hidden;
        rt[i].intermediate = tasks[i].intermediate;
        rt[i].device_io = 0;
    }
    const int ok = moe_model_run_grouped_expert_triplets_runtime_fp32_blas(backend, rt, task_count);
    free(rt);
    return ok;
}

static int moe_model_run_grouped_expert_triplets_device_fp32_blas(
    int32_t backend,
    const moe_model_grouped_expert_device_task* tasks,
    uint32_t task_count
) {
    if (!tasks || task_count < 2u) {
        return 0;
    }
    moe_model_grouped_expert_runtime_task* rt =
        (moe_model_grouped_expert_runtime_task*)calloc(task_count, sizeof(moe_model_grouped_expert_runtime_task));
    if (!rt) {
        return 0;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        rt[i].gate = tasks[i].gate;
        rt[i].up = tasks[i].up;
        rt[i].down = tasks[i].down;
        rt[i].x = tasks[i].d_x;
        rt[i].out = tasks[i].d_out;
        rt[i].batch_stride = tasks[i].batch_stride;
        rt[i].out_stride = tasks[i].out_stride;
        rt[i].batch = tasks[i].batch;
        rt[i].hidden = tasks[i].hidden;
        rt[i].intermediate = tasks[i].intermediate;
        rt[i].device_io = 1;
    }
    const int ok = moe_model_run_grouped_expert_triplets_runtime_fp32_blas(backend, rt, task_count);
    free(rt);
    return ok;
}

static int moe_model_run_grouped_expert_triplets_indexed_fp32_blas(
    int32_t backend,
    const moe_model_indexed_expert_task* tasks,
    uint32_t task_count,
    const float* batch_input,
    uint32_t input_stride,
    uint32_t input_count,
    const uint32_t* token_indices,
    const float* token_weights,
    uint32_t assignment_count,
    float* batch_accum,
    uint32_t accum_stride
) {
    if (!tasks || task_count < 2u || !batch_input || !token_indices || !token_weights ||
        !batch_accum || input_count == 0 || assignment_count == 0) {
        return 0;
    }
    const uint32_t batch = tasks[0].batch;
    const uint32_t hidden = tasks[0].hidden;
    const uint32_t intermediate = tasks[0].intermediate;
    if (batch == 0 || hidden == 0 || intermediate == 0 ||
        input_stride < hidden || accum_stride < hidden ||
        hidden > 2147483647u || intermediate > 2147483647u || batch > 65535u) {
        return 0;
    }
    for (uint32_t i = 0; i < assignment_count; ++i) {
        if (token_indices[i] >= input_count) {
            return 0;
        }
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        if (!tasks[i].gate || !tasks[i].up || !tasks[i].down ||
            tasks[i].batch != batch ||
            tasks[i].hidden != hidden ||
            tasks[i].intermediate != intermediate ||
            tasks[i].token_offset > assignment_count ||
            tasks[i].batch > assignment_count - tasks[i].token_offset) {
            return 0;
        }
    }
    moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
    if (!gpu || !gpu->gpu_malloc || !gpu->gpu_free || !gpu->gpu_memcpy) {
        return 0;
    }
    const int memcpy_h2d = 1;
    const int memcpy_d2h = 2;
    moe_gpu_expert_scratch* activation = 0;
    moe_model_grouped_expert_device_task* device_tasks = 0;
    int ok = 0;

    const size_t input_bytes = (size_t)input_count * input_stride * sizeof(float);
    const size_t accum_bytes = (size_t)input_count * accum_stride * sizeof(float);
    const size_t index_bytes = (size_t)assignment_count * sizeof(uint32_t);
    const size_t weight_bytes = (size_t)assignment_count * sizeof(float);
    const size_t gathered_bytes = (size_t)assignment_count * hidden * sizeof(float);
    if (input_bytes == 0 || accum_bytes == 0 || gathered_bytes == 0) {
        return 0;
    }
    activation = moe_acquire_expert_scratch(gpu, assignment_count, hidden, intermediate);
    if (!activation ||
        !moe_model_grow_device_buffer(gpu, &activation->d_batch_input,
                                      &activation->batch_input_bytes, input_bytes) ||
        !moe_model_grow_device_buffer(gpu, &activation->d_batch_accum,
                                      &activation->batch_accum_bytes, accum_bytes) ||
        !moe_model_grow_device_buffer(gpu, &activation->d_token_indices,
                                      &activation->token_indices_bytes, index_bytes) ||
        !moe_model_grow_device_buffer(gpu, &activation->d_token_weights,
                                      &activation->token_weights_bytes, weight_bytes)) {
        goto cleanup;
    }
    device_tasks =
        (moe_model_grouped_expert_device_task*)calloc(task_count, sizeof(moe_model_grouped_expert_device_task));
    if (!device_tasks) {
        goto cleanup;
    }
    if (gpu->gpu_memcpy(activation->d_batch_input, batch_input, input_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(activation->d_batch_accum, batch_accum, accum_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(activation->d_token_indices, token_indices, index_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(activation->d_token_weights, token_weights, weight_bytes, memcpy_h2d) != 0) {
        goto cleanup;
    }
    if (!moe_model_device_gather_rows(
            gpu, activation->d_batch_input, input_stride, activation->d_token_indices,
            assignment_count, hidden, activation->d_x)) {
        goto cleanup;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        const size_t row_offset = (size_t)tasks[i].token_offset * hidden * sizeof(float);
        device_tasks[i].layer = tasks[i].layer;
        device_tasks[i].expert = tasks[i].expert;
        device_tasks[i].gate = tasks[i].gate;
        device_tasks[i].up = tasks[i].up;
        device_tasks[i].down = tasks[i].down;
        device_tasks[i].d_x = (const void*)((const char*)activation->d_x + row_offset);
        device_tasks[i].d_out = (void*)((char*)activation->d_out + row_offset);
        device_tasks[i].batch_stride = hidden;
        device_tasks[i].out_stride = hidden;
        device_tasks[i].batch = tasks[i].batch;
        device_tasks[i].hidden = hidden;
        device_tasks[i].intermediate = intermediate;
    }
    if (!moe_model_run_grouped_expert_triplets_device_fp32_blas(
            backend, device_tasks, task_count)) {
        goto cleanup;
    }
    if (!moe_model_device_weighted_accum_rows(
            gpu, activation->d_out, activation->d_token_indices, activation->d_token_weights,
            assignment_count, hidden, accum_stride, activation->d_batch_accum)) {
        goto cleanup;
    }
    if (gpu->gpu_memcpy(batch_accum, activation->d_batch_accum, accum_bytes, memcpy_d2h) != 0) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(device_tasks);
    if (activation) {
        moe_release_expert_scratch(activation);
    }
    return ok;
}

static int moe_model_run_grouped_expert_triplets_indexed_device_fp32_blas(
    int32_t backend,
    const moe_model_indexed_expert_task* tasks,
    uint32_t task_count,
    const void* d_batch_input,
    uint32_t input_stride,
    uint32_t input_count,
    const void* d_token_indices,
    const void* d_token_weights,
    uint32_t assignment_count,
    void* d_batch_accum,
    uint32_t accum_stride
) {
    if (!tasks || task_count < 2u || !d_batch_input || !d_token_indices || !d_token_weights ||
        !d_batch_accum || input_count == 0 || assignment_count == 0) {
        return 0;
    }
    const uint32_t batch = tasks[0].batch;
    const uint32_t hidden = tasks[0].hidden;
    const uint32_t intermediate = tasks[0].intermediate;
    if (batch == 0 || hidden == 0 || intermediate == 0 ||
        input_stride < hidden || accum_stride < hidden ||
        hidden > 2147483647u || intermediate > 2147483647u || batch > 65535u) {
        return 0;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        if (!tasks[i].gate || !tasks[i].up || !tasks[i].down ||
            tasks[i].batch != batch ||
            tasks[i].hidden != hidden ||
            tasks[i].intermediate != intermediate ||
            tasks[i].token_offset > assignment_count ||
            tasks[i].batch > assignment_count - tasks[i].token_offset) {
            return 0;
        }
    }
    moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
    if (!gpu || !gpu->gpu_malloc || !gpu->gpu_free || !gpu->gpu_memcpy) {
        return 0;
    }
    moe_gpu_expert_scratch* activation =
        moe_acquire_expert_scratch(gpu, assignment_count, hidden, intermediate);
    if (!activation) {
        return 0;
    }
    moe_model_grouped_expert_device_task* device_tasks =
        (moe_model_grouped_expert_device_task*)calloc(task_count, sizeof(moe_model_grouped_expert_device_task));
    int ok = 0;
    if (!device_tasks) {
        goto cleanup;
    }
    if (!moe_model_device_gather_rows(
            gpu, d_batch_input, input_stride, d_token_indices,
            assignment_count, hidden, activation->d_x)) {
        goto cleanup;
    }
    for (uint32_t i = 0; i < task_count; ++i) {
        const size_t row_offset = (size_t)tasks[i].token_offset * hidden * sizeof(float);
        device_tasks[i].layer = tasks[i].layer;
        device_tasks[i].expert = tasks[i].expert;
        device_tasks[i].gate = tasks[i].gate;
        device_tasks[i].up = tasks[i].up;
        device_tasks[i].down = tasks[i].down;
        device_tasks[i].d_x = (const void*)((const char*)activation->d_x + row_offset);
        device_tasks[i].d_out = (void*)((char*)activation->d_out + row_offset);
        device_tasks[i].batch_stride = hidden;
        device_tasks[i].out_stride = hidden;
        device_tasks[i].batch = tasks[i].batch;
        device_tasks[i].hidden = hidden;
        device_tasks[i].intermediate = intermediate;
    }
    if (!moe_model_run_grouped_expert_triplets_device_fp32_blas(
            backend, device_tasks, task_count)) {
        goto cleanup;
    }
    if (!moe_model_device_weighted_accum_rows(
            gpu, activation->d_out, d_token_indices, d_token_weights,
            assignment_count, hidden, accum_stride, d_batch_accum)) {
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(device_tasks);
    moe_release_expert_scratch(activation);
    return ok;
}

static int moe_model_prefill_activation_acquire_f32(
    int32_t backend,
    const float* batch_input,
    uint32_t input_stride,
    uint32_t input_count,
    const uint32_t* token_indices,
    const float* token_weights,
    uint32_t assignment_count,
    const float* batch_accum,
    uint32_t accum_stride,
    uint64_t* out_handle,
    const void** out_d_batch_input,
    const void** out_d_token_indices,
    const void** out_d_token_weights,
    void** out_d_batch_accum
) {
    if (out_handle) *out_handle = 0;
    if (out_d_batch_input) *out_d_batch_input = 0;
    if (out_d_token_indices) *out_d_token_indices = 0;
    if (out_d_token_weights) *out_d_token_weights = 0;
    if (out_d_batch_accum) *out_d_batch_accum = 0;
    if (!batch_input || !token_indices || !token_weights || !batch_accum ||
        !out_handle || !out_d_batch_input || !out_d_token_indices ||
        !out_d_token_weights || !out_d_batch_accum ||
        input_stride == 0 || input_count == 0 || assignment_count == 0 ||
        accum_stride == 0) {
        return 0;
    }
    moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
    if (!gpu || !gpu->gpu_memcpy) {
        return 0;
    }
    for (uint32_t i = 0; i < assignment_count; ++i) {
        if (token_indices[i] >= input_count) {
            return 0;
        }
    }
    const int memcpy_h2d = 1;
    const size_t input_bytes = (size_t)input_count * input_stride * sizeof(float);
    const size_t accum_bytes = (size_t)input_count * accum_stride * sizeof(float);
    const size_t index_bytes = (size_t)assignment_count * sizeof(uint32_t);
    const size_t weight_bytes = (size_t)assignment_count * sizeof(float);
    moe_gpu_expert_scratch* s =
        moe_acquire_prefill_activation_scratch(gpu, input_bytes, accum_bytes, index_bytes, weight_bytes);
    if (!s) {
        return 0;
    }
    if (gpu->gpu_memcpy(s->d_batch_input, batch_input, input_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(s->d_batch_accum, batch_accum, accum_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(s->d_token_indices, token_indices, index_bytes, memcpy_h2d) != 0 ||
        gpu->gpu_memcpy(s->d_token_weights, token_weights, weight_bytes, memcpy_h2d) != 0) {
        moe_release_expert_scratch(s);
        return 0;
    }
    *out_handle = (uint64_t)(uintptr_t)s;
    *out_d_batch_input = s->d_batch_input;
    *out_d_token_indices = s->d_token_indices;
    *out_d_token_weights = s->d_token_weights;
    *out_d_batch_accum = s->d_batch_accum;
    return 1;
}

static int moe_model_prefill_activation_release_f32(
    int32_t backend,
    uint64_t handle,
    float* batch_accum,
    uint32_t input_count,
    uint32_t accum_stride,
    int copy_back
) {
    if (!handle) {
        return 0;
    }
    moe_gpu_expert_scratch* s = (moe_gpu_expert_scratch*)(uintptr_t)handle;
    int ok = 1;
    if (copy_back) {
        if (!batch_accum || input_count == 0 || accum_stride == 0) {
            ok = 0;
        } else {
            moe_dyn_gpu_blas* gpu = moe_model_gpu_blas(backend);
            const int memcpy_d2h = 2;
            const size_t accum_bytes = (size_t)input_count * accum_stride * sizeof(float);
            if (!gpu || !gpu->gpu_memcpy ||
                gpu->gpu_memcpy(batch_accum, s->d_batch_accum, accum_bytes, memcpy_d2h) != 0) {
                ok = 0;
            }
        }
    }
    moe_release_expert_scratch(s);
    return ok;
}

)MOE_MODEL_LIB";
    s << "extern \"C\" {\n";
    s << "moe_EXPORT uint32_t moe_model_lib_abi_version(){return 1u;}\n";
    s << "moe_EXPORT const char* moe_model_lib_model_id(){return \"storagellm-offload\";}\n";
    s << "moe_EXPORT const char* moe_model_lib_backend(){return \""
      << model_lib_backend_tag(backend) << "\";}\n";
    s << "moe_EXPORT uint64_t moe_model_lib_raw_tensor_bytes(){return "
      << (c ? c->raw_tensor_total_bytes : 0ull) << "ull;}\n";
    s << "moe_EXPORT uint32_t moe_model_lib_layers(){return "
      << (c ? c->num_hidden_layers : 0u) << "u;}\n";
    s << "moe_EXPORT uint32_t moe_model_lib_first_moe_layer(){return "
      << (c ? c->first_moe_layer : 0u) << "u;}\n";
    s << "moe_EXPORT uint32_t moe_model_lib_last_moe_layer(){return "
      << (c ? c->last_moe_layer : 0u) << "u;}\n";
    s << "moe_EXPORT uint32_t moe_model_lib_has_expert_triplet_batch_f32(){return "
      << generated_expert_batch_kernel << "u;}\n";
    s << "moe_EXPORT uint32_t moe_model_lib_l0_mmap_checksum_spv_size(){return 0u;}\n";
    s << "moe_EXPORT const void* moe_model_lib_l0_mmap_checksum_spv(){return 0;}\n";
    s << "moe_EXPORT const char* moe_model_lib_l0_mmap_checksum_kernel_name(){return \"storagellm_mmap_checksum\";}\n";
    s << "moe_EXPORT int moe_model_lib_run_expert_triplet_batch_f32("
         "uint32_t abi_version,int32_t backend,int32_t layer,int32_t expert,"
         "const void* gate,const void* up,const void* down,"
         "const float* const* batch_hidden,uint32_t batch_size,"
         "uint32_t hidden_size,uint32_t intermediate_size,float* out_pool,"
         "uint32_t out_count,int activation_mode){"
         "(void)layer;(void)expert;"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_expert_triplet_batch_fp32_blas("
         "backend,"
         "(const moe_model_device_tensor_batch_view*)gate,"
         "(const moe_model_device_tensor_batch_view*)up,"
         "(const moe_model_device_tensor_batch_view*)down,"
         "batch_hidden,0,0,0,hidden_size,hidden_size,batch_size,hidden_size,intermediate_size,"
         "out_pool,out_count,activation_mode);}\n";
    s << "moe_EXPORT int moe_model_lib_run_expert_triplet_batch_contiguous_f32("
         "uint32_t abi_version,int32_t backend,int32_t layer,int32_t expert,"
         "const void* gate,const void* up,const void* down,"
         "const float* batch_hidden_contiguous,uint32_t batch_stride,uint32_t batch_size,"
         "uint32_t hidden_size,uint32_t intermediate_size,float* out_pool,"
         "uint32_t out_count,int activation_mode){"
         "(void)layer;(void)expert;"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_expert_triplet_batch_fp32_blas("
         "backend,"
         "(const moe_model_device_tensor_batch_view*)gate,"
         "(const moe_model_device_tensor_batch_view*)up,"
         "(const moe_model_device_tensor_batch_view*)down,"
         "0,batch_hidden_contiguous,0,0,hidden_size,batch_stride,batch_size,hidden_size,intermediate_size,"
         "out_pool,out_count,activation_mode);}\n";
    s << "moe_EXPORT int moe_model_lib_run_expert_triplet_batch_device_f32("
         "uint32_t abi_version,int32_t backend,int32_t layer,int32_t expert,"
         "const void* gate,const void* up,const void* down,"
         "const void* d_batch_hidden,uint32_t batch_stride,uint32_t batch_size,"
         "uint32_t hidden_size,uint32_t intermediate_size,void* d_out,"
         "uint32_t out_stride,int activation_mode){"
         "(void)layer;(void)expert;"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_expert_triplet_batch_fp32_blas("
         "backend,"
         "(const moe_model_device_tensor_batch_view*)gate,"
         "(const moe_model_device_tensor_batch_view*)up,"
         "(const moe_model_device_tensor_batch_view*)down,"
         "0,0,d_batch_hidden,d_out,out_stride,batch_stride,batch_size,hidden_size,intermediate_size,"
         "0,hidden_size,activation_mode);}\n";
    s << "moe_EXPORT int moe_model_lib_run_grouped_expert_triplets_f32("
         "uint32_t abi_version,int32_t backend,const moe_model_grouped_expert_task* tasks,"
         "uint32_t task_count){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_grouped_expert_triplets_fp32_blas(backend,tasks,task_count);}\n";
    s << "moe_EXPORT int moe_model_lib_run_grouped_expert_triplets_device_f32("
         "uint32_t abi_version,int32_t backend,const moe_model_grouped_expert_device_task* tasks,"
         "uint32_t task_count){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_grouped_expert_triplets_device_fp32_blas(backend,tasks,task_count);}\n";
    s << "moe_EXPORT int moe_model_lib_run_grouped_expert_triplets_indexed_f32("
         "uint32_t abi_version,int32_t backend,const moe_model_indexed_expert_task* tasks,"
         "uint32_t task_count,const float* batch_input,uint32_t input_stride,uint32_t input_count,"
         "const uint32_t* token_indices,const float* token_weights,uint32_t assignment_count,"
         "float* batch_accum,uint32_t accum_stride){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_grouped_expert_triplets_indexed_fp32_blas("
         "backend,tasks,task_count,batch_input,input_stride,input_count,"
         "token_indices,token_weights,assignment_count,batch_accum,accum_stride);}\n";
    s << "moe_EXPORT int moe_model_lib_run_grouped_expert_triplets_indexed_device_f32("
         "uint32_t abi_version,int32_t backend,const moe_model_indexed_expert_task* tasks,"
         "uint32_t task_count,const void* d_batch_input,uint32_t input_stride,uint32_t input_count,"
         "const void* d_token_indices,const void* d_token_weights,uint32_t assignment_count,"
         "void* d_batch_accum,uint32_t accum_stride){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_run_grouped_expert_triplets_indexed_device_fp32_blas("
         "backend,tasks,task_count,d_batch_input,input_stride,input_count,"
         "d_token_indices,d_token_weights,assignment_count,d_batch_accum,accum_stride);}\n";
    s << "moe_EXPORT int moe_model_lib_prefill_activation_acquire_f32("
         "uint32_t abi_version,int32_t backend,const float* batch_input,uint32_t input_stride,"
         "uint32_t input_count,const uint32_t* token_indices,const float* token_weights,"
         "uint32_t assignment_count,const float* batch_accum,uint32_t accum_stride,"
         "uint64_t* out_handle,const void** out_d_batch_input,const void** out_d_token_indices,"
         "const void** out_d_token_weights,void** out_d_batch_accum){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_prefill_activation_acquire_f32("
         "backend,batch_input,input_stride,input_count,token_indices,token_weights,"
         "assignment_count,batch_accum,accum_stride,out_handle,out_d_batch_input,"
         "out_d_token_indices,out_d_token_weights,out_d_batch_accum);}\n";
    s << "moe_EXPORT int moe_model_lib_prefill_activation_release_f32("
         "uint32_t abi_version,int32_t backend,uint64_t handle,float* batch_accum,"
         "uint32_t input_count,uint32_t accum_stride,int copy_back){"
         "if(abi_version!=1u)return 0;"
         "return moe_model_prefill_activation_release_f32("
         "backend,handle,batch_accum,input_count,accum_stride,copy_back);}\n";
    s << "moe_EXPORT uint64_t moe_model_lib_gpu_expert_scratch_wait_ns(){"
         "return moe_model_gpu_expert_scratch_wait_ns.load(std::memory_order_relaxed);}\n";
    s << "moe_EXPORT uint64_t moe_model_lib_gpu_expert_scratch_backpressure_count(){"
         "return moe_model_gpu_expert_scratch_backpressure.load(std::memory_order_relaxed);}\n";
    s << "}\n";
    return s.str();
}

static int model_lib_write_source(moe_pc_engine* e) {
    if (!e || e->model_lib_source_path.empty()) {
        return 0;
    }
    std::ofstream out;
    open_ofstream_path(out, e->model_lib_source_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return 0;
    }
    const std::string source = model_lib_source_text(e);
    out.write(source.data(), (std::streamsize)source.size());
    e->model_lib_generated = out.good() ? 1 : 0;
    return e->model_lib_generated;
}

static void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static int model_lib_try_compile(moe_pc_engine* e) {
    if (!e || e->model_lib_compile_command.empty()) {
        return 0;
    }
    std::string cmd = e->model_lib_compile_command;
    replace_all(cmd, "{src}", e->model_lib_source_path);
    replace_all(cmd, "{out}", e->model_lib_path);
    replace_all(cmd, "{backend}", model_lib_backend_tag(e->backend_caps.backend));
    e->model_lib_compile_attempted = 1;
    int rc = -1;
#ifdef _WIN32
    STARTUPINFOW si = {};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi = {};
    const std::wstring wide_cmd = storagellm::wide_path_from_utf8(cmd);
    std::vector<wchar_t> cmdline(wide_cmd.begin(), wide_cmd.end());
    cmdline.push_back(L'\0');
    if (!wide_cmd.empty() &&
        CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        DWORD exitCode = 1;
        GetExitCodeProcess(pi.hProcess, &exitCode);
        rc = (int)exitCode;
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
#else
    pid_t pid = fork();
    if (pid == 0) {
        // Fix 22: Enhanced POSIX arg parsing with single quotes + backslash escapes
        std::vector<std::string> args;
        std::string current_arg;
        bool in_double = false;
        bool in_single = false;
        bool escaped = false;
        for (char c : cmd) {
            if (escaped) {
                current_arg += c;
                escaped = false;
            } else if (c == '\\' && !in_single) {
                escaped = true;
            } else if (c == '\'' && !in_double) {
                in_single = !in_single;
            } else if (c == '"' && !in_single) {
                in_double = !in_double;
            } else if (c == ' ' && !in_double && !in_single) {
                if (!current_arg.empty()) {
                    args.push_back(current_arg);
                    current_arg.clear();
                }
            } else {
                current_arg += c;
            }
        }
        if (!current_arg.empty()) {
            args.push_back(current_arg);
        }
        
        std::vector<char*> c_args;
        for (auto& a : args) {
            c_args.push_back(&a[0]);
        }
        c_args.push_back(nullptr);
        
        execvp(c_args[0], c_args.data());
        _exit(127);
    } else if (pid > 0) {
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            rc = WEXITSTATUS(status);
        }
    }
#endif
    e->model_lib_compile_succeeded = (rc == 0 && path_exists_readable(e->model_lib_path)) ? 1 : 0;
    return e->model_lib_compile_succeeded;
}
