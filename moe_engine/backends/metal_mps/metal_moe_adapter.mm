#include "moe_pc_engine.h"

#if defined(STORAGELLM_HAS_METAL_MPS)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <mutex>

struct storagellm_metal_tensor_view {
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

static id<MTLBuffer> storagellm_metal_weight_buffer_fp32(const void* view_ptr, uint32_t rows, uint32_t cols, uint64_t* byte_offset) {
    if (byte_offset) *byte_offset = 0;
    const auto* v = reinterpret_cast<const storagellm_metal_tensor_view*>(view_ptr);
    if (!v || !v->ptr || v->rows != rows || v->cols != cols) return nil;
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    const uint64_t total_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= total_bytes && v->weight_bytes >= total_bytes && v->weight_row_bytes >= row_bytes) {
        return (__bridge id<MTLBuffer>)reinterpret_cast<void*>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u && v->expert_gpu_layout_size >= total_bytes &&
        v->expert_gpu_layout_row_bytes >= row_bytes && v->expert_gpu_layout_offset <= v->bytes &&
        total_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        if (byte_offset) *byte_offset = v->expert_gpu_layout_offset;
        return (__bridge id<MTLBuffer>)reinterpret_cast<void*>(static_cast<uintptr_t>(v->ptr));
    }
    return nil;
}

static NSString* storagellm_metal_source() {
    return @"#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "static inline float gelu_erf_f(float x){return 0.5f*x*(1.0f+erf(x*0.7071067811865476f));}\n"
           "static inline float gelu_tanh_f(float x){float k=0.7978845608028654f;float inner=k*(x+0.044715f*x*x*x);return 0.5f*x*(1.0f+tanh(inner));}\n"
           "static inline float act_f(uint mode,float g,float u){if(!isfinite(g)||!isfinite(u))return 0.0f;float a=mode==2?gelu_tanh_f(g):(mode==1?gelu_erf_f(g):(g>40.0f?g:(g<-40.0f?0.0f:g/(1.0f+exp(-g)))));float y=a*u;return isfinite(y)?y:0.0f;}\n"
           "kernel void storagellm_metal_fused_moe_f32(\n"
           "device const float* gate [[buffer(0)]], constant ulong& gate_off_f [[buffer(1)]],\n"
           "device const float* up [[buffer(2)]], constant ulong& up_off_f [[buffer(3)]],\n"
           "device const float* down [[buffer(4)]], constant ulong& down_off_f [[buffer(5)]],\n"
           "device const float* input [[buffer(6)]], constant uint& input_stride [[buffer(7)]],\n"
           "device const uint* token_indices [[buffer(8)]], device const float* token_weights [[buffer(9)]],\n"
           "constant uint& assignment_offset [[buffer(10)]], constant uint& assignment_count [[buffer(11)]],\n"
           "constant uint& hidden [[buffer(12)]], constant uint& intermediate [[buffer(13)]], constant uint& activation_mode [[buffer(14)]],\n"
           "device float* accum [[buffer(15)]], constant uint& accum_stride [[buffer(16)]], uint2 gid [[thread_position_in_grid]]){\n"
           "uint local_row=gid.x;uint h=gid.y;if(local_row>=assignment_count||h>=hidden)return;uint row=assignment_offset+local_row;uint token=token_indices[row];float route=token_weights?token_weights[row]:1.0f;if(!isfinite(route))return;device const float* x=input+(ulong)token*input_stride;float y=0.0f;for(uint r=0;r<intermediate;++r){float g=0.0f;float u=0.0f;device const float* gw=gate+gate_off_f+(ulong)r*hidden;device const float* uw=up+up_off_f+(ulong)r*hidden;for(uint c=0;c<hidden;++c){float xv=x[c];g=fma(gw[c],xv,g);u=fma(uw[c],xv,u);}y=fma(down[down_off_f+(ulong)h*intermediate+r],act_f(activation_mode,g,u),y);}accum[(ulong)token*accum_stride+h]+=y*route;}\n";
}

struct storagellm_metal_cache {
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    std::mutex mutex;
};
static storagellm_metal_cache g_metal_cache;

static id<MTLComputePipelineState> storagellm_metal_pipeline(id<MTLDevice> device) {
    if (!device) return nil;
    std::lock_guard<std::mutex> lock(g_metal_cache.mutex);
    if (g_metal_cache.pipeline && g_metal_cache.device == device) return g_metal_cache.pipeline;
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:storagellm_metal_source() options:nil error:&err];
    if (!lib) return nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"storagellm_metal_fused_moe_f32"];
    if (!fn) return nil;
    g_metal_cache.pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
    g_metal_cache.device = device;
    return g_metal_cache.pipeline;
}

static int storagellm_metal_run(const moe_grouped_expert_device_task_t* tasks, uint32_t task_count, id<MTLCommandQueue> queue, id<MTLCommandBuffer> supplied_cb) {
    if (!tasks || task_count == 0 || !queue) return 0;
    id<MTLDevice> device = [queue device];
    id<MTLComputePipelineState> pipe = storagellm_metal_pipeline(device);
    if (!pipe) return 0;
    id<MTLCommandBuffer> cb = supplied_cb ? supplied_cb : [queue commandBuffer];
    if (!cb) return 0;
    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        if (!t.gate_weight || !t.up_weight || !t.down_weight || !t.d_input || !t.d_token_indices ||
            !t.d_token_weights || !t.d_accum || t.assignment_count == 0 ||
            t.input_stride < t.hidden_size || t.accum_stride < t.hidden_size ||
            t.hidden_size == 0 || t.intermediate_size == 0) return 0;
        uint64_t gate_off = 0, up_off = 0, down_off = 0;
        id<MTLBuffer> gate = storagellm_metal_weight_buffer_fp32(t.gate_weight, t.intermediate_size, t.hidden_size, &gate_off);
        id<MTLBuffer> up = storagellm_metal_weight_buffer_fp32(t.up_weight, t.intermediate_size, t.hidden_size, &up_off);
        id<MTLBuffer> down = storagellm_metal_weight_buffer_fp32(t.down_weight, t.hidden_size, t.intermediate_size, &down_off);
        if (!gate || !up || !down) return 0;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)const_cast<void*>(t.d_input);
        id<MTLBuffer> idx = (__bridge id<MTLBuffer>)const_cast<uint32_t*>(t.d_token_indices);
        id<MTLBuffer> weights = (__bridge id<MTLBuffer>)const_cast<float*>(t.d_token_weights);
        id<MTLBuffer> accum = (__bridge id<MTLBuffer>)t.d_accum;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (!enc) return 0;
        [enc setComputePipelineState:pipe];
        const uint64_t gate_f = gate_off / sizeof(float);
        const uint64_t up_f = up_off / sizeof(float);
        const uint64_t down_f = down_off / sizeof(float);
        [enc setBuffer:gate offset:0 atIndex:0]; [enc setBytes:&gate_f length:sizeof(gate_f) atIndex:1];
        [enc setBuffer:up offset:0 atIndex:2]; [enc setBytes:&up_f length:sizeof(up_f) atIndex:3];
        [enc setBuffer:down offset:0 atIndex:4]; [enc setBytes:&down_f length:sizeof(down_f) atIndex:5];
        [enc setBuffer:input offset:0 atIndex:6]; [enc setBytes:&t.input_stride length:sizeof(t.input_stride) atIndex:7];
        [enc setBuffer:idx offset:0 atIndex:8]; [enc setBuffer:weights offset:0 atIndex:9];
        [enc setBytes:&t.assignment_offset length:sizeof(t.assignment_offset) atIndex:10];
        [enc setBytes:&t.assignment_count length:sizeof(t.assignment_count) atIndex:11];
        [enc setBytes:&t.hidden_size length:sizeof(t.hidden_size) atIndex:12];
        [enc setBytes:&t.intermediate_size length:sizeof(t.intermediate_size) atIndex:13];
        [enc setBytes:&t.activation_mode length:sizeof(t.activation_mode) atIndex:14];
        [enc setBuffer:accum offset:0 atIndex:15]; [enc setBytes:&t.accum_stride length:sizeof(t.accum_stride) atIndex:16];
        MTLSize grid = MTLSizeMake(t.assignment_count, t.hidden_size, 1);
        NSUInteger w = pipe.threadExecutionWidth ? pipe.threadExecutionWidth : 32;
        MTLSize tg = MTLSizeMake(1, w, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
    if (!supplied_cb) [cb commit];
    return 1;
}
#endif

extern "C" int storagellm_metal_mps_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_METAL_MPS)
    (void)backend; (void)tasks; (void)task_count; (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_METAL || !stream_or_queue) return 0;
    return storagellm_metal_run(tasks, task_count, (__bridge id<MTLCommandQueue>)stream_or_queue, nil);
#endif
}

extern "C" int storagellm_metal_mps_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) return 0;
#if !defined(STORAGELLM_HAS_METAL_MPS)
    return 0;
#else
    id<MTLCommandQueue> q = nil;
    id<MTLCommandBuffer> cb = nil;
    if (request->context && request->context->context_kind == moe_FAST_BACKEND_CONTEXT_METAL) {
        q = (__bridge id<MTLCommandQueue>)request->context->u.metal.command_queue;
        cb = (__bridge id<MTLCommandBuffer>)request->context->u.metal.command_buffer;
    }
    if (!q && request->legacy_stream_or_queue) q = (__bridge id<MTLCommandQueue>)request->legacy_stream_or_queue;
    return storagellm_metal_run(request->tasks, request->task_count, q, cb);
#endif
}
