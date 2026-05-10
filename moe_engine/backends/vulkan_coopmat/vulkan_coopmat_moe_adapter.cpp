#include "moe_pc_engine.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(STORAGELLM_HAS_VULKAN_COOPMAT)
#include <vulkan/vulkan.h>
#include <shaderc/shaderc.hpp>

struct storagellm_vk_tensor_view {
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

static VkBuffer storagellm_vk_weight_buffer_fp32(
    const void* view_ptr,
    uint32_t rows,
    uint32_t cols,
    VkDeviceSize* offset_bytes,
    VkDeviceSize* range_bytes
) {
    if (offset_bytes) *offset_bytes = 0;
    if (range_bytes) *range_bytes = 0;
    const auto* v = reinterpret_cast<const storagellm_vk_tensor_view*>(view_ptr);
    if (!v || !v->ptr || v->rows != rows || v->cols != cols) return VK_NULL_HANDLE;
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * sizeof(float);
    const uint64_t total_bytes = static_cast<uint64_t>(rows) * row_bytes;
    if (v->weight_format == static_cast<uint32_t>(moe_WEIGHT_ENCODING_RAW_FP32) &&
        v->bytes >= total_bytes && v->weight_bytes >= total_bytes && v->weight_row_bytes >= row_bytes) {
        if (range_bytes) *range_bytes = static_cast<VkDeviceSize>(total_bytes);
        return reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(v->ptr));
    }
    if (v->expert_gpu_layout_kind == 3u &&
        v->expert_gpu_layout_size >= total_bytes &&
        v->expert_gpu_layout_row_bytes >= row_bytes &&
        v->expert_gpu_layout_offset <= v->bytes &&
        total_bytes <= v->bytes - v->expert_gpu_layout_offset) {
        if (offset_bytes) *offset_bytes = static_cast<VkDeviceSize>(v->expert_gpu_layout_offset);
        if (range_bytes) *range_bytes = static_cast<VkDeviceSize>(total_bytes);
        return reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(v->ptr));
    }
    return VK_NULL_HANDLE;
}

struct storagellm_vk_push_constants {
    uint32_t input_stride;
    uint32_t assignment_offset;
    uint32_t assignment_count;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t activation_mode;
    uint32_t accum_stride;
    uint32_t _pad;
};

static const char* storagellm_vk_fused_moe_glsl = R"GLSL(
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0) readonly buffer GateBuf { float gate[]; };
layout(set = 0, binding = 1) readonly buffer UpBuf { float up[]; };
layout(set = 0, binding = 2) readonly buffer DownBuf { float down[]; };
layout(set = 0, binding = 3) readonly buffer InputBuf { float input_x[]; };
layout(set = 0, binding = 4) readonly buffer IndexBuf { uint token_indices[]; };
layout(set = 0, binding = 5) readonly buffer WeightBuf { float token_weights[]; };
layout(set = 0, binding = 6) buffer AccumBuf { float accum[]; };
layout(push_constant) uniform Params {
    uint input_stride;
    uint assignment_offset;
    uint assignment_count;
    uint hidden;
    uint intermediate;
    uint activation_mode;
    uint accum_stride;
    uint _pad;
} p;
float gelu_erf_f32(float x) {
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}
float gelu_tanh_f32(float x) {
    const float k = 0.7978845608028654;
    float inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}
float act_f32(uint mode, float g, float u) {
    if (isnan(g) || isnan(u) || isinf(g) || isinf(u)) return 0.0;
    float a = 0.0;
    if (mode == 2u) a = gelu_tanh_f32(g);
    else if (mode == 1u) a = gelu_erf_f32(g);
    else a = g > 40.0 ? g : (g < -40.0 ? 0.0 : g / (1.0 + exp(-g)));
    float y = a * u;
    return (isnan(y) || isinf(y)) ? 0.0 : y;
}
void main() {
    uint local_row = gl_GlobalInvocationID.x;
    uint h = gl_GlobalInvocationID.y;
    if (local_row >= p.assignment_count || h >= p.hidden) return;
    uint row = p.assignment_offset + local_row;
    uint token = token_indices[row];
    float route = token_weights[row];
    if (isnan(route) || isinf(route)) return;
    float y = 0.0;
    for (uint r = 0u; r < p.intermediate; ++r) {
        float g = 0.0;
        float u = 0.0;
        uint gw_base = r * p.hidden;
        uint x_base = token * p.input_stride;
        for (uint c = 0u; c < p.hidden; ++c) {
            float xv = input_x[x_base + c];
            g = fma(gate[gw_base + c], xv, g);
            u = fma(up[gw_base + c], xv, u);
        }
        y = fma(down[h * p.intermediate + r], act_f32(p.activation_mode, g, u), y);
    }
    atomicAdd(accum[token * p.accum_stride + h], y * route);
}
)GLSL";

struct storagellm_vk_pipeline_cache_entry {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
};

static std::mutex g_vk_cache_mutex;
static std::unordered_map<VkDevice, storagellm_vk_pipeline_cache_entry> g_vk_cache;

static int storagellm_vk_build_pipeline(VkDevice device, storagellm_vk_pipeline_cache_entry* out) {
    if (!device || !out) return 0;
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);
    auto result = compiler.CompileGlslToSpv(
        storagellm_vk_fused_moe_glsl,
        shaderc_compute_shader,
        "storagellm_vulkan_fused_moe.comp",
        opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        return 0;
    }
    std::vector<uint32_t> spirv(result.cbegin(), result.cend());
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode = spirv.data();
    if (vkCreateShaderModule(device, &smci, nullptr, &out->shader) != VK_SUCCESS) return 0;

    VkDescriptorSetLayoutBinding bindings[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 7;
    dlci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &dlci, nullptr, &out->set_layout) != VK_SUCCESS) return 0;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(storagellm_vk_push_constants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &out->set_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &out->pipeline_layout) != VK_SUCCESS) return 0;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = out->shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = out->pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &out->pipeline) != VK_SUCCESS) return 0;

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 7u * 1024u;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 1024u;
    dpci.poolSizeCount = 1u;
    dpci.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(device, &dpci, nullptr, &out->descriptor_pool) != VK_SUCCESS) return 0;
    return 1;
}

static const storagellm_vk_pipeline_cache_entry* storagellm_vk_get_pipeline(VkDevice device) {
    std::lock_guard<std::mutex> lock(g_vk_cache_mutex);
    auto it = g_vk_cache.find(device);
    if (it != g_vk_cache.end() && it->second.pipeline) return &it->second;
    storagellm_vk_pipeline_cache_entry entry{};
    if (!storagellm_vk_build_pipeline(device, &entry)) return nullptr;
    auto inserted = g_vk_cache.emplace(device, entry);
    return &inserted.first->second;
}

static int storagellm_vk_run(
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    const moe_fast_backend_vulkan_context_t* ctx
) {
    if (!tasks || task_count == 0 || !ctx || !ctx->device) return 0;
    VkDevice device = reinterpret_cast<VkDevice>(ctx->device);
    const auto* pipe = storagellm_vk_get_pipeline(device);
    if (!pipe || !pipe->pipeline || !pipe->pipeline_layout || !pipe->set_layout || !pipe->descriptor_pool) return 0;

    VkCommandBuffer cmd = reinterpret_cast<VkCommandBuffer>(ctx->command_buffer);
    VkCommandBuffer owned_cmd = VK_NULL_HANDLE;
    const bool owns_command_buffer = !cmd;
    if (owns_command_buffer) {
        if (!ctx->command_pool || !ctx->queue) return 0;
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = reinterpret_cast<VkCommandPool>(ctx->command_pool);
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1u;
        if (vkAllocateCommandBuffers(device, &cbai, &owned_cmd) != VK_SUCCESS || !owned_cmd) return 0;
        cmd = owned_cmd;
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, reinterpret_cast<VkCommandPool>(ctx->command_pool), 1u, &owned_cmd);
            return 0;
        }
    }
    VkDescriptorPool pool = ctx->descriptor_pool ?
        reinterpret_cast<VkDescriptorPool>(ctx->descriptor_pool) :
        pipe->descriptor_pool;

    for (uint32_t i = 0; i < task_count; ++i) {
        const auto& t = tasks[i];
        if (!t.gate_weight || !t.up_weight || !t.down_weight || !t.d_input || !t.d_token_indices ||
            !t.d_token_weights || !t.d_accum || t.assignment_count == 0 ||
            t.input_stride < t.hidden_size || t.accum_stride < t.hidden_size ||
            t.hidden_size == 0 || t.intermediate_size == 0) return 0;
        VkDeviceSize gate_off = 0, up_off = 0, down_off = 0;
        VkDeviceSize gate_range = 0, up_range = 0, down_range = 0;
        VkBuffer gate = storagellm_vk_weight_buffer_fp32(t.gate_weight, t.intermediate_size, t.hidden_size, &gate_off, &gate_range);
        VkBuffer up = storagellm_vk_weight_buffer_fp32(t.up_weight, t.intermediate_size, t.hidden_size, &up_off, &up_range);
        VkBuffer down = storagellm_vk_weight_buffer_fp32(t.down_weight, t.hidden_size, t.intermediate_size, &down_off, &down_range);
        if (!gate || !up || !down) return 0;
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &pipe->set_layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(device, &dsai, &set) != VK_SUCCESS || !set) return 0;
        VkDescriptorBufferInfo infos[7]{};
        infos[0] = { gate, gate_off, gate_range };
        infos[1] = { up, up_off, up_range };
        infos[2] = { down, down_off, down_range };
        infos[3] = { reinterpret_cast<VkBuffer>(const_cast<void*>(t.d_input)), 0, VK_WHOLE_SIZE };
        infos[4] = { reinterpret_cast<VkBuffer>(const_cast<uint32_t*>(t.d_token_indices)), 0, VK_WHOLE_SIZE };
        infos[5] = { reinterpret_cast<VkBuffer>(const_cast<float*>(t.d_token_weights)), 0, VK_WHOLE_SIZE };
        infos[6] = { reinterpret_cast<VkBuffer>(t.d_accum), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t b = 0; b < 7; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = set;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);
        storagellm_vk_push_constants pc{};
        pc.input_stride = t.input_stride;
        pc.assignment_offset = t.assignment_offset;
        pc.assignment_count = t.assignment_count;
        pc.hidden = t.hidden_size;
        pc.intermediate = t.intermediate_size;
        pc.activation_mode = t.activation_mode;
        pc.accum_stride = t.accum_stride;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, pipe->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (t.assignment_count + 7u) / 8u, (t.hidden_size + 7u) / 8u, 1u);
    }
    if (owns_command_buffer) {
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, reinterpret_cast<VkCommandPool>(ctx->command_pool), 1u, &owned_cmd);
            return 0;
        }
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1u;
        si.pCommandBuffers = &cmd;
        VkQueue queue = reinterpret_cast<VkQueue>(ctx->queue);
        const VkResult submit_rc = vkQueueSubmit(queue, 1u, &si, VK_NULL_HANDLE);
        if (submit_rc != VK_SUCCESS) {
            vkFreeCommandBuffers(device, reinterpret_cast<VkCommandPool>(ctx->command_pool), 1u, &owned_cmd);
            return 0;
        }
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(device, reinterpret_cast<VkCommandPool>(ctx->command_pool), 1u, &owned_cmd);
    }
    return 1;
}
#endif

extern "C" int storagellm_vulkan_coopmat_grouped_moe_indexed_device_f32(
    int32_t backend,
    const moe_grouped_expert_device_task_t* tasks,
    uint32_t task_count,
    void* stream_or_queue
) {
#if !defined(STORAGELLM_HAS_VULKAN_COOPMAT)
    (void)backend; (void)tasks; (void)task_count; (void)stream_or_queue;
    return 0;
#else
    if (backend != moe_BACKEND_VULKAN || !stream_or_queue) return 0;
    moe_fast_backend_vulkan_context_t ctx{};
    ctx.command_buffer = stream_or_queue;
    // Legacy ABI does not carry VkDevice/descriptor pool. Use ABI v2 for Vulkan compute.
    return storagellm_vk_run(tasks, task_count, &ctx);
#endif
}

extern "C" int storagellm_vulkan_coopmat_grouped_moe_indexed_device_f32_v2(
    const moe_fast_backend_dispatch_request_t* request
) {
#if !defined(STORAGELLM_HAS_VULKAN_COOPMAT)
    (void)request;
    return 0;
#else
    if (!request || request->abi_version != STORAGELLM_FAST_BACKEND_DISPATCH_ABI_V2) return 0;
    if (request->backend != moe_BACKEND_VULKAN || !request->context ||
        request->context->context_kind != moe_FAST_BACKEND_CONTEXT_VULKAN) return 0;
    return storagellm_vk_run(request->tasks, request->task_count, &request->context->u.vulkan);
#endif
}
