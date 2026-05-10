// Vulkan transfer helpers shared by allocator and async copy paths.
// Device allocations are VkBuffer-backed; host bytes move through a
// host-visible staging VkBuffer and are released only after the submit fence.

static constexpr uint32_t moe_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12u;
static constexpr uint32_t moe_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5u;
static constexpr uint32_t moe_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40u;
static constexpr uint32_t moe_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42u;
static constexpr uint32_t moe_VK_STRUCTURE_TYPE_SUBMIT_INFO = 4u;
static constexpr uint32_t moe_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8u;
static constexpr uint32_t moe_VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001u;
static constexpr uint32_t moe_VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002u;
static constexpr uint32_t moe_VK_SHARING_MODE_EXCLUSIVE = 0u;
static constexpr uint32_t moe_VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001u;
static constexpr uint32_t moe_VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002u;
static constexpr uint32_t moe_VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004u;
static constexpr uint32_t moe_VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0u;
static constexpr uint32_t moe_VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001u;

struct moe_vk_buffer_create_info {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    uint64_t size;
    uint32_t usage;
    uint32_t sharingMode;
    uint32_t queueFamilyIndexCount;
    const uint32_t* pQueueFamilyIndices;
};

struct moe_vk_memory_requirements {
    uint64_t size;
    uint64_t alignment;
    uint32_t memoryTypeBits;
};

struct moe_vk_memory_allocate_info {
    uint32_t sType;
    const void* pNext;
    uint64_t allocationSize;
    uint32_t memoryTypeIndex;
};

struct moe_vk_physical_device_memory_properties {
    uint32_t memoryTypeCount;
    struct {
        uint32_t propertyFlags;
        uint32_t heapIndex;
    } memoryTypes[32];
    uint32_t memoryHeapCount;
    struct {
        uint64_t size;
        uint32_t flags;
    } memoryHeaps[16];
};

struct moe_vk_command_buffer_allocate_info {
    uint32_t sType;
    const void* pNext;
    void* commandPool;
    uint32_t level;
    uint32_t commandBufferCount;
};

struct moe_vk_command_buffer_begin_info {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
    const void* pInheritanceInfo;
};

struct moe_vk_buffer_copy {
    uint64_t srcOffset;
    uint64_t dstOffset;
    uint64_t size;
};

struct moe_vk_submit_info {
    uint32_t sType;
    const void* pNext;
    uint32_t waitSemaphoreCount;
    const void* const* pWaitSemaphores;
    const uint32_t* pWaitDstStageMask;
    uint32_t commandBufferCount;
    const void* const* pCommandBuffers;
    uint32_t signalSemaphoreCount;
    const void* const* pSignalSemaphores;
};

struct moe_vk_fence_create_info {
    uint32_t sType;
    const void* pNext;
    uint32_t flags;
};

static int moe_vulkan_transfer_runtime_ready(const moe_pc_engine* e) {
    return e &&
        e->backend_caps.backend == moe_BACKEND_VULKAN &&
        e->vk.library &&
        e->vk.physicalDevice &&
        e->vk.device &&
        e->vk.queue &&
        e->vk.commandPool &&
        e->vk.queueFamilyIndex != UINT32_MAX &&
        e->vk.vkGetPhysicalDeviceMemoryProperties &&
        e->vk.vkCreateBuffer &&
        e->vk.vkDestroyBuffer &&
        e->vk.vkGetBufferMemoryRequirements &&
        e->vk.vkAllocateMemory &&
        e->vk.vkFreeMemory &&
        e->vk.vkBindBufferMemory &&
        e->vk.vkMapMemory &&
        e->vk.vkUnmapMemory &&
        e->vk.vkAllocateCommandBuffers &&
        e->vk.vkFreeCommandBuffers &&
        e->vk.vkBeginCommandBuffer &&
        e->vk.vkEndCommandBuffer &&
        e->vk.vkCmdCopyBuffer &&
        e->vk.vkResetCommandBuffer &&
        e->vk.vkQueueSubmit &&
        e->vk.vkCreateFence &&
        e->vk.vkGetFenceStatus &&
        e->vk.vkWaitForFences &&
        e->vk.vkResetFences &&
        e->vk.vkDestroyFence;
}

static int moe_vulkan_find_memory_type(
    const moe_pc_engine* e,
    uint32_t memory_type_bits,
    uint32_t required_flags,
    uint32_t* out_index
) {
    if (!moe_vulkan_transfer_runtime_ready(e) || !out_index || memory_type_bits == 0) {
        return 0;
    }
    moe_vk_physical_device_memory_properties props{};
    e->vk.vkGetPhysicalDeviceMemoryProperties(e->vk.physicalDevice, &props);
    const uint32_t count = std::min<uint32_t>(props.memoryTypeCount, 32u);
    for (uint32_t i = 0; i < count; ++i) {
        if ((memory_type_bits & (1u << i)) == 0) {
            continue;
        }
        if ((props.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            *out_index = i;
            return 1;
        }
    }
    return 0;
}

static void moe_vulkan_destroy_buffer_allocation(
    moe_pc_engine* e,
    void* buffer,
    void* memory
) {
    if (!e) {
        return;
    }
    if (buffer && e->vk.vkDestroyBuffer && e->vk.device) {
        e->vk.vkDestroyBuffer(e->vk.device, buffer, nullptr);
    }
    if (memory && e->vk.vkFreeMemory && e->vk.device) {
        e->vk.vkFreeMemory(e->vk.device, memory, nullptr);
    }
}

static int moe_vulkan_create_buffer_allocation(
    moe_pc_engine* e,
    uint64_t bytes,
    uint32_t usage,
    uint32_t memory_flags,
    void** out_buffer,
    void** out_memory
) {
    if (out_buffer) {
        *out_buffer = nullptr;
    }
    if (out_memory) {
        *out_memory = nullptr;
    }
    if (!moe_vulkan_transfer_runtime_ready(e) || !out_buffer || !out_memory || bytes == 0) {
        return 0;
    }

    moe_vk_buffer_create_info buffer_info{
        moe_VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        nullptr,
        0u,
        bytes,
        usage,
        moe_VK_SHARING_MODE_EXCLUSIVE,
        0u,
        nullptr
    };

    void* buffer = nullptr;
    if (e->vk.vkCreateBuffer(e->vk.device, &buffer_info, nullptr, &buffer) != 0 || !buffer) {
        return 0;
    }

    moe_vk_memory_requirements req{};
    e->vk.vkGetBufferMemoryRequirements(e->vk.device, buffer, &req);
    if (req.size == 0 || req.memoryTypeBits == 0) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, nullptr);
        return 0;
    }

    uint32_t memory_type = UINT32_MAX;
    if (!moe_vulkan_find_memory_type(e, req.memoryTypeBits, memory_flags, &memory_type)) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, nullptr);
        return 0;
    }

    moe_vk_memory_allocate_info alloc_info{
        moe_VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        nullptr,
        req.size,
        memory_type
    };

    void* memory = nullptr;
    if (e->vk.vkAllocateMemory(e->vk.device, &alloc_info, nullptr, &memory) != 0 || !memory) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, nullptr);
        return 0;
    }
    if (e->vk.vkBindBufferMemory(e->vk.device, buffer, memory, 0) != 0) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, memory);
        return 0;
    }

    *out_buffer = buffer;
    *out_memory = memory;
    return 1;
}

static int moe_vulkan_create_device_buffer(
    moe_pc_engine* e,
    uint64_t bytes,
    void** out_buffer,
    void** out_memory
) {
    return moe_vulkan_create_buffer_allocation(
        e,
        bytes,
        moe_VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        moe_VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        out_buffer,
        out_memory);
}

static int moe_vulkan_create_staging_buffer(
    moe_pc_engine* e,
    uint64_t bytes,
    void** out_buffer,
    void** out_memory,
    void** out_mapped
) {
    if (out_mapped) {
        *out_mapped = nullptr;
    }
    if (!out_mapped) {
        return 0;
    }
    void* buffer = nullptr;
    void* memory = nullptr;
    if (!moe_vulkan_create_buffer_allocation(
            e,
            bytes,
            moe_VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            moe_VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                moe_VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &buffer,
            &memory)) {
        return 0;
    }
    void* mapped = nullptr;
    if (e->vk.vkMapMemory(e->vk.device, memory, 0, bytes, 0, &mapped) != 0 || !mapped) {
        moe_vulkan_destroy_buffer_allocation(e, buffer, memory);
        return 0;
    }
    *out_buffer = buffer;
    *out_memory = memory;
    *out_mapped = mapped;
    return 1;
}

static void* moe_vulkan_create_command_buffer(moe_pc_engine* e) {
    if (!moe_vulkan_transfer_runtime_ready(e)) {
        return nullptr;
    }
    moe_vk_command_buffer_allocate_info alloc_info{
        moe_VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        e->vk.commandPool,
        moe_VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1u
    };
    void* cmd = nullptr;
    if (e->vk.vkAllocateCommandBuffers(e->vk.device, &alloc_info, &cmd) != 0) {
        return nullptr;
    }
    return cmd;
}

static void moe_vulkan_destroy_command_buffer(moe_pc_engine* e, void* cmd) {
    if (!e || !cmd || !e->vk.vkFreeCommandBuffers || !e->vk.device || !e->vk.commandPool) {
        return;
    }
    const void* command_buffer = cmd;
    e->vk.vkFreeCommandBuffers(e->vk.device, e->vk.commandPool, 1u, &command_buffer);
}

static int moe_vulkan_reset_command_buffer(moe_pc_engine* e, void* cmd) {
    return e && cmd && e->vk.vkResetCommandBuffer &&
        e->vk.vkResetCommandBuffer(cmd, 0u) == 0;
}

static int moe_vulkan_record_copy_buffer(
    moe_pc_engine* e,
    void* cmd,
    void* source_buffer,
    void* dst_buffer,
    uint64_t bytes
) {
    if (!moe_vulkan_transfer_runtime_ready(e) || !cmd || !source_buffer ||
        !dst_buffer || bytes == 0) {
        return 0;
    }
    moe_vk_command_buffer_begin_info begin_info{
        moe_VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        moe_VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };
    if (e->vk.vkBeginCommandBuffer(cmd, &begin_info) != 0) {
        return 0;
    }
    moe_vk_buffer_copy region{0u, 0u, bytes};
    e->vk.vkCmdCopyBuffer(cmd, source_buffer, dst_buffer, 1u, &region);
    return e->vk.vkEndCommandBuffer(cmd) == 0;
}

static void* moe_vulkan_create_fence(moe_pc_engine* e) {
    if (!moe_vulkan_transfer_runtime_ready(e)) {
        return nullptr;
    }
    moe_vk_fence_create_info info{moe_VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0u};
    void* fence = nullptr;
    return e->vk.vkCreateFence(e->vk.device, &info, nullptr, &fence) == 0 ? fence : nullptr;
}

static void moe_vulkan_destroy_fence(moe_pc_engine* e, void* fence) {
    if (e && fence && e->vk.vkDestroyFence && e->vk.device) {
        e->vk.vkDestroyFence(e->vk.device, fence, nullptr);
    }
}

static int moe_vulkan_reset_fence(moe_pc_engine* e, void* fence) {
    if (!e || !fence || !e->vk.vkResetFences || !e->vk.device) {
        return 0;
    }
    const void* fence_handle = fence;
    return e->vk.vkResetFences(e->vk.device, 1u, &fence_handle) == 0;
}

static int moe_vulkan_fence_signaled(moe_pc_engine* e, void* fence) {
    return e && fence && e->vk.vkGetFenceStatus &&
        e->vk.vkGetFenceStatus(e->vk.device, fence) == 0;
}

static int moe_vulkan_wait_fence(moe_pc_engine* e, void* fence) {
    if (!e || !fence || !e->vk.vkWaitForFences || !e->vk.device) {
        return 0;
    }
    const void* fence_handle = fence;
    return e->vk.vkWaitForFences(e->vk.device, 1u, &fence_handle, 1u, UINT64_MAX) == 0;
}

static int moe_vulkan_submit_command_buffer(moe_pc_engine* e, void* cmd, void* fence) {
    if (!moe_vulkan_transfer_runtime_ready(e) || !cmd) {
        return 0;
    }
    const void* command_buffer = cmd;
    moe_vk_submit_info submit_info{
        moe_VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0u,
        nullptr,
        nullptr,
        1u,
        &command_buffer,
        0u,
        nullptr
    };
    return e->vk.vkQueueSubmit(e->vk.queue, 1u, &submit_info, fence) == 0;
}
