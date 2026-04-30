// Device tensor copy tracking and completion

static int device_mark_copy_pending_locked(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return 0;
    }
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end() || it->second.pending_free) {
        return 0;
    }
    ++it->second.inflight_copies;
    return 1;
}

static void device_complete_copy_locked(
    glm5_pc_engine* e,
    uint64_t key,
    std::vector<std::pair<CUdeviceptr, uint64_t>>* deferred_free_ptrs = nullptr
) {
    if (!e) {
        return;
    }
    auto it = e->device_allocations.find(key);
    if (it == e->device_allocations.end()) {
        return;
    }
    if (it->second.inflight_copies > 0) {
        --it->second.inflight_copies;
    }
    if (it->second.inflight_copies == 0 && it->second.pending_free) {
        device_release_allocation_locked(e, it, deferred_free_ptrs);
    }
}

static void device_complete_pending_copies(glm5_pc_engine* e) {
    if (!e) return;
    auto it = s_tl_tensor_pending_keys.find(e);
    if (it == s_tl_tensor_pending_keys.end() || it->second.empty()) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (uint64_t key : it->second) {
            device_complete_copy_locked(e, key, &deferred_free_ptrs);
        }
    }
    it->second.clear();
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_complete_pending_copy_key(glm5_pc_engine* e, uint64_t key) {
    if (!e || key == 0) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_complete_copy_locked(e, key, &deferred_free_ptrs);
    }
    auto it = s_tl_tensor_pending_keys.find(e);
    if (it != s_tl_tensor_pending_keys.end() && !it->second.empty()) {
        auto key_it = std::find(it->second.begin(), it->second.end(), key);
        if (key_it != it->second.end()) {
            *key_it = it->second.back();
            it->second.pop_back();
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_force_clear_pending_copies(glm5_pc_engine* e) {
    if (!e) return;
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        for (auto it = e->device_allocations.begin(); it != e->device_allocations.end(); ) {
            it->second.inflight_copies = 0;
            if (it->second.pending_free) {
                auto victim = it++;
                device_release_allocation_locked(e, victim, &deferred_free_ptrs);
            } else {
                ++it;
            }
        }
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_cancel_pending_copy(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    {
            std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
        {
            std::lock_guard<std::mutex> lock(e->device_mutex);
            device_complete_copy_locked(e, key, &deferred_free_ptrs);
        }
        device_free_deferred_ptrs(e, deferred_free_ptrs);
    }
}

static void device_free_key(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    std::vector<std::pair<CUdeviceptr, uint64_t>> deferred_free_ptrs;
    {
        std::lock_guard<std::mutex> lock(e->device_mutex);
        device_free_locked(e, key, &deferred_free_ptrs);
    }
    device_free_deferred_ptrs(e, deferred_free_ptrs);
}

static void device_touch_allocation(glm5_pc_engine* e, uint64_t key) {
    if (!e) {
        return;
    }
    std::lock_guard<std::mutex> lock(e->device_mutex);
    auto it = e->device_allocations.find(key);
    if (it != e->device_allocations.end()) {
        it->second.last_touch_epoch = e->expert_touch_epoch.fetch_add(1) + 1;
    }
}
