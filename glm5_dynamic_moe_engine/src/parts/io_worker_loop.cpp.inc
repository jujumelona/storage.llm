static int prefetch_one_raw_span(glm5_pc_engine* e, const glm5_storage_raw_span_spec_t& s, uint8_t pool);

static void io_worker_loop(glm5_pc_engine* engine, uint32_t worker_index) {
    if (!engine) {
        return;
    }
    glm5_io_path_t active_path = choose_active_io_path(engine);
    engine->io_atomic.active_path.store(static_cast<uint32_t>(active_path));
    uint32_t refresh_turn = 0;
    uint64_t last_vram_free = engine->device_free_bytes.load(std::memory_order_relaxed);
    for (;;) {
        // Fix 10: GDS/mmap routing changes rarely — refresh every 10 tasks.
        // VRAM pressure changes fast — force refresh when free bytes shift >10%.
        const uint64_t cur_vram_free = engine->device_free_bytes.load(std::memory_order_relaxed);
        const bool vram_shifted = cur_vram_free > 0 &&
            (cur_vram_free > last_vram_free * 11 / 10 || cur_vram_free < last_vram_free * 9 / 10);
        if (vram_shifted || (refresh_turn++ % 10u) == 0u) {
            active_path = static_cast<glm5_io_path_t>(
                engine->io_atomic.active_path.load(std::memory_order_relaxed));
            last_vram_free = cur_vram_free;
        }
        glm5_prefetch_task task{};
        if (!pop_stage_task(engine, engine->disk_stage, &task, &engine->disk_worker_target, worker_index)) {
            return;
        }
        int ok = 0;
        int terminal = 1;
        if (task.raw_span_index >= 0) {
            const glm5_storage_raw_span_spec_t* span = glm5_storage_raw_span_at((uint32_t)task.raw_span_index);
            ok = span ? prefetch_one_raw_span(engine, *span, task.raw_pool) : 0;
        } else if (task.tensor_index < engine->tensors.size()) {
            const glm5_tensor_record& rec = engine->tensors[task.tensor_index];
            if (should_use_direct_device_path(engine, rec, task.target_tier, active_path)) {
                ok = prefetch_tensor_direct_to_gpu(engine, rec);
            } else {
                ok = prefetch_tensor_disk_to_ram(engine, rec);
                if (ok && engine->io_config.enable_multistage_pipeline && engine->staging) {
                    glm5_prefetch_task next = task;
                    next.stage = GLM5_TRANSFER_STAGE_RAM_TO_PINNED;
                    if (enqueue_stage_or_retry(engine, engine->pinned_stage, next)) {
                        terminal = 0;
                        engine->pinned_stage.cv.notify_one();
                    } else {
                        ok = task.target_tier == GLM5_TIER_VRAM ? 0 : ok;
                    }
                } else if (ok && task.target_tier == GLM5_TIER_VRAM) {
                    ok = prefetch_tensor_pinned_to_gpu(engine, rec, UINT32_MAX, 0, 1, 0);
                    if (ok == GLM5_DEVICE_COPY_BACKPRESSURE) {
                        // Bug #4: GPU full, re-enqueue to gpu_stage not disk_stage
                        task.urgent = 1;
                        task.stage = GLM5_TRANSFER_STAGE_PINNED_TO_GPU;
                        ok = enqueue_stage_or_retry(engine, engine->gpu_stage, task) ? 1 : 0;
                        if (ok) {
                            terminal = 0;
                            engine->gpu_stage.cv.notify_one();
                        }
                    }
                }
            }
        }

        finish_stage_task(engine, engine->disk_stage, ok, terminal);
        if (worker_index >= engine->disk_worker_target.load()) {
            return;
        }
    }
}

