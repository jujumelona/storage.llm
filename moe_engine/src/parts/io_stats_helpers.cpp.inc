static void reset_io_stats_for_config(moe_pc_engine* engine) {
    if (!engine) {
        return;
    }
    engine->io_atomic.queued_requests.store(0);
    engine->io_atomic.completed_requests.store(0);
    engine->io_atomic.failed_requests.store(0);
    engine->io_atomic.dropped_requests.store(0);
    engine->io_atomic.bytes_prefetched.store(0);
    engine->io_atomic.bytes_disk_to_ram.store(0);
    engine->io_atomic.bytes_disk_hint_issued.store(0);
    engine->io_atomic.bytes_ram_to_pinned.store(0);
    engine->io_atomic.bytes_pinned_to_gpu.store(0);
    engine->io_atomic.bytes_direct_to_gpu.store(0);
    engine->io_atomic.bytes_touch_fallback.store(0);
    engine->io_atomic.bytes_sync_h2d_fallback.store(0);
    engine->io_atomic.device_alloc_failures.store(0);
    engine->io_atomic.device_copy_failures.store(0);
    engine->io_atomic.disk_stage_done.store(0);
    engine->io_atomic.pinned_stage_done.store(0);
    engine->io_atomic.gpu_stage_done.store(0);
    engine->io_atomic.direct_stage_done.store(0);
    engine->io_atomic.staging_wraps.store(0);
    engine->io_atomic.staging_slot_waits.store(0);
    engine->io_atomic.staging_slot_contention.store(0);
    engine->io_atomic.topology_predictions.store(0);
    engine->io_atomic.topology_updates.store(0);
    const moe_io_path_t active_path = choose_active_io_path(engine);
    engine->io_atomic.active_path.store(static_cast<uint32_t>(active_path));
    engine->io_atomic.pinned_host_active.store(engine->staging_pinned ? 1u : 0u);
    engine->io_atomic.directstorage_active.store(active_path == moe_IO_PATH_DIRECTSTORAGE ? 1u : 0u);
    engine->io_atomic.gpudirect_storage_active.store(active_path == moe_IO_PATH_GPUDIRECT_STORAGE ? 1u : 0u);
    engine->io_atomic.bam_nvme_active.store(active_path == moe_IO_PATH_BAM_NVME ? 1u : 0u);
}

static void snapshot_io_stats(const moe_pc_engine* engine, moe_io_stats_t* out) {
    if (!engine || !out) {
        return;
    }
    std::memset(out, 0, sizeof(*out));
    out->queued_requests = engine->io_atomic.queued_requests.load();
    out->completed_requests = engine->io_atomic.completed_requests.load();
    out->failed_requests = engine->io_atomic.failed_requests.load();
    out->dropped_requests = engine->io_atomic.dropped_requests.load();
    out->bytes_prefetched = engine->io_atomic.bytes_prefetched.load();
    out->bytes_disk_to_ram = engine->io_atomic.bytes_disk_to_ram.load();
    out->bytes_disk_hint_issued = engine->io_atomic.bytes_disk_hint_issued.load();
    out->bytes_ram_to_pinned = engine->io_atomic.bytes_ram_to_pinned.load();
    out->bytes_pinned_to_gpu = engine->io_atomic.bytes_pinned_to_gpu.load();
    out->bytes_direct_to_gpu = engine->io_atomic.bytes_direct_to_gpu.load();
    out->bytes_touch_fallback = engine->io_atomic.bytes_touch_fallback.load();
    out->bytes_sync_h2d_fallback = engine->io_atomic.bytes_sync_h2d_fallback.load();
    out->device_alloc_failures = engine->io_atomic.device_alloc_failures.load();
    out->device_copy_failures = engine->io_atomic.device_copy_failures.load();
    out->disk_stage_done = engine->io_atomic.disk_stage_done.load();
    out->pinned_stage_done = engine->io_atomic.pinned_stage_done.load();
    out->gpu_stage_done = engine->io_atomic.gpu_stage_done.load();
    out->direct_stage_done = engine->io_atomic.direct_stage_done.load();
    out->staging_wraps = engine->io_atomic.staging_wraps.load();
    out->staging_slot_deficit = engine->io_atomic.staging_slot_deficit.load();
    out->staging_slot_waits = engine->io_atomic.staging_slot_waits.load();
    out->staging_slot_contention = engine->io_atomic.staging_slot_contention.load();
    out->recommended_staging_bytes = moe_recommended_staging_bytes(engine);
    out->topology_predictions = engine->io_atomic.topology_predictions.load();
    out->topology_updates = engine->io_atomic.topology_updates.load();
    out->disk_queue_depth = engine->disk_stage.depth.load();
    out->pinned_queue_depth = engine->pinned_stage.depth.load();
    out->gpu_queue_depth = engine->gpu_stage.depth.load();
    out->active_workers = engine->disk_stage.inflight.load() +
        engine->pinned_stage.inflight.load() + engine->gpu_stage.inflight.load();
    out->active_path = static_cast<moe_io_path_t>(engine->io_atomic.active_path.load());
    out->pinned_host_active = static_cast<int>(engine->io_atomic.pinned_host_active.load());
    out->directstorage_active = static_cast<int>(engine->io_atomic.directstorage_active.load());
    out->gpudirect_storage_active = static_cast<int>(engine->io_atomic.gpudirect_storage_active.load());
    out->bam_nvme_active = static_cast<int>(engine->io_atomic.bam_nvme_active.load());
}
