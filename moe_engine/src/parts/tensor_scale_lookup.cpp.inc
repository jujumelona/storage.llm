static float moe_raw_scale_at(
    const moe_tensor_record& rec,
    const mmap_context_t* shard,
    uint32_t row,
    uint32_t group
) {
    if (!shard || !mmap_get_ptr(shard, 0) || rec.info.scale_byte_length == 0 || rec.info.groups == 0) {
        return 1.0f;
    }
    // BUGFIX 336: Check row * groups doesn't overflow before adding group ★
    // Problem: row * rec.info.groups can overflow uint64_t
    // Solution: Check overflow before multiplication
    if (row > UINT64_MAX / rec.info.groups) {
        return 1.0f;  // Overflow would occur
    }
    const uint64_t idx = (uint64_t)row * rec.info.groups + group;
    // BUGFIX 337: Check rec.info.rows * rec.info.groups doesn't overflow ★
    // Problem: rows * groups can overflow uint64_t
    // Solution: Check overflow before multiplication
    if (rec.info.rows > UINT64_MAX / rec.info.groups) {
        return 1.0f;  // Overflow would occur
    }
    const uint64_t count = (uint64_t)rec.info.rows * rec.info.groups;
    const uint64_t shard_size = (uint64_t)mmap_get_size(shard);
    if (idx >= count || rec.info.scale_byte_offset > shard_size ||
        rec.info.scale_byte_length > shard_size - rec.info.scale_byte_offset) {
        return 1.0f;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mmap_get_ptr(shard, static_cast<size_t>(rec.info.scale_byte_offset)));
    if (!base) {
        return 1.0f;
    }
    if (rec.scale_format == 1) {
        // BUGFIX 338: Check idx * 4 doesn't overflow before accessing ★
        // Problem: idx * 4 can overflow uint64_t
        // Solution: Check overflow before multiplication
        if (idx > UINT64_MAX / 4ull) {
            return 1.0f;
        }
        float v;
        std::memcpy(&v, base + idx * 4ull, sizeof(v));
        return v;
    }
    if (rec.scale_format == 2) {
        // BUGFIX 339: Check idx * 2 doesn't overflow before accessing ★
        // Problem: idx * 2 can overflow uint64_t
        // Solution: Check overflow before multiplication
        if (idx > UINT64_MAX / 2ull) {
            return 1.0f;
        }
        uint16_t v;
        std::memcpy(&v, base + idx * 2ull, sizeof(v));
        return moe_raw_fp16_to_f32(v);
    }
    if (rec.scale_format == 3) {
        // BUGFIX 340: Check idx fits in size_t before array access ★
        // Problem: idx can exceed SIZE_MAX on 32-bit systems
        // Solution: Check idx <= SIZE_MAX before accessing
        if (idx > SIZE_MAX) {
            return 1.0f;
        }
        return storage_f8_e4m3_to_f32(base[idx]);
    }
    return 1.0f;
}
