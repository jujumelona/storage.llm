moe_scale4_file_t* moe_scale4_open(const char* path) {
    if (!path) {
        return NULL;
    }

    moe_scale4_file_t* file = (moe_scale4_file_t*)calloc(1, sizeof(moe_scale4_file_t));
    if (!file) {
        return NULL;
    }
    file->key_map = nullptr;
    if (!mmap_file(&file->mmap, path)) {
        free(file);
        return NULL;
    }

    file->data = (const uint8_t*)file->mmap.addr;
    file->size = file->mmap.size;
    if (!moe_scale4_bounds_ok(file, 0, sizeof(moe_scale4_header_t))) {
        moe_scale4_close(file);
        return NULL;
    }

    file->header = (const moe_scale4_header_t*)file->data;
    if (memcmp(file->header->magic, moe_SCALE4_MAGIC, 8) != 0 ||
        file->header->version != moe_SCALE4_VERSION) {
        moe_scale4_close(file);
        return NULL;
    }

    const uint64_t table_bytes =
        (uint64_t)file->header->entry_count * (uint64_t)sizeof(moe_scale4_entry_t);
    if (!moe_scale4_bounds_ok(file, file->header->entry_table_offset, table_bytes)) {
        moe_scale4_close(file);
        return NULL;
    }
    file->entries = (const moe_scale4_entry_t*)(file->data + file->header->entry_table_offset);
    file->key_map = new (std::nothrow) std::unordered_map<std::string, uint32_t>();
    if (file->key_map) {
        file->key_map->reserve(file->header->entry_count);
        for (uint32_t i = 0; i < file->header->entry_count; ++i) {
            const moe_scale4_entry_t* entry = &file->entries[i];
            uint32_t key_len = 0;
            memcpy(&key_len, &entry->key_length, sizeof(key_len));
            if (key_len == 0 ||
                !moe_scale4_bounds_ok(file, entry->key_offset, (uint64_t)key_len + 1u)) {
                continue;
            }
            const char* key = (const char*)(file->data + entry->key_offset);
            file->key_map->emplace(std::string(key, key_len), i);
            if (((i + 1u) % 10000u) == 0u) {
                fprintf(stderr, "[storagellm] scale4 keys indexed: %u/%u\n",
                    i + 1u,
                    file->header->entry_count);
                fflush(stderr);
            }
        }
        fprintf(stderr, "[storagellm] scale4 keys indexed: %u/%u\n",
            file->header->entry_count,
            file->header->entry_count);
        fflush(stderr);
    }

    return file;
}
