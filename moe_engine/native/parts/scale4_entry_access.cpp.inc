void moe_scale4_close(moe_scale4_file_t* file) {
    if (!file) {
        return;
    }
    delete file->key_map;
    file->key_map = nullptr;
    munmap_file(&file->mmap);
    free(file);
}

uint32_t moe_scale4_entry_count(const moe_scale4_file_t* file) {
    if (!file || !file->header) {
        return 0;
    }
    return file->header->entry_count;
}

const moe_scale4_entry_t* moe_scale4_entry_at(const moe_scale4_file_t* file, uint32_t index) {
    if (!file || !file->entries || !file->header || index >= file->header->entry_count) {
        return NULL;
    }
    return &file->entries[index];
}

const char* moe_scale4_entry_key(const moe_scale4_file_t* file, const moe_scale4_entry_t* entry) {
    if (!file || !entry) {
        return NULL;
    }
    if (!moe_scale4_bounds_ok(file, entry->key_offset, (uint64_t)entry->key_length + 1u)) {
        return NULL;
    }
    return (const char*)(file->data + entry->key_offset);
}
