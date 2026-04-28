void glm5_scale4_close(glm5_scale4_file_t* file) {
    if (!file) {
        return;
    }
    munmap_file(&file->mmap);
    free(file);
}

uint32_t glm5_scale4_entry_count(const glm5_scale4_file_t* file) {
    if (!file || !file->header) {
        return 0;
    }
    return file->header->entry_count;
}

const glm5_scale4_entry_t* glm5_scale4_entry_at(const glm5_scale4_file_t* file, uint32_t index) {
    if (!file || !file->entries || !file->header || index >= file->header->entry_count) {
        return NULL;
    }
    return &file->entries[index];
}

const char* glm5_scale4_entry_key(const glm5_scale4_file_t* file, const glm5_scale4_entry_t* entry) {
    if (!file || !entry) {
        return NULL;
    }
    if (!glm5_scale4_bounds_ok(file, entry->key_offset, (uint64_t)entry->key_length + 1u)) {
        return NULL;
    }
    return (const char*)(file->data + entry->key_offset);
}
