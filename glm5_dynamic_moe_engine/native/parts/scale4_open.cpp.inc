glm5_scale4_file_t* glm5_scale4_open(const char* path) {
    if (!path) {
        return NULL;
    }

    glm5_scale4_file_t* file = (glm5_scale4_file_t*)calloc(1, sizeof(glm5_scale4_file_t));
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
    if (!glm5_scale4_bounds_ok(file, 0, sizeof(glm5_scale4_header_t))) {
        glm5_scale4_close(file);
        return NULL;
    }

    file->header = (const glm5_scale4_header_t*)file->data;
    if (memcmp(file->header->magic, GLM5_SCALE4_MAGIC, 8) != 0 ||
        file->header->version != GLM5_SCALE4_VERSION) {
        glm5_scale4_close(file);
        return NULL;
    }

    const uint64_t table_bytes =
        (uint64_t)file->header->entry_count * (uint64_t)sizeof(glm5_scale4_entry_t);
    if (!glm5_scale4_bounds_ok(file, file->header->entry_table_offset, table_bytes)) {
        glm5_scale4_close(file);
        return NULL;
    }
    file->entries = (const glm5_scale4_entry_t*)(file->data + file->header->entry_table_offset);

    return file;
}
