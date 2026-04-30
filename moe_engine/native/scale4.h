#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define moe_SCALE4_MAGIC "G5S4IDX1"
#define moe_SCALE4_VERSION 1u

#pragma pack(push, 1)

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t entry_count;
    uint64_t entry_table_offset;
    uint64_t string_table_offset;
    uint64_t data_offset;
    uint64_t file_size;
    uint32_t flags;
    uint32_t reserved0;
} moe_scale4_header_t;

typedef struct {
    uint64_t key_offset;
    uint32_t key_length;
    uint32_t rows;
    uint32_t groups;
    uint32_t group_size;
    uint32_t bits;
    uint32_t centroids;
    float scale2;
    float max_abs_error;
    uint64_t codebook_offset;
    uint64_t index_offset;
    uint64_t index_bytes;
} moe_scale4_entry_t;

#pragma pack(pop)

typedef struct moe_scale4_file_t moe_scale4_file_t;

moe_scale4_file_t* moe_scale4_open(const char* path);
void moe_scale4_close(moe_scale4_file_t* file);

uint32_t moe_scale4_entry_count(const moe_scale4_file_t* file);
const moe_scale4_entry_t* moe_scale4_entry_at(const moe_scale4_file_t* file, uint32_t index);
const moe_scale4_entry_t* moe_scale4_find(const moe_scale4_file_t* file, const char* key);

const char* moe_scale4_entry_key(const moe_scale4_file_t* file, const moe_scale4_entry_t* entry);
float moe_scale4_get_scale(const moe_scale4_file_t* file, const moe_scale4_entry_t* entry, uint32_t row, uint32_t group);
int moe_scale4_decode_row(
    const moe_scale4_file_t* file,
    const moe_scale4_entry_t* entry,
    uint32_t row,
    float* out_scales,
    uint32_t out_count
);

#ifdef __cplusplus
}
#endif

