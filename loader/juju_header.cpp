#include "juju_header.h"

#include <cstring>

namespace storagellm {

static uint32_t juju_u32(const uint8_t* p) {
    uint32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

static uint64_t juju_u64(const uint8_t* p) {
    uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

bool juju_header_valid(const JujuHeader& h, size_t mapped_size) {
    if (std::memcmp(h.magic, "G51STOR1", 8) != 0) return false;
    if (h.header_bytes < 64 || h.data_offset < h.header_bytes) return false;
    if (h.file_size != 0 && h.file_size != mapped_size) return false;
    if (h.json_offset > mapped_size || h.json_bytes > mapped_size) return false;
    return h.json_offset + h.json_bytes <= mapped_size;
}

bool juju_parse_header(const void* data, size_t size, JujuHeader* out) {
    if (!data || !out || size < 64) return false;
    const auto* p = static_cast<const uint8_t*>(data);
    std::memcpy(out->magic, p, 8);
    out->version = juju_u32(p + 8);
    out->flags = juju_u32(p + 12);
    out->header_bytes = juju_u64(p + 16);
    out->data_offset = juju_u64(p + 24);
    out->json_offset = juju_u64(p + 32);
    out->json_bytes = juju_u64(p + 40);
    out->file_size = juju_u64(p + 48);
    out->block_count = juju_u64(p + 56);
    return juju_header_valid(*out, size);
}

}  // namespace storagellm
