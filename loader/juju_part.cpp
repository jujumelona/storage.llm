#include "juju_part.h"

namespace storagellm {

JujuPart::~JujuPart() {
    close();
}

bool JujuPart::open(const char* path) {
    close();
    if (!path || !mmap_file(&map_, path)) {
        return false;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mmap_get_ptr(&map_, 0));
    const size_t size = mmap_get_size(&map_);
    if (!juju_parse_header(base, size, &header_)) {
        munmap_file(&map_);
        return false;
    }
    path_ = path;
    mapped_ = true;
    return true;
}

void JujuPart::close() {
    if (mapped_) {
        munmap_file(&map_);
    }
    mapped_ = false;
    path_.clear();
    header_ = JujuHeader{};
}

bool JujuPart::slice(uint64_t offset, uint64_t length, const uint8_t** out) const {
    const size_t size = mmap_get_size(&map_);
    if (!mapped_ || !out || offset > size || length > size - offset) {
        return false;
    }
    *out = static_cast<const uint8_t*>(mmap_get_ptr(&map_, static_cast<size_t>(offset)));
    return true;
}

void JujuPart::prefetch(uint64_t offset, uint64_t length) {
    if (mapped_ && offset < mmap_get_size(&map_)) {
        mmap_prefetch(&map_, static_cast<size_t>(offset), static_cast<size_t>(length));
    }
}

}  // namespace storagellm
