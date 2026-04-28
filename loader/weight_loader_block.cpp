#include "weight_loader.h"

namespace storagellm {

bool WeightLoader::fill_block(uint32_t block_id, StorageBlockSlice* out) {
    if (!out) {
        return false;
    }
    *out = StorageBlockSlice{};
    out->block_id = block_id;
    if (block_id == UINT32_MAX) {
        return true;
    }
    JujuFooterBlock meta{};
    if (!footer_.find_block(block_id, &meta)) {
        return false;
    }
    // Bug 4: Prefetch BEFORE slice to allow OS to start loading pages
    part_.prefetch(meta.offset, meta.length);
    const uint8_t* data = nullptr;
    if (!part_.slice(meta.offset, meta.length, &data)) {
        return false;
    }
    out->data = data;
    out->bytes = meta.length;
    out->kind = meta.kind;
    out->key = meta.key;
    return true;
}

}  // namespace storagellm
