// Device tensor copy - Main entry point
// Modularized: Sections extracted to separate files

static void device_complete_pending_copies(moe_pc_engine* e);
static void device_complete_pending_copy_key(moe_pc_engine* e, uint64_t key);

#include "device_backend_async.cpp.inc"
#include "device_tensor_copy_stream.cpp.inc"
#include "device_tensor_copy_tracking.cpp.inc"
#include "device_tensor_copy_alloc.cpp.inc"
#include "device_tensor_copy_ops.cpp.inc"
