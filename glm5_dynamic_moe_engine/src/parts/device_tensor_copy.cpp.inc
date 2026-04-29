// Device tensor copy - Main entry point
// Modularized: Sections extracted to separate files

static void device_complete_pending_copies(glm5_pc_engine* e);

#include "device_tensor_copy_stream.cpp.inc"
#include "device_tensor_copy_tracking.cpp.inc"
#include "device_tensor_copy_alloc.cpp.inc"
#include "device_tensor_copy_ops.cpp.inc"
