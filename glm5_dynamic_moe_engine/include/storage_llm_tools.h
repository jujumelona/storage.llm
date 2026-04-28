#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Fix 4: Include all storage API headers
#include "tools/storage_f8_api.h.inc"
#if defined(__has_include)
#if __has_include("tools/storage_cache_api.h.inc")
#include "tools/storage_cache_api.h.inc"
#endif
#if __has_include("tools/storage_fp4_api.h.inc")
#include "tools/storage_fp4_api.h.inc"
#endif
#if __has_include("tools/storage_juju_api.h.inc")
#include "tools/storage_juju_api.h.inc"
#endif
#if __has_include("tools/storage_manifest_api.h.inc")
#include "tools/storage_manifest_api.h.inc"
#endif
#if __has_include("tools/storage_sha1_api.h.inc")
#include "tools/storage_sha1_api.h.inc"
#endif
#endif

#ifdef __cplusplus
}
#endif
