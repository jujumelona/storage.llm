#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct moe_pc_engine moe_pc_engine_t;

#include "parts/moe_enums.h.inc"
#include "parts/moe_config_types.h.inc"
#include "parts/moe_stats_types.h.inc"
#include "parts/moe_backend_types.h.inc"
#include "parts/moe_model_types.h.inc"
#include "parts/moe_hardcode_types.h.inc"
#include "parts/moe_tensor_types.h.inc"
#include "parts/moe_lifecycle_api.h.inc"
#include "parts/moe_residency_api.h.inc"
#include "parts/moe_io_api.h.inc"
#include "parts/moe_juju_api.h.inc"
#include "parts/moe_topology_api.h.inc"
#include "parts/moe_codec_api.h.inc"
#include "parts/moe_names_api.h.inc"
#include "parts/moe_hardcode_api.h.inc"
#include "parts/moe_forward_api.h.inc"

#ifdef __cplusplus
}
#endif
