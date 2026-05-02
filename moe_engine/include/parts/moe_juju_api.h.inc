#define MOE_JUJU_HEADER_BYTES 4096u
#define MOE_JUJU_SECTION_ENTRY_WIRE_BYTES 96u
#define MOE_JUJU_MAX_TASK_TYPES 8u
#define MOE_JUJU_MAX_BUDDIES 8u
#define MOE_JUJU_MAX_PREFETCH_HINTS 16u
#define MOE_JUJU_MAX_EXPERT_SEGMENTS 8u

typedef enum {
    moe_JUJU_SECTION_MODEL_META = 0x0001u,
    moe_JUJU_SECTION_PREDICTOR = 0x0002u,
    moe_JUJU_SECTION_BUDDY_MAP = 0x0003u,
    moe_JUJU_SECTION_TIER_HINT = 0x0004u,
    moe_JUJU_SECTION_SHARED_WEIGHTS = 0x0010u,
    moe_JUJU_SECTION_HOT_EXPERTS = 0x0011u,
    moe_JUJU_SECTION_WARM_EXPERTS = 0x0012u,
    moe_JUJU_SECTION_COLD_EXPERTS = 0x0013u,
    moe_JUJU_SECTION_LAYER_ORDER_INDEX = 0x0020u,
    moe_JUJU_SECTION_QKV_POLICY = 0x0021u,
    moe_JUJU_SECTION_CUSTOM = 0xFF00u
} moe_juju_section_type_t;

typedef enum {
    moe_JUJU_TIER_ROLE_COMPUTE_MEM = 0,
    moe_JUJU_TIER_ROLE_FAST_MEM = 1,
    moe_JUJU_TIER_ROLE_SLOW_MEM = 2
} moe_juju_tier_role_t;

typedef enum {
    moe_JUJU_EXEC_FULL_COMPUTE_MEM = 0,
    moe_JUJU_EXEC_DIRECT_STORAGE_TO_COMPUTE = 1,
    moe_JUJU_EXEC_THREE_TIER_PIPELINE = 2,
    moe_JUJU_EXEC_ROTATING_BUFFER = 3,
    moe_JUJU_EXEC_CPU_WITH_CACHE = 4,
    moe_JUJU_EXEC_STORAGE_STREAMING = 5
} moe_juju_execution_path_t;

typedef enum {
    moe_JUJU_SEGMENT_POLICY_AUTO = 0,
    moe_JUJU_SEGMENT_POLICY_WHOLE_EXPERT = 1,
    moe_JUJU_SEGMENT_POLICY_IO_SEGMENTS = 2,
    moe_JUJU_SEGMENT_POLICY_TOP_BOTTOM = 3,
    moe_JUJU_SEGMENT_POLICY_ROW_COLUMN = 4
} moe_juju_segment_policy_t;

typedef struct {
    uint64_t capacity_bytes;
    double bandwidth_gbps;
    double latency_us;
    double compute_tflops;
    int supports_async;
    int supports_dma;
    int is_unified_memory;
} moe_juju_tier_capability_t;

typedef struct {
    uint32_t prefetch_depth;
    uint64_t tier0_expert_count;
    uint64_t tier1_expert_count;
    uint32_t io_threads;
    uint64_t block_size;
    uint64_t segment_bytes;
    uint32_t segment_count;
    double compute_ms;
    double tier1_to_tier0_ms;
    double tier2_to_tier1_ms;
} moe_juju_pipeline_config_t;

typedef struct {
    uint8_t magic[8];
    uint32_t version_major;
    uint32_t version_minor;
    uint64_t created_at;
    uint64_t file_size;
    uint64_t section_table_offset;
    uint64_t section_table_size;
    uint64_t shared_weights_size;
    uint64_t hot_section_size;
    uint64_t warm_section_size;
    uint64_t cold_section_size;
    uint64_t index_file_checksum;
    char model_name[64];
    uint32_t section_count;
    uint32_t header_size;
    uint32_t layer_count;
    uint32_t expert_count_per_layer;
    uint32_t top_k;
    uint32_t hidden_dim;
    uint32_t expert_dim;
    uint32_t quant_type;
    uint32_t arch_type;
    uint32_t segment_policy;
    uint32_t qkv_policy;
    uint32_t preferred_segment_bytes;
    uint32_t max_segments_per_expert;
    uint32_t recommended_vram_mb;
    uint32_t recommended_ram_mb;
    uint32_t reserved_u32;
    uint8_t file_hash[32];
    uint8_t header_hash[32];
    uint8_t reserved[3816];
} moe_juju_header_t;

typedef struct {
    uint32_t type;
    uint32_t flags;
    uint64_t offset;
    uint64_t size;
    uint64_t uncompressed_size;
    uint32_t sequential_block_size;
    uint32_t random_block_size;
    uint8_t compression;
    uint8_t prefetch_distance;
    uint8_t mmap_friendly;
    uint8_t reserved_u8;
    uint8_t checksum[16];
    char name[32];
} moe_juju_section_entry_t;

typedef struct {
    uint64_t offset;
    uint32_t size;
    uint8_t priority;
    uint8_t can_partial_exec;
    uint8_t segment_kind;
    uint8_t reserved_u8;
} moe_juju_expert_segment_t;

typedef struct {
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t total_size;
    uint32_t segment_count;
    uint32_t flags;
    float access_frequency;
    float task_affinity[MOE_JUJU_MAX_TASK_TYPES];
    uint16_t buddy_expert_ids[MOE_JUJU_MAX_BUDDIES];
    uint16_t prefetch_hints[MOE_JUJU_MAX_PREFETCH_HINTS];
    uint32_t optimal_read_size;
    uint32_t tier_hint;
    float partial_accuracy;
    moe_juju_expert_segment_t segments[MOE_JUJU_MAX_EXPERT_SEGMENTS];
    uint8_t chunk_hash[8];
    uint8_t reserved[28];
} moe_juju_expert_chunk_header_t;

typedef struct {
    uint8_t magic[8];
    uint32_t version;
    uint32_t header_size;
    uint64_t model_hash[4];
    uint64_t last_updated;
    uint64_t total_tokens_processed;
    uint32_t record_count;
    uint32_t trajectory_count;
} moe_juju_idx_header_t;

typedef struct {
    uint32_t layer_id;
    uint32_t expert_id;
    uint64_t access_count;
    uint64_t last_access_token;
    float recent_frequency;
    float task_frequency[MOE_JUJU_MAX_TASK_TYPES];
    uint8_t current_tier;
    uint8_t tier_locked;
    uint8_t reserved_u8[2];
    uint32_t prefetch_hits;
    uint32_t prefetch_misses;
    uint64_t physical_offset;
    uint64_t physical_size;
} moe_juju_expert_record_t;

typedef struct {
    uint64_t pattern_hash;
    uint16_t predicted_experts[MOE_JUJU_MAX_PREFETCH_HINTS];
    float confidence;
    uint32_t observation_count;
} moe_juju_trajectory_entry_t;

int moe_juju_header_valid(const moe_juju_header_t* header);

int moe_juju_read_header(const char* path, moe_juju_header_t* out_header);

int moe_juju_read_section_table(
    const char* path,
    const moe_juju_header_t* header,
    moe_juju_section_entry_t* out_entries,
    uint32_t max_entries,
    uint32_t* out_count
);

uint64_t moe_juju_align_up(uint64_t value, uint64_t alignment);

int moe_juju_compute_pipeline_config(
    const moe_juju_tier_capability_t* tier0,
    const moe_juju_tier_capability_t* tier1,
    const moe_juju_tier_capability_t* tier2,
    uint64_t expert_bytes,
    double expert_flops,
    moe_juju_pipeline_config_t* out_config
);

moe_juju_execution_path_t moe_juju_select_execution_path(
    int has_accelerator,
    uint64_t model_total_bytes,
    uint64_t expert_bytes,
    const moe_juju_tier_capability_t* tier0,
    const moe_juju_tier_capability_t* tier1,
    const moe_juju_tier_capability_t* tier2
);

const char* moe_juju_execution_path_name(moe_juju_execution_path_t path);

const char* moe_juju_section_type_name(uint32_t type);

#ifdef __cplusplus
static_assert(sizeof(moe_juju_header_t) == MOE_JUJU_HEADER_BYTES, "JUJU header must stay 4096 bytes");
static_assert(sizeof(moe_juju_section_entry_t) == MOE_JUJU_SECTION_ENTRY_WIRE_BYTES, "JUJU section entry must stay 96 bytes");
#endif
