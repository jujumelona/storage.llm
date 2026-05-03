# JUJU Offload Colab

Use `GGUF_Offload_Metadata_Patch_Stream.ipynb` as the single generic Colab
notebook. Model-specific source URLs and Hub targets should be supplied through
the notebook configuration or environment variables, not committed as separate
model notebooks.

The model build path is GGUF source -> JUJU shard artifact. Colab disk use stays
bounded by processing one source shard at a time, uploading
`<original-shard-stem>.juju` plus `<original-shard-stem>.juju.idx`, then deleting
the local output before moving to the next shard. Do not upload patched GGUF as
the final runtime model.

The generated metadata treats GGUF as the conversion source and JUJU as the
target custom MoE engine format. The physical materializer rewrites tensor
payloads into 4KB-aligned JUJU sections so Direct I/O/GDS eligibility and
hot/warm/cold physical placement can become real runtime behavior rather than
metadata-only hints. The JUJU MODEL_META contract includes QKV packed-cache settings, source
weight quantization labels, active-set residency, batch-aware prefetch and
eviction policy, I/O scheduling, runtime telemetry hooks, and exactly 5000
optimization catalog items.

The same contract also exports direct engine-readable keys for layout and
runtime planning, including `juju_container_contract`,
`final_model_structure_contract`, `routing_optimization_contract`,
`code_generation_moe_contract`, `structure_finalization_contract`,
`universal_tier_contract`, `pipeline_budget_contract`,
`execution_path_contract`, `expert_segmentation_contract`,
`chunk_io_contract`, `qkv_cache_contract`, `format_layout_contract`,
`engine_pipeline_contract`, `expert_scheduler_contract`, and
`runtime_update_files`.

`final_model_structure_contract` locks the JUJU runtime structure as
`JUJU_MAX_MOE_STRUCTURE_V1`: shared core, routing brain, predictor bank, expert
groups, expert segments, buddy fallback, required QKV subsystem, coding locality
cache, and tier scheduler. Further work should tune policies, kernels, predictor
weights, and mutable index statistics instead of changing the top-level model
structure.
