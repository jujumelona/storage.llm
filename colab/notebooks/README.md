# JUJU Offload Colab

Use `GGUF_Offload_Metadata_Patch_Stream.ipynb` as the metadata/contract
template, and use `colab/juju_shard_materializer.py` for the physical JUJU shard
rewrite step.

The numbered notebooks keep model-specific source URLs inside the notebook only.
Their file names stay generic:

- `GGUF_Offload_Patch_01.ipynb`
- `GGUF_Offload_Patch_02.ipynb`
- `GGUF_Offload_Patch_03.ipynb`

The model build path is GGUF source -> JUJU shard artifact. Colab disk use stays
bounded by processing one source shard at a time, uploading
`<original-shard-stem>.juju` plus `<original-shard-stem>.juju.idx`, then deleting
the local output before moving to the next shard. Do not upload patched GGUF as
the final runtime model; patched GGUF is only a bridge/debug artifact.

The generated metadata treats GGUF as the conversion source and JUJU as the
target custom MoE engine format. The physical materializer rewrites tensor
payloads into 4KB-aligned JUJU sections so Direct I/O/GDS eligibility and
hot/warm/cold physical placement can become real runtime behavior rather than
metadata-only hints. The contract includes QKV packed-cache settings, source
weight quantization labels, active-set residency, batch-aware prefetch and
eviction policy, I/O scheduling, runtime telemetry hooks, and exactly 5000
optimization catalog items under `offload.optimization_catalog_v1` and
`offload.metadata_v2`.

The same metadata also exports direct engine-readable keys for layout and runtime
planning, including `offload.juju_container_contract`,
`offload.final_model_structure_contract`,
`offload.routing_optimization_contract`,
`offload.code_generation_moe_contract`,
`offload.structure_finalization_contract`,
`offload.universal_tier_contract`, `offload.pipeline_budget_contract`,
`offload.execution_path_contract`, `offload.expert_segmentation_contract`,
`offload.chunk_io_contract`, `offload.qkv_cache_contract`,
`offload.format_layout_contract`, `offload.engine_pipeline_contract`,
`offload.expert_scheduler_contract`, and `offload.runtime_update_files`.

`offload.final_model_structure_contract` locks the JUJU runtime structure as
`JUJU_MAX_MOE_STRUCTURE_V1`: shared core, routing brain, predictor bank, expert
groups, expert segments, buddy fallback, required QKV subsystem, coding locality
cache, and tier scheduler. Further work should tune policies, kernels, predictor
weights, and mutable index statistics instead of changing the top-level model
structure.
