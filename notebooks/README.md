# Generic GGUF Offload Colab

Use `GGUF_Offload_Metadata_Patch_Stream.ipynb` to patch any sharded GGUF folder
with StorageLLM `offload.*` metadata.

Only edit the `CHANGE HERE` block in the first code cell:

```python
SOURCE_HF_REPO_ID = "unsloth/GLM-5.1-GGUF"
SOURCE_SUBDIR = "UD-IQ2_M"
SOURCE_FILE_PREFIX = ""        # blank means auto-discover
SOURCE_FILE_COUNT_TEXT = ""    # blank means auto-discover
TARGET_HF_OWNER = "storagejuju"
TARGET_HF_REPO_ID = ""         # blank means derive from source prefix
```

The notebook discovers `.gguf` shard names from the Hugging Face folder, patches
one shard at a time, uploads it, deletes local output, and then continues. This
keeps Colab disk use bounded for large GGUF variants.

The generated GGUF metadata includes both QKV cache settings and source weight
quantization labels such as `gguf_iq2_m` or `gguf_mxfp4`, so the runtime can
choose a quant-family-specific backend instead of guessing from the bit count.

Local scratch notebooks under `colab/` remain ignored and should not be pushed.
