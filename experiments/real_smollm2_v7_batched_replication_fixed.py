from pathlib import Path

source_path = Path(__file__).with_name("real_smollm2_v7_batched_replication.py")
source = source_path.read_text(encoding="utf-8")
source = source.replace(
    'ROOT = Path("out/real_smollm2_v7_batched_replication")',
    'ROOT = Path("out/real_smollm2_v7_batched_replication_fixed")',
)
source = source.replace('[4400:5200]', '[2200:2450]')
if '[4400:5200]' in source:
    raise RuntimeError("WikiText range replacement failed")
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace)
