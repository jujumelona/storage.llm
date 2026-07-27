from __future__ import annotations
import importlib.util
from pathlib import Path
from datasets import load_dataset

HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('v14',HERE/'real_smollm2_v14_fisher_chebyshev.py')
v14=importlib.util.module_from_spec(sp); sp.loader.exec_module(v14)
v14.ROOT=Path('out/real_smollm2_v15_fisher_chebyshev_holdout'); v14.ROOT.mkdir(parents=True,exist_ok=True)
core=v14.v13
core.N_TARGET=128
core.N_MCQ=192
core.WIKI_BLOCKS=32
core.BOOTSTRAPS=5000
original_sample=core.deterministic_sample
seed_map={1301:5301,1302:5302,1303:5303,1304:5304,1305:5305}
core.deterministic_sample=lambda rows,n,seed: original_sample(rows,n,seed_map.get(seed,seed))
def holdout_wikitext():
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    texts=[str(row['text']) for row in ds if str(row['text']).strip()]
    return texts[1400:4200]
core.build_wikitext=holdout_wikitext
v14.main()
