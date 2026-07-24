#!/usr/bin/env python3
import json, sys
from huggingface_hub import hf_hub_download
sys.path.insert(0, 'experiments')
import cpoc_attributionbench as experiment

FILES = (
    'train_all_subset_balanced.jsonl',
    'test_all_subset_balanced.jsonl',
    'test_ood_all_subset_balanced.jsonl',
)

def load_rows():
    splits=[]
    for name in FILES:
        path=hf_hub_download('osunlp/AttributionBench',name,repo_type='dataset')
        with open(path,encoding='utf-8') as f:
            splits.append([json.loads(line) for line in f if line.strip()])
    return splits

experiment.load_rows=load_rows
experiment.main()
