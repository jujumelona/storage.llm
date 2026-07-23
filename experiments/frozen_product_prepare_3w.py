#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import frozen_product_codetrace_full as frozen


def main():
    from datasets import load_dataset
    import codetrace_full_parallel_prepare as prep

    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='full')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('NJU-LINK/CodeTraceBench', split=args.split)
    items = list(enumerate(ds))

    def one(item):
        return frozen.retry_process(prep, item, cache, attempts=12)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, result in enumerate(ex.map(one, items), 1):
            results.append(result)
            if n % 25 == 0:
                print(json.dumps({'processed': n}), flush=True)

    results.sort(key=lambda x: x[0])
    blinds, labels, stats = [], [], []
    for _, blind, label, stat in results:
        if blind:
            mapping = frozen.official_map(len(blind['events']), int(blind['expected_step_count']))
            for event, step in zip(blind['events'], mapping):
                event['official_step'] = step
            for event, step in zip(blind['events_ablation'], mapping):
                event['official_step'] = step
            blind['coordinate_alignment'] = 'monotone_rank_to_public_step_count'
            blinds.append(blind)
        if label:
            labels.append(label)
        stats.append(stat)

    blind_path = out / 'blind_events.jsonl'
    with blind_path.open('w', encoding='utf-8') as f:
        for x in blinds:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    label_path = out / 'labels_sealed.json'
    label_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'parse_stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')

    parser_counts = {}
    agent_stats = {}
    ratios = []
    for x in stats:
        parser_counts[x.get('selected_parser', 'none')] = parser_counts.get(x.get('selected_parser', 'none'), 0) + 1
        agent = x.get('agent', 'unknown')
        z = agent_stats.setdefault(agent, {'rows': 0, 'parsed': 0, 'exact': 0, 'ratios': []})
        z['rows'] += 1
        parsed = int(x.get('parsed', 0))
        expected = max(1, int(x.get('expected', 1)))
        if parsed > 0:
            z['parsed'] += 1
            z['ratios'].append(parsed / expected)
            ratios.append(parsed / expected)
        z['exact'] += int(parsed == expected)
    for z in agent_stats.values():
        rs = sorted(z.pop('ratios'))
        z['median_count_ratio'] = rs[len(rs)//2] if rs else 0.0
        z['mean_count_ratio'] = sum(rs)/len(rs) if rs else 0.0

    manifest = {
        'dataset': 'NJU-LINK/CodeTraceBench',
        'split': args.split,
        'rows': len(ds),
        'parsed': len(blinds),
        'exact_step_count': sum(int(x.get('parsed', 0)) == int(x.get('expected', -1)) for x in stats),
        'mean_count_ratio': sum(ratios)/len(ratios) if ratios else 0.0,
        'blind_instances': len(blinds),
        'sealed_labels': len(labels),
        'workers': args.workers,
        'parser_counts': parser_counts,
        'agent_parse_stats': agent_stats,
        'blind_sha256': hashlib.sha256(blind_path.read_bytes()).hexdigest(),
        'labels_sha256': hashlib.sha256(label_path.read_bytes()).hexdigest(),
        'labels_not_used_for_parser_selection': True,
        'parser_selection_signal': 'public step_count only',
    }
    (out / 'prepare_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
