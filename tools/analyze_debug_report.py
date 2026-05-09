#!/usr/bin/env python3
"""Analyze StorageLLM debug_report outputs for PPL and latency regressions."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FORWARD_BEGIN_RE = re.compile(r"eval_forward_call_begin\s+pos=(?P<pos>\d+)\s+token=(?P<token>\d+)\s+target=(?P<target>\d+)")
LM_HEAD_END_RE = re.compile(r"lm_head_logprob_end\s+target=(?P<target>\d+).*?best=(?P<best>\d+).*?target_rank=(?P<rank>\d+).*?best_gap=(?P<gap>[-+0-9.eE]+)")
FORWARD_LAYER_RE = re.compile(r"forward_layer\s+position=(?P<pos>\d+)\s+layer=(?P<layer>\d+).*?attention_ms=(?P<attention>[-+0-9.eE]+)\s+mlp_ms=(?P<mlp>[-+0-9.eE]+)\s+total_ms=(?P<total>[-+0-9.eE]+)")
FORWARD_TOKEN_END_RE = re.compile(r"forward_token_end\s+position=(?P<pos>\d+)\s+token=(?P<token>\d+)\s+final_norm_ms=(?P<final_norm>[-+0-9.eE]+)\s+total_ms=(?P<total>[-+0-9.eE]+)")
ROPE_RE = re.compile(r"attn_standard_rope\s+layer=(?P<layer>\d+)\s+pos=(?P<pos>\d+).*?ms=(?P<ms>[-+0-9.eE]+)")
IO_RE = re.compile(r"io_queued=(?P<queued>\d+)\s+io_done=(?P<done>\d+)\s+io_failed=(?P<failed>\d+).*?bytes_prefetched=(?P<prefetched>\d+)")


@dataclass
class FailureMode:
    name: str
    severity: str
    evidence: str
    recommendation: str


@dataclass
class DebugReportSummary:
    path: str
    ppl_count: int
    mean_ppl: float | None
    max_ppl: float | None
    mean_nll: float | None
    current_token_copy_rate: float
    current_token_copy_hits: int
    scored_tokens_seen: int
    curl_tokens_per_second_min: float | None
    curl_tokens_per_second_mean: float | None
    forward_token_ms_median: float | None
    forward_layer_attention_ms_median: float | None
    forward_layer_mlp_ms_median: float | None
    rope_ms_max: float | None
    io_failed_total: int
    io_prefetched_bytes_max: int
    failed_tools: list[str]
    failure_modes: list[FailureMode]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def open_report(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a directory or ZIP file: {path}")
    tmp = tempfile.TemporaryDirectory(prefix="storagellm-debug-")
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp.name)
    root = Path(tmp.name)
    nested = root / "debug_report"
    return nested if nested.exists() else root, tmp


def analyze_report(path: str | Path) -> DebugReportSummary:
    input_path = Path(path)
    report, tmp = open_report(input_path)
    try:
        ppl_rows = read_jsonl(report / "ppl_results.jsonl")
        ppls = [x for x in (finite_float(r.get("ppl", r.get("perplexity"))) for r in ppl_rows) if x is not None]
        nlls = [x for x in (finite_float(r.get("mean_nll")) for r in ppl_rows) if x is not None]
        curl_rows = read_jsonl(report / "curl_latency.jsonl")
        tps = [x for x in (finite_float(r.get("tok_per_sec")) for r in curl_rows) if x is not None]
        failed_tools = [str(r.get("name")) for r in read_jsonl(report / "summary.jsonl") if str(r.get("status", "")).lower() == "failed"]

        queue: list[dict[str, int]] = []
        current_copy_hits = 0
        scored_seen = 0
        token_ms: list[float] = []
        attention_ms: list[float] = []
        mlp_ms: list[float] = []
        rope_ms: list[float] = []
        io_failed_total = 0
        io_prefetched_bytes_max = 0

        server_log = report / "server.log"
        if server_log.exists():
            with server_log.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if m := FORWARD_BEGIN_RE.search(line):
                        queue.append({k: int(v) for k, v in m.groupdict().items()})
                        queue = queue[-64:]
                    elif m := LM_HEAD_END_RE.search(line):
                        target = int(m.group("target"))
                        best = int(m.group("best"))
                        scored_seen += 1
                        paired = None
                        for idx in range(len(queue) - 1, -1, -1):
                            if queue[idx]["target"] == target:
                                paired = queue.pop(idx)
                                break
                        if paired and best == paired["token"] and paired["token"] != target:
                            current_copy_hits += 1
                    elif m := FORWARD_TOKEN_END_RE.search(line):
                        token_ms.append(float(m.group("total")))
                    elif m := FORWARD_LAYER_RE.search(line):
                        attention_ms.append(float(m.group("attention")))
                        mlp_ms.append(float(m.group("mlp")))
                    elif m := ROPE_RE.search(line):
                        rope_ms.append(float(m.group("ms")))
                    elif m := IO_RE.search(line):
                        io_failed_total += int(m.group("failed"))
                        io_prefetched_bytes_max = max(io_prefetched_bytes_max, int(m.group("prefetched")))

        mean_nll = mean(nlls)
        mean_ppl = mean(ppls)
        max_ppl = max(ppls) if ppls else None
        tps_min = min(tps) if tps else None
        tps_mean = mean(tps)
        copy_rate = current_copy_hits / scored_seen if scored_seen else 0.0
        failures: list[FailureMode] = []
        if mean_nll is not None and mean_nll > 8.0:
            failures.append(FailureMode("ppl_quality_regression", "critical", f"mean_nll={mean_nll:.3f}, mean_ppl={mean_ppl:.3g}, max_ppl={max_ppl:.3g}", "Do not treat eval as passed; compare logits with a reference backend first."))
        if scored_seen >= 4 and copy_rate >= 0.10:
            failures.append(FailureMode("current_token_copy_bias", "critical", f"best token equaled the just-forwarded token for {current_copy_hits}/{scored_seen} scored positions", "Inspect transformer branch scaling, final hidden handed to LM head, and chat-template/prompt formatting."))
        if tps_min is not None and tps_min < 1.0:
            failures.append(FailureMode("decode_throughput_regression", "high", f"min_tok_per_sec={tps_min:.4f}, mean_tok_per_sec={tps_mean:.4f}", "Profile layer medians; wall-clock success is not enough."))
        if rope_ms and max(rope_ms) > 100.0:
            failures.append(FailureMode("rope_warmup_spike", "medium", f"max_rope_ms={max(rope_ms):.3f}", "Cache or precompute dynamic RoPE tables outside the first-token critical path."))
        if io_failed_total > 0:
            failures.append(FailureMode("prefetch_io_failures", "medium", f"aggregated_io_failed={io_failed_total}, max_prefetched_bytes={io_prefetched_bytes_max}", "Gate speculative prefetch during eval/PPL or suppress impossible storage-tier requests."))
        if failed_tools:
            failures.append(FailureMode("benchmark_tooling_missing", "low", f"failed tools: {', '.join(failed_tools)}", "Install optional perf tools or mark them skipped."))

        return DebugReportSummary(
            path=str(input_path),
            ppl_count=len(ppls),
            mean_ppl=mean_ppl,
            max_ppl=max_ppl,
            mean_nll=mean_nll,
            current_token_copy_rate=copy_rate,
            current_token_copy_hits=current_copy_hits,
            scored_tokens_seen=scored_seen,
            curl_tokens_per_second_min=tps_min,
            curl_tokens_per_second_mean=tps_mean,
            forward_token_ms_median=median(token_ms),
            forward_layer_attention_ms_median=median(attention_ms),
            forward_layer_mlp_ms_median=median(mlp_ms),
            rope_ms_max=max(rope_ms) if rope_ms else None,
            io_failed_total=io_failed_total,
            io_prefetched_bytes_max=io_prefetched_bytes_max,
            failed_tools=failed_tools,
            failure_modes=failures,
        )
    finally:
        if tmp is not None:
            tmp.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="debug_report directory or debug_report.zip")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args(argv)
    summary = analyze_report(args.report)
    if args.json:
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    else:
        print(f"report: {summary.path}")
        print(f"mean_nll: {summary.mean_nll}")
        print(f"mean_ppl: {summary.mean_ppl}")
        print(f"current_token_copy: {summary.current_token_copy_hits}/{summary.scored_tokens_seen} ({summary.current_token_copy_rate:.1%})")
        print(f"tok_per_sec_min: {summary.curl_tokens_per_second_min}")
        print("failure_modes:")
        for failure in summary.failure_modes:
            print(f"- [{failure.severity}] {failure.name}: {failure.evidence}")
            print(f"  recommendation: {failure.recommendation}")
    failed = any(f.severity in {"critical", "high"} for f in summary.failure_modes)
    return 0 if args.allow_fail or not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
