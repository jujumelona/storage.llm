from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_debug_report import analyze_report


class DebugReportAnalysisTests(unittest.TestCase):
    def test_detects_current_token_copy_bias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ppl_results.jsonl").write_text(json.dumps({"mean_nll": 12.0, "ppl": 160000.0}) + "\n", encoding="utf-8")
            (root / "curl_latency.jsonl").write_text(json.dumps({"tok_per_sec": 0.08}) + "\n", encoding="utf-8")
            (root / "summary.jsonl").write_text(json.dumps({"name": "genai_perf", "status": "failed"}) + "\n", encoding="utf-8")
            (root / "server.log").write_text(
                "\n".join([
                    "[storagellm trace] eval_forward_call_begin pos=1 token=818 target=3890",
                    "[storagellm trace] lm_head_logprob_end target=3890 target_score=-12.7 best=818 best_score=17.3 target_rank=92387 best_gap=30.0 global_max=17.3 logsumexp=17.3 logprob=-30.0 softcap=30",
                    "[storagellm trace] eval_forward_call_begin pos=2 token=3890 target=563",
                    "[storagellm trace] lm_head_logprob_end target=563 target_score=11.1 best=3890 best_score=12.1 target_rank=2 best_gap=1.0 global_max=12.1 logsumexp=13.2 logprob=-2.1 softcap=30",
                    "[storagellm trace] eval_forward_call_begin pos=3 token=563 target=236743",
                    "[storagellm trace] lm_head_logprob_end target=236743 target_score=1.4 best=563 best_score=15.5 target_rank=22 best_gap=14.0 global_max=15.5 logsumexp=15.5 logprob=-14.0 softcap=30",
                    "[storagellm trace] eval_forward_call_begin pos=4 token=236743 target=236812",
                    "[storagellm trace] lm_head_logprob_end target=236812 target_score=11.3 best=236772 best_score=12.1 target_rank=4 best_gap=0.75 global_max=12.1 logsumexp=13.6 logprob=-2.2 softcap=30",
                    "[storagellm trace] forward_token_end position=1 token=818 final_norm_ms=130.0 total_ms=8121.0",
                    "[storagellm trace] forward_layer position=1 layer=0 split=1 is_moe=1 attention_ms=171.0 mlp_ms=86.0 total_ms=259.0",
                    "[storagellm trace] attn_standard_rope layer=0 pos=0 rope_dim=256 ms=488.0",
                    "[storagellm request] eval_engine_end io_queued=10 io_done=8 io_failed=2 io_dropped=0 disk_q=0 pinned_q=0 gpu_q=0 active_workers=0 bytes_prefetched=4096",
                ]),
                encoding="utf-8",
            )

            summary = analyze_report(root)
            names = {item.name for item in summary.failure_modes}
            self.assertIn("current_token_copy_bias", names)
            self.assertIn("ppl_quality_regression", names)
            self.assertIn("decode_throughput_regression", names)
            self.assertEqual(summary.current_token_copy_hits, 3)
            self.assertEqual(summary.scored_tokens_seen, 4)
            self.assertAlmostEqual(summary.current_token_copy_rate, 0.75)

    def test_clean_report_has_no_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ppl_results.jsonl").write_text(json.dumps({"mean_nll": 2.0, "ppl": 7.4}) + "\n", encoding="utf-8")
            (root / "curl_latency.jsonl").write_text(json.dumps({"tok_per_sec": 12.5}) + "\n", encoding="utf-8")
            (root / "summary.jsonl").write_text(json.dumps({"name": "ppl", "status": "ok"}) + "\n", encoding="utf-8")
            (root / "server.log").write_text(
                "\n".join([
                    "[storagellm trace] eval_forward_call_begin pos=1 token=10 target=11",
                    "[storagellm trace] lm_head_logprob_end target=11 target_score=5 best=11 best_score=5 target_rank=1 best_gap=0 global_max=5 logsumexp=6 logprob=-1 softcap=30",
                    "[storagellm trace] forward_token_end position=1 token=10 final_norm_ms=2 total_ms=20",
                    "[storagellm trace] forward_layer position=1 layer=0 split=1 is_moe=1 attention_ms=3 mlp_ms=4 total_ms=7",
                ]),
                encoding="utf-8",
            )
            summary = analyze_report(root)
            self.assertEqual(summary.failure_modes, [])
            self.assertEqual(summary.current_token_copy_hits, 0)


if __name__ == "__main__":
    unittest.main()
