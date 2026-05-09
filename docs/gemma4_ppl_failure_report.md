# Gemma4 PPL/debug-report failure report

This note records the failure signature observed in `debug_report.zip` from the Gemma4 26B A4B MXFP4/JUJU run and defines the checks that must gate future fixes.

## Observed failure

The run did not crash, but it was not a valid quality pass.

- PPL results were extremely high: the three examples reported roughly `2.56e5`, `4.23e7`, and `4.80e6` perplexity.
- The generation latency probe produced corrupted/repetitive text such as `hellohellohello로` and ` is is is is--1` at only about `0.07-0.10 tok/s`.
- The LM-head traces show a current-token copy pattern. In several scored positions, after forwarding token `x`, the best LM-head token was again `x` instead of the target next token. Example: after forwarding token `818`, the best token was `818` while the target was `3890`.
- `genai_perf` and `aiperf` were not installed in the run environment, so their failure must be treated as missing tooling rather than engine validation.

## Primary hypotheses to verify before kernel changes

1. **Prompt/PPL protocol mismatch**: the run used raw strings against an instruction-tuned Gemma artifact. Raw-text PPL can be misleading unless the same chat template and BOS/EOS policy used by generation is applied, or a base model is used.
2. **Transformer output under-contribution**: because the LM head is tied to `token_embd.weight`, a weak or incorrectly scaled transformer branch makes the current token embedding dominate the final logits.
3. **Layer/output scale placement**: traces show `after_output_scale` reducing the whole hidden vector by roughly a constant factor in late layers. Verify against Graph IR whether `layer_output_scale` applies to the residual stream or only the branch output before residual addition.
4. **Eval-time speculative prefetch waste**: the report shows large queued/prefetched byte counts and failed I/O requests during a small PPL run. Eval/PPL should be able to disable speculative prefetch unless the test is explicitly measuring the prefetcher.
5. **CPU-bound baseline**: median token latency is multiple seconds on CPU. This should be separated from quality regressions so that a slow but correct reference mode can still be used for logit comparison.

## Regression gate added

`tools/analyze_debug_report.py` parses a debug report directory or ZIP and fails on high-severity signatures:

- high `mean_nll` / PPL quality regression;
- current-token copy bias in LM-head traces;
- sub-1 tok/s decode probes;
- large RoPE warmup spikes;
- prefetch I/O failures;
- missing optional benchmark tools.

Run it on a report:

```bash
python tools/analyze_debug_report.py debug_report.zip
python tools/analyze_debug_report.py debug_report.zip --json
```

Run the unit tests:

```bash
python -m unittest tests/test_debug_report_analysis.py
```

## Required follow-up validation

A real engine fix is not complete until the same Gemma artifact is rerun and this checker no longer reports `ppl_quality_regression` or `current_token_copy_bias`. The next code patch should compare logits for a short token sequence against a known-good backend before changing sampler behavior.
