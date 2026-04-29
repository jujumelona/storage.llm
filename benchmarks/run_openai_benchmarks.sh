#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-http://127.0.0.1:8000}"
MODEL="${MODEL:-glm5.1-storage}"
MODEL_ROOT="${MODEL_ROOT:-/teamspace/studios/this_studio/storagellm_bench/models/GLM5.1-4q-storage}"
TOKENIZER="${TOKENIZER:-gpt2}"
PROMPT_TOKENS="${PROMPT_TOKENS:-256}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
CONCURRENCY="${CONCURRENCY:-1}"
MEASUREMENT_INTERVAL_MS="${MEASUREMENT_INTERVAL_MS:-30000}"
OUT_DIR="${OUT_DIR:-/teamspace/studios/this_studio/storagellm_bench/benchmark_results}"
URL_NO_SCHEME="${TARGET#http://}"
URL_NO_SCHEME="${URL_NO_SCHEME#https://}"

mkdir -p "$OUT_DIR"
export PATH="$HOME/.local/bin:$PATH"

run_step() {
  local name="$1"
  shift
  echo
  echo "=== $name ==="
  if "$@" 2>&1 | tee "$OUT_DIR/${name}.log"; then
    echo "{\"name\":\"$name\",\"status\":\"ok\"}" >> "$OUT_DIR/summary.jsonl"
  else
    local code=$?
    echo "{\"name\":\"$name\",\"status\":\"failed\",\"exit_code\":$code}" >> "$OUT_DIR/summary.jsonl"
    return 0
  fi
}

run_required_step() {
  local name="$1"
  shift
  echo
  echo "=== $name ==="
  "$@" 2>&1 | tee "$OUT_DIR/${name}.log"
}

wait_ready() {
  local ready=0
  for i in $(seq 1 1800); do
    health="$(curl -s "$TARGET/health" || true)"
    echo "$health" > "$OUT_DIR/health_latest.json"
    if echo "$health" | grep -q '"modelReady":true'; then
      echo "$health" | tee "$OUT_DIR/health_before_bench.json"
      ready=1
      break
    fi
    if echo "$health" | grep -q '"modelFailed":true'; then
      echo "$health" | tee "$OUT_DIR/health_failed.json"
      return 2
    fi
    if [ "$i" -eq 1 ] || [ $((i % 10)) -eq 0 ]; then
      stage="$(echo "$health" | sed -n 's/.*"modelLoadStage":"\([^"]*\)".*/\1/p')"
      echo "waiting ${i}s stage=${stage:-unknown}"
    fi
    sleep 1
  done
  [ "$ready" -eq 1 ]
}

system_snapshot() {
  echo "=== SYSTEM ==="
  date
  uname -a || true
  lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket|MHz' || true
  free -h || true
  df -h "$MODEL_ROOT" || true
  echo
  echo "=== GPU ==="
  nvidia-smi || true
  echo
  echo "=== HEALTH ==="
  curl -s "$TARGET/health" || true
  echo
}

curl_latency_bench() {
  local cases='[
    {"name":"hello_4","input":"hello","max_output_tokens":4},
    {"name":"answer_8","input":"The answer is","max_output_tokens":8},
    {"name":"storage_16","input":"Storage offloading benchmark:","max_output_tokens":16}
  ]'
  python3 - "$TARGET" "$MODEL" "$OUT_DIR/curl_latency.jsonl" "$cases" <<'PY'
import json
import sys
import time
import urllib.request

target, model, out_path, cases_json = sys.argv[1:]
cases = json.loads(cases_json)
with open(out_path, "w", encoding="utf-8") as out:
    for case in cases:
        payload = {
            "model": model,
            "input": case["input"],
            "max_output_tokens": case["max_output_tokens"],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            target.rstrip("/") + "/v1/responses",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=3600) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        sec = time.perf_counter() - t0
        obj = json.loads(raw)
        usage = obj.get("usage", {})
        out_tok = int(usage.get("output_tokens") or case["max_output_tokens"])
        rec = {
            "name": case["name"],
            "seconds": sec,
            "output_tokens": out_tok,
            "tok_per_sec": out_tok / sec if sec > 0 else None,
            "raw": obj,
        }
        print(json.dumps(rec, ensure_ascii=False))
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
PY
}

ppl_bench() {
  cat > "$OUT_DIR/ppl_inputs.jsonl" <<EOF
{"model":"$MODEL","input":"The answer is 42."}
{"model":"$MODEL","input":"Storage offloading keeps hot tensors close to compute while cold tensors stream from local storage."}
{"model":"$MODEL","input":"A benchmark should report latency, throughput, memory usage, and perplexity."}
EOF

  : > "$OUT_DIR/ppl_results.jsonl"
  while IFS= read -r payload; do
    echo "$payload" >&2
    curl -s "$TARGET/v1/perplexity" \
      -H "Content-Type: application/json" \
      -d "$payload" | tee -a "$OUT_DIR/ppl_results.jsonl"
    echo >> "$OUT_DIR/ppl_results.jsonl"
  done < "$OUT_DIR/ppl_inputs.jsonl"
}

genai_perf_bench() {
  if ! command -v genai-perf >/dev/null 2>&1; then
    python3 -m pip install --no-cache-dir --user genai-perf
  fi
  genai-perf --version || true
  PROFILE_HELP="$(genai-perf profile --help 2>&1 || true)"
  GENAI_FLAGS=()
  if echo "$PROFILE_HELP" | grep -q -- '--streaming'; then
    GENAI_FLAGS+=(--streaming)
  fi
  if echo "$PROFILE_HELP" | grep -q -- '--generate-plots'; then
    GENAI_FLAGS+=(--generate-plots)
  fi

  genai-perf profile \
    -m "$MODEL" \
    --service-kind openai \
    --endpoint-type completions \
    --url "$URL_NO_SCHEME" \
    --tokenizer "$TOKENIZER" \
    --num-prompts "$NUM_PROMPTS" \
    --synthetic-input-tokens-mean "$PROMPT_TOKENS" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OUTPUT_TOKENS" \
    --output-tokens-stddev 0 \
    --output-tokens-mean-deterministic \
    --concurrency "$CONCURRENCY" \
    --measurement-interval "$MEASUREMENT_INTERVAL_MS" \
    --profile-export-file "$OUT_DIR/genai_perf_profile.json" \
    "${GENAI_FLAGS[@]}"
}

aiperf_bench() {
  if ! command -v aiperf >/dev/null 2>&1; then
    python3 -m pip install --no-cache-dir --user aiperf
  fi
  aiperf --version || true
  aiperf profile \
    --model "$MODEL" \
    --endpoint-type completions \
    --endpoint /v1/completions \
    --url "$URL_NO_SCHEME" \
    --synthetic-input-tokens-mean "$PROMPT_TOKENS" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean "$OUTPUT_TOKENS" \
    --output-tokens-stddev 0 \
    --profile-export-file "$OUT_DIR/aiperf_profile"
}

guidellm_bench() {
  if ! command -v guidellm >/dev/null 2>&1; then
    echo "guidellm not installed; skipping"
    return 0
  fi
  guidellm benchmark \
    --target "$TARGET" \
    --model "$MODEL" \
    --rate-type sweep \
    --max-seconds 30 \
    --data "prompt_tokens=$PROMPT_TOKENS,output_tokens=$OUTPUT_TOKENS"
}

: > "$OUT_DIR/summary.jsonl"
run_step system_before system_snapshot
run_required_step wait_model_ready wait_ready
run_step curl_latency curl_latency_bench
run_step ppl ppl_bench
run_step genai_perf genai_perf_bench
run_step aiperf aiperf_bench
run_step guidellm_if_available guidellm_bench
run_step system_after system_snapshot

echo
echo "RESULT_DIR=$OUT_DIR"
echo "SUMMARY=$OUT_DIR/summary.jsonl"
echo "CURL_LATENCY=$OUT_DIR/curl_latency.jsonl"
echo "PPL_JSONL=$OUT_DIR/ppl_results.jsonl"
echo "GENAI_PERF_JSON=$OUT_DIR/genai_perf_profile.json"
echo "AIPERF_PREFIX=$OUT_DIR/aiperf_profile"
