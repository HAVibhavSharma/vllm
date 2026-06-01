#!/usr/bin/env bash
# Manual end-to-end test for the agent prefetch flow with a seed text.
#
# Pipeline:
#   1. Reset APC + registry + LMCache (cold start).
#   2. POST /v1/agents/prefetch with text = contents of prefix.txt and
#      agent_kind=non-react. Server seeds the registry with the
#      chunk-aligned prefix and warms APC.
#   3. POST /v1/agents/chat/completions with the same prefix as the
#      system message and a fresh user question appended. If the
#      prefetch worked, the response's usage block should report
#      most of the system tokens as cached.
#   4. Print /v1/agents/registry_stats for sanity.
#
# Requires: curl, jq, python3.
#
# Override via env vars:
#   HOST=http://localhost:8000
#   MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
#   AGENT_ID=policy_bot
#   USER_QUESTION="What should I do if a teammate leaks credentials?"

set -euo pipefail

HOST="${HOST:-http://localhost:8000}"
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct-AWQ}"
AGENT_ID="${AGENT_ID:-policy_bot}"
USER_QUESTION="${USER_QUESTION:-What should I do if a teammate leaks credentials?}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX_FILE="$HERE/prefix.txt"

if [[ ! -f "$PREFIX_FILE" ]]; then
    echo "prefix.txt not found at $PREFIX_FILE" >&2
    exit 1
fi

PREFIX_TEXT="$(cat "$PREFIX_FILE")"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1. Reset APC + registry + connector (cold start)"
curl -sS -X POST \
  "$HOST/v1/agents/reset_prefix_cache?reset_apc=true&reset_registry=true&reset_connector=true" \
  | python3 -m json.tool

hr "2. Seed registry from prefix.txt + warm APC (agent_kind=non-react)"
jq -n \
    --arg agent "$AGENT_ID" \
    --arg text  "$PREFIX_TEXT" \
    '{
        agent_id:    $agent,
        agent_kind:  "non-react",
        text:        $text,
        wait:        true
    }' \
  | curl -sS -X POST "$HOST/v1/agents/prefetch" \
      -H 'Content-Type: application/json' \
      --data-binary @- \
  | python3 -m json.tool

metric() {
    # Sum every Prometheus sample whose name starts with $1 (allowing an
    # optional `_total` suffix), across all label combinations. Skips
    # comment lines and HELP/TYPE annotations. Returns 0 if no match.
    local name="$1"
    curl -sS "$HOST/metrics" \
      | awk -v n="^${name}(_total)?(\\{| )" '
          $0 !~ /^#/ && $0 ~ n { sum += $NF }
          END { printf "%d", (sum ? sum : 0) }
        '
}

hr "3a. Prefix-cache counters BEFORE chat"
HITS_BEFORE=$(metric vllm:gpu_prefix_cache_hits)
QUERIES_BEFORE=$(metric vllm:gpu_prefix_cache_queries)
printf '  hits_before    = %s\n  queries_before = %s\n' \
    "$HITS_BEFORE" "$QUERIES_BEFORE"

hr "3b. Chat with same prefix + new user turn"
# record_in_registry=false keeps the registry at exactly 1 entry (the
# step-2 seed), so step 4's count is clean.
jq -n \
    --arg model "$MODEL" \
    --arg agent "$AGENT_ID" \
    --arg sys   "$PREFIX_TEXT" \
    --arg usr   "$USER_QUESTION" \
    '{
        model:               $model,
        agent_id:            $agent,
        record_in_registry:  false,
        messages: [
            {role: "system", content: $sys},
            {role: "user",   content: $usr}
        ],
        max_tokens:  64,
        temperature: 0.0,
        stream:      false
    }' \
  | curl -sS -X POST "$HOST/v1/agents/chat/completions" \
      -H 'Content-Type: application/json' \
      --data-binary @- \
  | python3 -m json.tool

hr "3c. Prefix-cache counters AFTER chat (delta = cache effect of step 3)"
HITS_AFTER=$(metric vllm:gpu_prefix_cache_hits)
QUERIES_AFTER=$(metric vllm:gpu_prefix_cache_queries)
HITS_DELTA=$(python3 -c "print(int(float('$HITS_AFTER')) - int(float('$HITS_BEFORE')))")
QUERIES_DELTA=$(python3 -c "print(int(float('$QUERIES_AFTER')) - int(float('$QUERIES_BEFORE')))")
printf '  hits_after     = %s   (delta = %s)\n' "$HITS_AFTER" "$HITS_DELTA"
printf '  queries_after  = %s   (delta = %s)\n' "$QUERIES_AFTER" "$QUERIES_DELTA"
if [[ "$HITS_DELTA" -gt 0 ]]; then
    printf '  RESULT: cache hit observed (%s tokens reused)\n' "$HITS_DELTA"
else
    printf '  RESULT: NO cache hit — investigate prefetch path\n'
fi

hr "4. Registry stats (expect 1 prefix for $AGENT_ID)"
curl -sS "$HOST/v1/agents/registry_stats" | python3 -m json.tool
