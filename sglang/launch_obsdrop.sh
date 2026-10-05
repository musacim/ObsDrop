#!/bin/bash
# Launch a patched SGLang server with ObsDrop on one 4-GPU node: 4 attention data-parallel
# replicas, experts split across the 4 GPUs. Settings match the paper's SGLang runs.
#
#   MODEL=/path/to/Qwen3-Coder-30B-A3B-Instruct OBS_K=8 bash launch_obsdrop.sh
#
# Environment:
#   MODEL         model directory (required)
#   OBS_K         ObsDrop window: observations visible at each model call (unset or 0 = no dropping)
#   RESTORE_MASK  1 = mask dropped observations when an evicted conversation returns (default 1)
#   ATTN          attention backend: flashinfer (default) or triton
#   GRAPH         1 = CUDA graphs for decode (default), 0 = eager
#   KV_TOKENS     KV pool size per replica in tokens (default 122400)
#   PORT          server port (default 8000)
set -e
: "${MODEL:?set MODEL to the model directory}"
ATTN=${ATTN:-flashinfer}; GRAPH=${GRAPH:-1}; KV_TOKENS=${KV_TOKENS:-122400}; PORT=${PORT:-8000}

export SGLANG_DP_STICKY=1                                   # all requests of a conversation go to one replica
if [ -n "$OBS_K" ] && [ "$OBS_K" != 0 ]; then
  export SGLANG_OBS_WINDOW=$OBS_K                           # turns ObsDrop on (drops happen inside the radix cache)
  [ "${RESTORE_MASK:-1}" = 1 ] && export SGLANG_OBS_RESTORE_MASK=1
fi
export SGLANG_TRUE_TTFT=1 SGLANG_LOG_TTFT=1 SGLANG_LOG_EVICT=1   # per-request [REQ]/[TTFT] and [EVICT] log lines

if [ "$GRAPH" = 1 ]; then GRAPHFLAGS="--cuda-graph-max-bs 16"; else GRAPHFLAGS="--disable-cuda-graph"; fi

exec python -m sglang.launch_server --model-path "$MODEL" --served-model-name qwen3-coder \
  --host 127.0.0.1 --port "$PORT" \
  --tp 4 --dp 4 --enable-dp-attention --ep-size 4 \
  --mem-fraction-static 0.85 --max-total-tokens "$KV_TOKENS" --context-length 114688 \
  --max-running-requests 64 --chunked-prefill-size 32768 --max-prefill-tokens 8192 \
  --attention-backend "$ATTN" --sampling-backend pytorch $GRAPHFLAGS \
  --tool-call-parser qwen3_coder --enable-metrics --enable-cache-report
