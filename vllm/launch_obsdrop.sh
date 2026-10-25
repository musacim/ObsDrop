#!/bin/bash
# Launch a stock vLLM server with the ObsDrop plugin on one 4-GPU node (tensor parallel 4).
# No vLLM file is edited: after `pip install ./vllm`, vLLM's plugin loader applies ObsDrop.
#
#   MODEL=/path/to/Qwen3-Coder-30B-A3B-Instruct OBS_K=8 bash launch_obsdrop.sh
#
# Environment:
#   MODEL      model directory (required)
#   OBS_K      ObsDrop window: observations visible at each model call (unset or 0 = no dropping)
#   MARKERS    open,close token ids of a tool output (default Qwen3 <tool_response> = 151665,151666)
#   KV_TOKENS  KV pool size in tokens (default 489600)
#   PORT       server port (default 8000)
set -e
: "${MODEL:?set MODEL to the model directory}"
KV_TOKENS=${KV_TOKENS:-489600}; PORT=${PORT:-8000}

if [ -n "$OBS_K" ] && [ "$OBS_K" != 0 ]; then
  export OBSDROP_K=$OBS_K OBSDROP_MARKERS=${MARKERS:-151665,151666}   # turns ObsDrop on
else
  unset OBSDROP_K
fi

exec vllm serve "$MODEL" --served-model-name qwen3-coder --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 4 --max-model-len 131072 --gpu-memory-utilization 0.85 \
  --num-gpu-blocks-override $((KV_TOKENS / 16)) --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --enable-prompt-tokens-details --disable-cascade-attn \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
