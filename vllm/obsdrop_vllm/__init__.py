"""ObsDrop for vLLM v1: drop the KV of old tool observations while keeping the prefix cache hit.

Loaded by vLLM's plugin system (entry point group `vllm.general_plugins`, declared in ../pyproject.toml)
in every vLLM process after `pip install ./vllm`. No vLLM file is edited. Off unless OBSDROP_K is set.

  OBSDROP_K        keep the KV of the last K observations (SGLang SGLANG_OBS_WINDOW semantics)
  OBSDROP_MARKERS  open,close token ids of an observation (Qwen3 <tool_response> = 151665,151666;
                   Devstral [TOOL_RESULTS] = 7,8)
  OBSDROP_LOG      1 (default): one [OBSDROP] line per admitted request in the engine-core log
Serve with --disable-cascade-attn (prefix caching stays on).
"""
import os


def register() -> None:
    if not os.environ.get("OBSDROP_K"):
        return
    from . import sched_patch, worker_patch

    sched_patch.apply()
    try:
        worker_patch.apply()
    except ImportError as e:   # process without the GPU model runner
        print(f"[OBSDROP] worker patch skipped in this process: {e}", flush=True)
