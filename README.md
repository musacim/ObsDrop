# ObsDrop

Coding agents resend their whole history on every model call, and most of it is old tool
output. Editing the prompt to remove it (masking, summarization) breaks prefix-cache reuse,
so the next call recomputes everything after the edit. ObsDrop removes old observations
from the **KV cache** instead of the prompt: after each turn, the KV of all tool outputs
except the newest K is freed, while the prompt stays append-only. The next call still
matches the full cached prefix, prefills only the new turn, and attends to fewer tokens.

## SGLang

ObsDrop runs inside SGLang's radix cache (patch against `v0.5.3rc0`, commit `86a32bb`).

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout v0.5.3rc0
git apply /path/to/ObsDrop/sglang/sglang.patch
pip install -e python
MODEL=/path/to/Qwen3-Coder-30B-A3B-Instruct OBS_K=8 bash /path/to/ObsDrop/sglang/launch_obsdrop.sh
```

`OBS_K` is the window (unset = no dropping); `launch_obsdrop.sh` sets the paper's server flags.
Observation delimiters default to Qwen3's `<tool_response>` tokens; for other models set
`SGLANG_OBS_MARKERS` (open,close token ids) and `SGLANG_OBS_PH` (placeholder token ids),
e.g. Devstral's `[TOOL_RESULTS]` = `7,8`.
