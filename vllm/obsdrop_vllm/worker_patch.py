"""Worker side of ObsDrop for vLLM v1 (GPU worker processes).

A dead KV block appears in a request's block table as the null block (block 0, never a real block).
Attention must not read it, so before the attention metadata is built we compact every block-table row
on the GPU (drop the null entries, keep order) and shorten that request's KV length by the dropped
tokens. This is exact:
  * the slot mapping of the new tokens was already computed from the real table in _prepare_inputs;
  * cached keys already carry their RoPE positions, and query positions come from `positions`
    (logical, untouched), so removing earlier keys does not move anything;
  * the new tokens stay the last keys of the row, so the bottom-right causal alignment that vLLM uses
    for chunked prefill still lines up.
The CPU block table (the source of truth) is not touched; vLLM re-copies it to the GPU every step, and
recomputes seq_lens / optimistic_seq_lens_cpu every step, so the per-step adjustment never accumulates.
Same idea as the SGLang version, which filters dead slot 0 out of the KV index list.

All affected rows are compacted at once in numpy, written to a pinned staging buffer and sent with ONE non-blocking
copy, plus one non-blocking copy and one subtraction for seq_lens.
"""
import os
import time

import numpy as np
import torch

_LOG_EVERY = int(os.environ.get("OBSDROP_WORKER_LOG_EVERY", "200"))
_step = 0
_tot_dead = 0
_t_patch = 0.0          # seconds spent in the patch, all steps
_n_dead_steps = 0       # steps that had at least one dead block
_stage = None           # [2] rotating (pinned block-table buffer, pinned drop buffer, cuda event)
_cur = 0


def _staging(shape, dtype, n):
    global _stage
    if _stage is None:
        _stage = [(torch.zeros(shape, dtype=dtype, pin_memory=True), torch.zeros(n, dtype=torch.int32, pin_memory=True),
                   torch.cuda.Event()) for _ in range(2)]
    return _stage


def _vec(self, bt, rows, dead, valid, n_dead, num_reqs):
    global _tot_dead, _cur
    rs = np.nonzero(n_dead)[0]
    lo, hi = int(rs[0]), int(rs[-1]) + 1                      # rows [lo, hi) are rewritten (others unchanged)
    sub, width = rows[lo:hi], rows.shape[1]
    gone = dead[lo:hi] | ~valid[lo:hi]                        # dead blocks and the unused tail go to the end
    order = np.argsort(gone, axis=1, kind="stable")           # live blocks first, original order kept
    packed = np.where(np.take_along_axis(gone, order, axis=1), 0, np.take_along_axis(sub, order, axis=1))
    drop = (n_dead * bt.block_size).astype(np.int32)
    tab, dbuf, ev = _staging(bt.block_table.np.shape, bt.block_table.gpu.dtype, len(self.optimistic_seq_lens_cpu))[_cur]
    _cur ^= 1
    ev.synchronize()                                          # this buffer's previous copy (2 steps ago) is done
    tab.numpy()[: hi - lo, :width] = packed
    dbuf.numpy()[:num_reqs] = drop
    bt.block_table.gpu[lo:hi, :width].copy_(tab[: hi - lo, :width], non_blocking=True)
    self.seq_lens[:num_reqs].sub_(dbuf[:num_reqs].to(self.seq_lens.device, non_blocking=True))
    ev.record()
    self.optimistic_seq_lens_cpu[:num_reqs] -= torch.from_numpy(drop)
    _tot_dead += int(drop.sum())


def apply() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_obsdrop_patched", False):
        return
    orig = GPUModelRunner._build_attention_metadata

    def _build_attention_metadata(self, num_tokens, num_reqs, max_query_len, *args, **kwargs):
        global _step, _t_patch, _n_dead_steps
        if not kwargs.get("for_cudagraph_capture", False) and num_reqs > 0:
            t0 = time.perf_counter()
            mbt = self.input_batch.block_table
            if getattr(self, "cascade_attn_enabled", False):
                raise RuntimeError("ObsDrop-vLLM needs --disable-cascade-attn (cascade attention reads shared prefix blocks)")
            if len(self.kv_cache_config.kv_cache_groups) != 1:
                raise RuntimeError("ObsDrop-vLLM supports a single KV cache group")
            bt = mbt[0]
            bpk = bt.blocks_per_kv_block          # null KV block 0 maps to kernel blocks [0, bpk)
            nb = bt.num_blocks_per_row[:num_reqs]
            width = int(nb.max()) if num_reqs else 0
            if width > 0:
                rows = bt.block_table.np[:num_reqs, :width]
                valid = np.arange(width)[None, :] < nb[:, None]
                dead = (rows < bpk) & valid
                n_dead = dead.sum(axis=1)
                if n_dead.any():
                    _n_dead_steps += 1
                    _vec(self, bt, rows, dead, valid, n_dead, num_reqs)
            _t_patch += time.perf_counter() - t0
            _step += 1
            if _LOG_EVERY and _step % _LOG_EVERY == 0:
                print(f"[OBSDROP-W] step {_step}: dead KV tokens skipped so far {_tot_dead}, "
                      f"steps with drops {_n_dead_steps}, patch time {1e6 * _t_patch / _step:.0f} us/step", flush=True)
        return orig(self, num_tokens, num_reqs, max_query_len, *args, **kwargs)

    GPUModelRunner._build_attention_metadata = _build_attention_metadata
    GPUModelRunner._obsdrop_patched = True
    print("[OBSDROP] worker patch on: null blocks are removed from the attention block table", flush=True)
