"""Scheduler side of ObsDrop for vLLM v1 (engine-core process).

1. When a request is admitted (KVCacheManager.get_computed_blocks), every KV block of the prompt that
   lies inside an observation older than the last K is marked DEAD (by its block hash, which depends
   only on token ids, so it is the same for every later turn of the conversation). A dead block that
   is still cached is evicted from the prefix cache and moved to the head of the free queue, so its
   memory is reused first.
2. BlockPool.get_cached_block returns the null block (block 0) for a dead hash. vLLM's own prefix-hit
   loop then walks through the hole instead of stopping, so the request still hits the whole logical
   prefix, holds no memory for the dead blocks, and only the new suffix is prefilled. Null blocks in a
   request's block list are already supported by vLLM (sliding-window attention uses them).
3. The worker side (worker_patch.py) removes the null blocks from the block table given to attention.

Only blocks that the prefix hit reaches are dropped. A dead-candidate block that is not cached (the
conversation was evicted) stops the hit, and the request recomputes from there with the observation
visible (no restore masking yet); the block is dropped at the next turn.
"""
import os
import time
from collections import OrderedDict

from .policy import dead_blocks

_K = int(os.environ.get("OBSDROP_K", "8"))
_OPEN, _CLOSE = (int(x) for x in os.environ.get("OBSDROP_MARKERS", "151665,151666").split(","))
_LOG = os.environ.get("OBSDROP_LOG", "1") == "1"
_DEAD_CAP = int(os.environ.get("OBSDROP_DEAD_CAP", "4000000"))

DEAD: "OrderedDict[object, None]" = OrderedDict()   # dead block hashes (LRU-capped)
STATS = dict(reqs=0, dead_tokens=0, freed_blocks=0)


def _mark_dead(h) -> None:
    DEAD[h] = None
    DEAD.move_to_end(h)
    if len(DEAD) > _DEAD_CAP:
        DEAD.popitem(last=False)


def _free_cached(block_pool, blocks) -> int:
    """Evict cached copies of a dead block that no running request holds; move them to the queue head."""
    n = 0
    for blk in blocks:
        if blk.is_null or blk.ref_cnt != 0:
            continue
        block_pool._maybe_evict_cached_block(blk)
        block_pool.free_block_queue.remove(blk)
        block_pool.free_block_queue.prepend_n([blk])
        n += 1
    return n


def _prepass(mgr, request) -> tuple[int, int]:
    """Mark the reachable dead-candidate blocks of `request` DEAD; return (dead blocks, freed blocks)."""
    pool = mgr.block_pool
    bs = pool.hash_block_size
    hashes = request.block_hashes
    groups = list(range(len(mgr.kv_cache_config.kv_cache_groups)))
    cand = set(dead_blocks(request.prompt_token_ids, _K, _OPEN, _CLOSE, bs))
    if not cand:
        return 0, 0
    n_dead = n_freed = 0
    limit = min(len(hashes), (request.num_tokens - 1) // bs)
    for i in range(limit):
        h = hashes[i]
        if h in DEAD:                      # already dropped in an earlier turn
            n_dead += i in cand
            stale = _ORIG_GET_CACHED(pool, h, groups)   # recomputed after an eviction and re-cached
            if stale is not None:
                n_freed += _free_cached(pool, stale)
            continue
        cached = _ORIG_GET_CACHED(pool, h, groups)
        if cached is None:                 # prefix hit ends here
            break
        if i in cand:                      # drop it now
            _mark_dead(h)
            n_freed += _free_cached(pool, cached)
            n_dead += 1
    return n_dead, n_freed


_ORIG_GET_CACHED = None


def apply() -> None:
    global _ORIG_GET_CACHED
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(BlockPool, "_obsdrop_patched", False):
        return
    _ORIG_GET_CACHED = BlockPool.get_cached_block

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        if block_hash in DEAD:
            return [self.null_block] * len(kv_cache_group_ids)
        return _ORIG_GET_CACHED(self, block_hash, kv_cache_group_ids)

    BlockPool.get_cached_block = get_cached_block
    BlockPool._obsdrop_patched = True

    orig_gcb = KVCacheManager.get_computed_blocks

    def get_computed_blocks(self, request):
        n_dead = n_freed = 0
        if self.prefix_cache_lookup_enabled(request):
            groups = self.kv_cache_config.kv_cache_groups
            if len(groups) != 1 or self.block_pool.hash_block_size != groups[0].kv_cache_spec.block_size:
                raise RuntimeError("ObsDrop-vLLM supports one full-attention KV group with block size = hash block size")
            n_dead, n_freed = _prepass(self, request)
        out = orig_gcb(self, request)
        STATS["reqs"] += 1
        STATS["dead_tokens"] += n_dead * self.block_pool.hash_block_size
        STATS["freed_blocks"] += n_freed
        if _LOG:
            print(
                f"[OBSDROP] t={time.time():.3f} rid={request.request_id} prompt={request.num_tokens} "
                f"hit={out[1]} dead={n_dead * self.block_pool.hash_block_size} freed_now={n_freed} "
                f"free_blocks={self.block_pool.get_num_free_blocks()} dead_hashes={len(DEAD)}",
                flush=True,
            )
        return out

    KVCacheManager.get_computed_blocks = get_computed_blocks
    print(f"[OBSDROP] scheduler patch on: keep last K={_K} observations, markers=({_OPEN},{_CLOSE})", flush=True)
