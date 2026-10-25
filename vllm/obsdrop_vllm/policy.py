"""ObsDrop drop policy: which KV blocks of a prompt are dead.

Same rule as the SGLang implementation (sglang_fi/sglang/srt/mem_cache/radix_cache.py, _ds_obs_drop):
an observation is the CONTENT between an open marker and the next close marker (markers stay live);
every observation except the last `keep` is dropped. vLLM manages KV in fixed-size blocks, so only
blocks that lie entirely inside a dropped content span are dead; the (at most two) partial blocks at
the edges of a span stay live.
"""


def obs_spans(token_ids, open_id: int, close_id: int) -> list[tuple[int, int]]:
    """[start, end) of every observation's content (markers excluded), in order."""
    spans, start = [], None
    for i, t in enumerate(token_ids):
        if t == open_id:
            start = i + 1
        elif t == close_id and start is not None:
            if i > start:
                spans.append((start, i))
            start = None
    return spans


def dead_blocks(token_ids, keep: int, open_id: int, close_id: int, block_size: int) -> list[int]:
    """Indices of the blocks that lie fully inside a dropped observation (all but the last `keep`)."""
    spans = obs_spans(token_ids, open_id, close_id)
    out: list[int] = []
    for a, b in spans[: max(0, len(spans) - keep)]:
        first = -(-a // block_size)          # first block starting at or after a
        last = b // block_size               # blocks [first, last) end at or before b
        out.extend(range(first, last))
    return out


def dead_token_mask(token_ids, keep: int, open_id: int, close_id: int, block_size: int) -> list[bool]:
    """Per-token version of dead_blocks (used by the reference test)."""
    mask = [False] * len(token_ids)
    for blk in dead_blocks(token_ids, keep, open_id, close_id, block_size):
        for j in range(blk * block_size, min((blk + 1) * block_size, len(token_ids))):
            mask[j] = True
    return mask
