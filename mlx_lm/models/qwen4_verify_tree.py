# Copyright © 2026 Apple Inc.

"""Shallow-wide MTP tree proposer + verifier for the Flash-Next VERIFY path.

The Qwen3-Next / Qwen4 MTP head is structurally depth-1: it predicts token
``p+2`` from the trunk hidden at ``p`` and the committed token ``p+1``
(``Model.mtp_step``).  So the speculation tree this module builds is
SHALLOW-WIDE: breadth ``b`` in {2,3,4} candidates, depth ``d`` in {1,2}
(depth-2 is mildly out-of-distribution for the head, so it is optional and
parameterised).

Two halves, split so the topology and the acceptance walk are pure and
CPU-testable with no model and no Metal:

  * ``assemble_mtp_tree`` / ``accept_tree`` / ``tree_control_entries`` are
    pure numpy.  They own the node order, the ancestry mask, the per-node
    absolute positions, and the lossless greedy-target acceptance walk.
  * ``build_mtp_tree`` is the thin model-facing wrapper.  It runs the MTP head
    to gather the per-parent candidate tokens, then calls ``assemble_mtp_tree``.

Node order is deterministic BFS: node 0 is the committed root, then the ``b``
depth-1 children left-to-right, then (depth-2) the ``b`` grandchildren of each
depth-1 node, grouped by parent in parent order.  Within a parent, candidates
are ranked by descending MTP logit, ties broken by ascending token id.

Per-node positions and the verify control block
-----------------------------------------------
Every node's absolute (trunk) position is ``root_pos + depth_of_node``.
Siblings therefore SHARE a position -- that is the tree-attention RoPE trick:
the ``b`` depth-1 nodes all sit at ``root_pos + 1``, the grandchildren at
``root_pos + 2``.  This is the one number that must reach the megakernel verify
slab correctly.  ``qwen4_megakernel_runtime.build_control_block`` lays out one
per-query block where "query ``m`` sits at position ``position + m``" -- a
LINEAR chain.  For a tree that linear rule is wrong: the control block must use
``positions[node]`` per query instead of ``position + m``, so each query's
``tail_block = pos // block_size``, ``logical_len = pos + 1`` and closed-block
count are its own.  ``tree_control_entries`` reproduces exactly that arithmetic
per node (the 2026-09-03 tail-block trap: the open tail block holds the newest
1..block_size-1 positions, including the query's own, and must be attended).
See the module report for the two structural gaps in the current slab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MTPTree:
    """A shallow-wide speculation tree over committed + drafted tokens.

    All arrays are indexed by node, in the deterministic BFS order described in
    the module docstring.  Node 0 is the committed root.
    """

    tokens: Tuple[int, ...]           # [n] candidate token id per node
    parent: Tuple[int, ...]           # [n] parent node index, parent[0] == -1
    depth: Tuple[int, ...]            # [n] tree depth, root == 0
    positions: Tuple[int, ...]        # [n] absolute trunk pos = root_pos + depth
    children: Tuple[Tuple[int, ...], ...]  # [n] child node indices per node
    tree_mask: np.ndarray             # bool [n, n], ancestor-or-self causal
    breadth: int
    max_depth: int
    root_pos: int

    @property
    def num_nodes(self) -> int:
        return len(self.tokens)

    def ancestors(self, node: int) -> List[int]:
        """Node indices from ``node`` up to the root, inclusive, deepest first."""
        chain: List[int] = []
        j = node
        while j != -1:
            chain.append(j)
            j = self.parent[j]
        return chain


def assemble_mtp_tree(
    *,
    root_token: int,
    root_pos: int,
    breadth: int,
    depth: int,
    depth1_tokens: Sequence[int],
    depth2_tokens: Optional[Sequence[Sequence[int]]] = None,
) -> MTPTree:
    """Build the tree topology, mask and positions from candidate tokens.

    ``depth1_tokens`` is the breadth-``b`` candidate list for the root, already
    ranked (descending MTP logit, ties by ascending id).  For depth 2,
    ``depth2_tokens[i]`` is the ranked breadth-``b`` list for depth-1 node ``i``
    (in node order).  Pure: no model, no mlx, no Metal.
    """
    if breadth < 1:
        raise ValueError("breadth must be >= 1")
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2 (the MTP head is depth-1)")
    if len(depth1_tokens) != breadth:
        raise ValueError(
            f"depth1_tokens must hold exactly breadth={breadth} tokens"
        )
    if depth == 2:
        if depth2_tokens is None or len(depth2_tokens) != breadth:
            raise ValueError(
                "depth 2 needs depth2_tokens with one breadth-list per depth-1 "
                "node"
            )
        if any(len(row) != breadth for row in depth2_tokens):
            raise ValueError("each depth2_tokens row must hold breadth tokens")

    tokens: List[int] = [int(root_token)]
    parent: List[int] = [-1]
    node_depth: List[int] = [0]

    # Level 1: the root's b children (siblings share position root_pos + 1).
    level1: List[int] = []
    for tok in depth1_tokens:
        tokens.append(int(tok))
        parent.append(0)
        node_depth.append(1)
        level1.append(len(tokens) - 1)

    # Level 2: b grandchildren per depth-1 node, grouped by parent in order.
    if depth == 2:
        for i, node_i in enumerate(level1):
            for tok in depth2_tokens[i]:
                tokens.append(int(tok))
                parent.append(node_i)
                node_depth.append(2)

    n = len(tokens)
    positions = [root_pos + node_depth[k] for k in range(n)]

    children: List[List[int]] = [[] for _ in range(n)]
    for k in range(1, n):
        children[parent[k]].append(k)

    # Ancestor-or-self causal mask: node attends only its ancestor chain + self.
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        j = i
        while j != -1:
            mask[i, j] = True
            j = parent[j]

    return MTPTree(
        tokens=tuple(tokens),
        parent=tuple(parent),
        depth=tuple(node_depth),
        positions=tuple(positions),
        children=tuple(tuple(c) for c in children),
        tree_mask=mask,
        breadth=int(breadth),
        max_depth=int(depth),
        root_pos=int(root_pos),
    )


def accept_tree(
    tree: MTPTree, target_logits_per_node: Any
) -> Tuple[Tuple[int, ...], int, int]:
    """Lossless longest-accepted-path walk under greedy target semantics.

    ``target_logits_per_node`` is ``[num_nodes, V]``: row ``k`` is the target's
    logits for the token that FOLLOWS node ``k``.  Walk from the root; at each
    node take the greedy target token ``argmax`` and descend into the child
    that holds it, accepting while such a child exists.

    Returns ``(accepted_token_ids, num_accepted, next_seed_node)`` where the
    accepted ids are the drafted tokens confirmed correct (root excluded), and
    ``next_seed_node`` is the deepest accepted node (or 0).  The always-correct
    correction/bonus token is ``argmax(target_logits[next_seed_node])`` -- the
    caller commits it and seeds the next MTP round from ``next_seed_node``'s
    hidden.  ``argmax`` ties resolve to the lowest index, matching ``mx.argmax``.
    """
    logits = np.asarray(target_logits_per_node)
    if logits.ndim != 2 or logits.shape[0] != tree.num_nodes:
        raise ValueError("target_logits_per_node must be [num_nodes, vocab]")

    node = 0
    accepted: List[int] = []
    while True:
        greedy = int(np.argmax(logits[node]))
        match = -1
        for c in tree.children[node]:
            if tree.tokens[c] == greedy:
                match = c
                break
        if match == -1:
            break
        accepted.append(tree.tokens[match])
        node = match
    return tuple(accepted), len(accepted), node


def tree_control_entries(tree: MTPTree, block_size: int) -> List[dict]:
    """Per-node control-block arithmetic for the megakernel verify slab.

    Mirrors ``qwen4_megakernel_runtime.build_control_block`` but keyed on the
    tree's per-node absolute positions instead of the linear ``position + m``.
    ``block_size`` is the compression ratio ``IDX_COMPRESS`` (4).  Each entry is
    what query ``node`` must carry so its own tail block is scored -- siblings
    share a position and therefore a tail block, while depth advances it.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    entries: List[dict] = []
    for node in range(tree.num_nodes):
        pos = tree.positions[node]
        length = pos + 1
        n_blocks = length // block_size          # closed blocks only
        has_tail = int(length % block_size != 0)  # open tail holds newest tokens
        entries.append(
            {
                "node": node,
                "q_pos": pos,
                "rope_pos": pos,
                "kv_slot": pos,
                "logical_len": length,
                "n_blocks": n_blocks,
                "tail_block": pos // block_size,
                "has_tail": has_tail,
                "complete": n_blocks * block_size,
            }
        )
    return entries


# --------------------------------------------------------------------------
# Model-facing wrapper.  Not exercised by the CPU test (it drives the MTP head
# and touches mlx/Metal); the topology it produces is the pure path above.
# --------------------------------------------------------------------------

def _ranked_top_b(logits_row: "np.ndarray", b: int) -> List[int]:
    """Top-``b`` token ids: descending logit, ties by ascending token id."""
    row = np.asarray(logits_row).reshape(-1)
    order = np.lexsort((np.arange(row.shape[0]), -row))
    return order[:b].astype(np.int64).tolist()


def _clone_kv(cache: Sequence[Any]) -> List[Any]:
    """Deep copy supported MTP caches without dropping QSA side state."""
    import mlx.core as mx  # local import keeps the pure path Metal-free

    from .cache import KVCache
    from .qwen4_exp import QSAKVCache

    out: List[Any] = []
    for src in cache:
        if isinstance(src, QSAKVCache):
            if (getattr(src, "_mtp_share_topk", False)
                    or getattr(src, "_mtp_shared_topk", None) is not None):
                raise RuntimeError("cannot clone an armed QSA MTP cache")
            identity = getattr(src, "_qsa_summary_identity", None)
            if identity is not None:
                identity = dict(identity)
                identity["complete_blocks"] = 0
            c = QSAKVCache(identity)
        elif type(src) is KVCache:
            c = KVCache()
        else:
            raise TypeError(
                f"MTP tree cache clone does not support {type(src).__name__}"
            )
        if src.offset > 0:
            c.keys = mx.contiguous(src.keys[..., : src.offset, :])
            c.values = mx.contiguous(src.values[..., : src.offset, :])
            c.offset = src.offset
        if isinstance(src, QSAKVCache):
            index_keys = src.index_keys
            if src.offset and index_keys is None:
                raise RuntimeError(
                    "populated QSA cache has no raw-key ledger"
                )
            if index_keys is not None:
                if index_keys.shape[1] < src.offset:
                    raise RuntimeError(
                        "QSA raw-key ledger is shorter than its KV cursor"
                    )
                c.index_keys = mx.contiguous(index_keys[:, : src.offset])
            pooled = getattr(src, "_qsa_pooled_keys", None)
            ratio = getattr(src, "_qsa_pooled_ratio", None)
            if pooled is not None:
                if not ratio:
                    raise RuntimeError("QSA pooled keys have no compression ratio")
                complete = min(src.offset // ratio, pooled.shape[1])
                if complete:
                    c._qsa_pooled_keys = mx.contiguous(pooled[:, :complete])
                    c._qsa_pooled_ratio = ratio
                    if c._qsa_summary_identity is not None:
                        c._qsa_summary_identity["complete_blocks"] = complete
        out.append(c)
    return out


def _tile_kv(cache: Sequence[Any], b: int) -> List[Any]:
    """Merge each single-sequence cache into a native B=b cache."""
    if b < 1:
        raise ValueError("cache tile breadth must be positive")

    out: List[Any] = []
    for src in cache:
        merge = getattr(type(src), "merge", None)
        if merge is None:
            raise TypeError(
                f"MTP tree cache tile does not support {type(src).__name__}"
            )
        out.append(merge([src] * b))
    return out


def build_mtp_tree(
    model: Any,
    seed_hidden: Any,
    cur_token: int,
    mtp_cache: Sequence[Any],
    breadth: int,
    depth: int,
    *,
    root_pos: Optional[int] = None,
) -> MTPTree:
    """Run the MTP head to propose a shallow-wide tree.

    ``seed_hidden`` is the trunk post-norm hidden ``[H]`` / ``[1,1,H]`` at the
    last committed position; ``cur_token`` is the committed token ``p+1``.  The
    head predicts ``p+2``; its top-``breadth`` tokens are the depth-1 children.
    For depth 2, each depth-1 token is re-fed (on a tiled cache copy) to
    predict its own top-``breadth`` grandchildren.

    ``root_pos`` is the committed root's absolute TRUNK position (feeds the
    verify control block).  It defaults to ``mtp_cache[0].offset`` but the
    caller should pass the true trunk offset -- the MTP cache offset need not
    equal it.  The caller's ``mtp_cache`` is not mutated: a clone is used.
    """
    import mlx.core as mx

    if breadth < 1 or depth not in (1, 2):
        raise ValueError("breadth >= 1 and depth in {1,2} required")

    if root_pos is None:
        root_pos = int(mtp_cache[0].offset) if len(mtp_cache) else 0

    hidden = mx.array(seed_hidden).reshape(1, 1, -1)
    tok0 = mx.array([[int(cur_token)]])

    work = _clone_kv(mtp_cache)
    logits0, post0 = model.mtp_step(hidden, tok0, work)
    mx.eval(logits0, post0)
    depth1 = _ranked_top_b(np.asarray(logits0[0, -1]), breadth)

    depth2: Optional[List[List[int]]] = None
    if depth == 2:
        # Branch the MTP cache to B=breadth and predict all grandchildren in
        # one batched forward: each depth-1 token seeds its own row from post0.
        tiled = _tile_kv(work, breadth)
        h_b = mx.repeat(post0.reshape(1, 1, -1), breadth, axis=0)  # [b,1,H]
        t_b = mx.array(depth1).reshape(breadth, 1)                 # [b,1]
        logits1, _ = model.mtp_step(h_b, t_b, tiled)
        mx.eval(logits1)
        depth2 = [
            _ranked_top_b(np.asarray(logits1[i, -1]), breadth)
            for i in range(breadth)
        ]

    return assemble_mtp_tree(
        root_token=int(cur_token),
        root_pos=int(root_pos),
        breadth=breadth,
        depth=depth,
        depth1_tokens=depth1,
        depth2_tokens=depth2,
    )
