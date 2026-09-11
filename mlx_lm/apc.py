"""Automatic prefix caching shared by MLX language-model serving paths.

This module gives the existing radix-backed prompt cache a model-independent
APC interface.  Cache topology remains owned by ``model.make_cache()``: plain
KV, rotating/full hybrids (Laguna, North, and Gemma/Muse text backbones), and
checkpointed recurrent hybrids all use the same lookup and storage policy.

The implementation stores whole prompt-cache snapshots.  The radix indexes
token sequences; it is not a paged-KV allocator and does not claim block-level
copy-on-write sharing.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, List, Optional

import mlx.core as mx

from .cow_cache import (
    COWCacheStale,
    COWCacheTelemetry,
    COWFrozenPromptCache,
    COWPromptCacheBranch,
    cow_cache_enabled,
    freeze_prompt_cache,
)
from .cache_planes import (
    CompiledScheduleMetadata,
    PLEResidencyHints,
    PromptHostPlane,
)
from .models.cache import (
    _copy_prompt_cache_for_restore,
    _mark_prompt_cache_restored,
    ArraysCache,
    CacheList,
    KVCache,
    LRUPromptCache,
    PromptTrie,
    RotatingKVCache,
    can_trim_prompt_cache,
)

try:
    from .models.cache import achievable_trim
except ImportError:  # Older mlx-lm runtimes can still use backend adapters.
    achievable_trim = None


@dataclass(frozen=True)
class APCKey:
    """Fail-closed identity for a reusable prompt-cache namespace.

    ``semantic_fingerprint`` is where multimodal callers bind image/audio or
    other non-token inputs.  It must change whenever equal token IDs could
    produce different model state.
    """

    model: Hashable
    revision: Optional[Hashable] = None
    adapter: Optional[Hashable] = None
    tokenizer_fingerprint: Optional[Hashable] = None
    cache_layout_fingerprint: Optional[Hashable] = None
    semantic_fingerprint: Optional[Hashable] = None


@dataclass(frozen=True)
class APCCapabilities:
    topology: str
    exact_prefix: bool
    arbitrary_branch: bool
    reason: Optional[str] = None
    stored: Optional[bool] = None
    native: Any = None


@dataclass
class APCLookup:
    cache: Optional[List[Any]]
    remaining_tokens: List[int]
    cached_tokens: int
    hit: bool
    hit_kind: Optional[str]
    miss_reason: Optional[str]
    native: Any = None
    sidecar: Any = None
    prep_telemetry: Any = None
    prompt_host: Optional[PromptHostPlane] = None


@dataclass
class MTPAPCSidecar:
    """Persistent draft state captured at an exact target-cache boundary.

    ``rng_key``/``rng_draws`` carry the decode lane's position in its own
    random stream, so a resumed request continues that stream instead of
    repeating draws it already made. They stay ``None``/``0`` for a greedy or
    keyless lane, which draws nothing.
    """

    state: Any
    covered_tokens: int
    rng_key: Optional[Any] = None
    rng_draws: int = 0

    @property
    def nbytes(self) -> int:
        mtp_cache, tail_hidden = self.state
        cache_bytes = sum(
            int(getattr(entry, "nbytes", 0))
            for entry in _walk_cache_entries(mtp_cache)
        )
        hidden_bytes = int(getattr(tail_hidden, "nbytes", 0))
        return cache_bytes + hidden_bytes


def _walk_cache_entries(prompt_cache: Iterable[Any]):
    for entry in prompt_cache:
        if isinstance(entry, CacheList):
            yield from _walk_cache_entries(entry.caches)
        elif isinstance(entry, (list, tuple)):
            yield from _walk_cache_entries(entry)
        else:
            yield entry


def _iter_trie_entries(trie: PromptTrie):
    """Yield every stored cache entry, whatever the LRU bookkeeping says."""
    stack = [trie._trie]
    while stack:
        node = stack.pop()
        for token, child in node.items():
            if token == "__value__":
                yield child
            else:
                stack.append(child)


def inspect_apc_capabilities(prompt_cache: List[Any]) -> APCCapabilities:
    """Describe which lossless APC operations a concrete cache supports."""

    leaves = list(_walk_cache_entries(prompt_cache))
    if not leaves:
        return APCCapabilities("empty", False, False, "empty_cache")

    has_rotating = any(isinstance(c, RotatingKVCache) for c in leaves)
    has_full = any(isinstance(c, KVCache) for c in leaves)
    has_state = any(isinstance(c, ArraysCache) for c in leaves)
    known = all(
        isinstance(c, (KVCache, RotatingKVCache, ArraysCache))
        or hasattr(c, "state")
        for c in leaves
    )

    if has_state:
        topology = "checkpointed_hybrid"
    elif has_rotating and has_full:
        topology = "mixed_rotating_kv"
    elif has_rotating:
        topology = "rotating_kv"
    elif has_full:
        topology = "kv"
    else:
        topology = "custom"

    arbitrary_branch = can_trim_prompt_cache(prompt_cache)
    if not arbitrary_branch and achievable_trim is not None:
        # Checkpoint-aware hybrids can still branch at recorded positions.
        arbitrary_branch = achievable_trim(prompt_cache, 1) is not None

    return APCCapabilities(
        topology=topology,
        exact_prefix=known,
        arbitrary_branch=arbitrary_branch,
        reason=None if known else "unsupported_cache_entry",
    )


class AutomaticPrefixCache(LRUPromptCache):
    """Shared radix APC for standard and model-defined MLX cache topologies.

    The legacy ``fetch_nearest_cache`` / ``insert_cache`` methods remain
    available, so this is a drop-in replacement for ``LRUPromptCache``.  New
    callers should use ``lookup`` / ``store`` for explicit hit telemetry.
    """

    _STAT_KEYS = ("lookups", "hits", "misses", "cached_tokens", "stores")

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        *,
        cow_branching: Optional[bool] = None,
    ):
        super().__init__(max_size=max_size, max_bytes=max_bytes)
        self._apc_lock = threading.RLock()
        self._cow_branching = cow_cache_enabled(cow_branching)
        self._cow_telemetry = COWCacheTelemetry()
        self._apc_stats = {key: 0 for key in self._STAT_KEYS}
        # Totals for the whole process. ``_apc_stats`` counts only the entries
        # the live cache could still serve, so it restarts at every clear.
        self._apc_lifetime = {key: 0 for key in self._STAT_KEYS}
        self._apc_clears = 0

    @staticmethod
    def key(
        model: Hashable,
        *,
        revision: Optional[Hashable] = None,
        adapter: Optional[Hashable] = None,
        tokenizer_fingerprint: Optional[Hashable] = None,
        cache_layout_fingerprint: Optional[Hashable] = None,
        semantic_fingerprint: Optional[Hashable] = None,
    ) -> APCKey:
        return APCKey(
            model=model,
            revision=revision,
            adapter=adapter,
            tokenizer_fingerprint=tokenizer_fingerprint,
            cache_layout_fingerprint=cache_layout_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )

    def lookup(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        # Search, candidate selection, restoration, and hit accounting must
        # observe one trie generation. clear()/trim_to()/store() use the same
        # lock, so a live reader gets either the old hit or the new miss.
        with self._apc_lock:
            return self._lookup_locked(key, tokens)

    def _lookup_locked(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        tokens = [int(token) for token in tokens]
        trie_result = self._trie.search(key, tokens)
        # A target cache may only compose with an MTP sidecar at the exact
        # boundary jointly captured by the two states. Prefer the deepest such
        # candidate whose stored token path still matches through that
        # boundary. Do not trim either cache: the uncached token tail starts at
        # ``covered_tokens`` and teacher-forces forward from there.
        sidecar_candidates = []
        for path, common in (
            (trie_result.exact, len(tokens)),
            (trie_result.longer, trie_result.common_prefix),
            (
                trie_result.shorter,
                len(trie_result.shorter)
                if trie_result.shorter is not None
                else 0,
            ),
        ):
            if path is None:
                continue
            entry = self._trie.get(trie_result.model, path)
            sidecar = getattr(entry, "sidecar", None)
            covered = int(getattr(sidecar, "covered_tokens", 0))
            if (
                sidecar is not None
                and 0 < covered < len(tokens)
                and common >= covered
            ):
                cache_offset = max(
                    (
                        getattr(c, "offset", 0)
                        for c in _walk_cache_entries(entry.prompt_cache)
                    ),
                    default=0,
                )
                if cache_offset == covered:
                    sidecar_candidates.append((covered, entry, sidecar))
        if sidecar_candidates:
            covered, entry, sidecar = max(
                sidecar_candidates, key=lambda item: item[0]
            )
            try:
                restored_cache = _copy_prompt_cache_for_restore(
                    entry.prompt_cache
                )
            except COWCacheStale:
                self._apc_stats["lookups"] += 1
                self._apc_stats["misses"] += 1
                return APCLookup(
                    None,
                    tokens,
                    0,
                    False,
                    None,
                    "stale_cow_generation",
                )
            self._apc_stats["lookups"] += 1
            self._apc_stats["hits"] += 1
            self._apc_stats["cached_tokens"] += covered
            restored_sidecar = getattr(restored_cache, "cow_sidecar", None)
            if (
                isinstance(restored_cache, COWPromptCacheBranch)
                and restored_sidecar is None
            ):
                # The target plane remains reusable when only the MTP plane
                # was invalidated. Drop this provisional sidecar-path branch
                # and retry below through the ordinary target-only lookup.
                restored_cache.close()
            else:
                if restored_sidecar is None:
                    restored_sidecar = copy.deepcopy(sidecar)
                _mark_prompt_cache_restored(restored_sidecar.state[0])
                return APCLookup(
                    restored_cache,
                    tokens[covered:],
                    covered,
                    True,
                    "mtp_sidecar",
                    None,
                    sidecar=restored_sidecar,
                    prep_telemetry=getattr(
                        restored_cache, "cow_prep_telemetry", None
                    ),
                    prompt_host=getattr(
                        getattr(restored_cache, "cow_metadata", None),
                        "prompt_host",
                        None,
                    ),
                )
        try:
            cache, remaining = super().fetch_nearest_cache(key, tokens)
        except COWCacheStale:
            cache, remaining = None, tokens
            stale_generation = True
        else:
            stale_generation = False
        cached_tokens = len(tokens) - len(remaining) if cache is not None else 0
        hit = cache is not None and cached_tokens > 0

        self._apc_stats["lookups"] += 1
        self._apc_stats["hits" if hit else "misses"] += 1
        self._apc_stats["cached_tokens"] += cached_tokens

        if not hit:
            kind = None
            short_length = (
                len(trie_result.shorter) if trie_result.shorter is not None else 0
            )
            has_unusable_branch = trie_result.exact is not None or (
                trie_result.longer is not None
                and trie_result.common_prefix > short_length
            )
            if stale_generation:
                reason = "stale_cow_generation"
            else:
                reason = (
                    "untrimmable_branch"
                    if has_unusable_branch
                    else "no_compatible_prefix"
                )
        elif trie_result.exact is not None:
            kind = "exact"
            reason = None
        else:
            kind = "prefix"
            reason = None
        return APCLookup(
            cache,
            remaining,
            cached_tokens,
            hit,
            kind,
            reason,
            prep_telemetry=getattr(cache, "cow_prep_telemetry", None),
            prompt_host=getattr(
                getattr(cache, "cow_metadata", None), "prompt_host", None
            ),
        )

    def store(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prompt_host: Optional[PromptHostPlane] = None,
        ple_hints: Optional[PLEResidencyHints] = None,
        compiled_schedule: Optional[CompiledScheduleMetadata] = None,
    ) -> APCCapabilities:
        with self._apc_lock:
            return self._store_locked(
                key,
                tokens,
                prompt_cache,
                cache_type=cache_type,
                sidecar=sidecar,
                prompt_host=prompt_host,
                ple_hints=ple_hints,
                compiled_schedule=compiled_schedule,
            )

    def _store_locked(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prompt_host: Optional[PromptHostPlane] = None,
        ple_hints: Optional[PLEResidencyHints] = None,
        compiled_schedule: Optional[CompiledScheduleMetadata] = None,
    ) -> APCCapabilities:
        tokens = [int(token) for token in tokens]
        capabilities = inspect_apc_capabilities(prompt_cache)
        if not capabilities.exact_prefix:
            return capabilities
        if self._cow_branching:
            try:
                prompt_cache, sidecar = freeze_prompt_cache(
                    prompt_cache,
                    key=key,
                    tokens=tokens,
                    cache_type=cache_type,
                    sidecar=sidecar,
                    prompt_host=prompt_host,
                    ple_hints=ple_hints,
                    compiled_schedule=compiled_schedule,
                    telemetry=self._cow_telemetry,
                )
            except Exception:
                # Product safety is the incumbent behavior. An unsupported
                # cache-local field must not turn a valid store into a serving
                # failure.
                self._cow_telemetry.add("freeze_failures")
        cow_source = (
            prompt_cache
            if isinstance(prompt_cache, COWFrozenPromptCache)
            else None
        )
        before = (
            {id(entry): entry for entry in _iter_trie_entries(self._trie)}
            if self._cow_branching
            else {}
        )
        super().insert_cache(
            key,
            tokens,
            prompt_cache,
            cache_type=cache_type,
            sidecar=sidecar,
        )
        if before or cow_source is not None:
            live_entries = list(_iter_trie_entries(self._trie))
            live = {id(entry) for entry in live_entries}
            for ident, entry in before.items():
                if ident not in live and isinstance(
                    entry.prompt_cache, COWFrozenPromptCache
                ):
                    entry.prompt_cache.close()
            if cow_source is not None and not any(
                entry.prompt_cache is cow_source for entry in live_entries
            ):
                cow_source.close()
        self._apc_stats["stores"] += 1
        return capabilities

    def fetch_nearest_cache(self, model: Hashable, tokens: List[int]):
        result = self.lookup(model, tokens)
        return result.cache, result.remaining_tokens

    def insert_cache(
        self,
        model: Hashable,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prompt_host: Optional[PromptHostPlane] = None,
        ple_hints: Optional[PLEResidencyHints] = None,
        compiled_schedule: Optional[CompiledScheduleMetadata] = None,
    ):
        return self.store(
            model,
            tokens,
            prompt_cache,
            cache_type=cache_type,
            sidecar=sidecar,
            prompt_host=prompt_host,
            ple_hints=ple_hints,
            compiled_schedule=compiled_schedule,
        )

    def clear(self, *, release_memory: bool = True) -> dict:
        """Drop every stored prefix, its MTP sidecar, and its bytes.

        A serving lever can change what a prefix *means*, not only how stale
        it is: the same tokens under two settings give different state, and
        some settings change the cache layout itself.  So entries are dropped,
        never marked stale, and the sidecars go with them.

        Safe to call on a live server.  The new trie and LRU are built first
        and then rebound, so a concurrent reader sees either the old cache or
        the empty one and never a half-emptied one.  It does not stop a
        request that is already generating from storing its own result
        afterwards; drain first when that matters.
        """
        with self._apc_lock:
            return self._clear_locked(release_memory=release_memory)

    def _clear_locked(self, *, release_memory: bool = True) -> dict:
        entries = list(_iter_trie_entries(self._trie))
        report = {
            "entries": len(entries),
            "sidecars": sum(
                1 for entry in entries if getattr(entry, "sidecar", None) is not None
            ),
            "bytes": int(self._n_bytes),
        }

        fresh_trie = PromptTrie()
        fresh_lru = LRUPromptCache.CacheOrder(list(self._lru._ordering))
        self._trie = fresh_trie
        self._lru = fresh_lru
        self._n_bytes = 0
        self._n_bytes_by_type = {key: 0 for key in fresh_lru._ordering}

        # Invalidate COW generations before detaching their source lists. Live
        # branches pin descriptors until explicit or GC release.
        for entry in entries:
            if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                entry.prompt_cache.close()

        # Release the arrays the detached entries still hold. Rebind, never
        # mutate in place: a caller may hold the same list.
        for entry in entries:
            entry.prompt_cache = []
            entry.sidecar = None
            entry.nbytes = 0
        entries.clear()

        for key in self._STAT_KEYS:
            self._apc_lifetime[key] += self._apc_stats[key]
            self._apc_stats[key] = 0
        self._apc_clears += 1

        if release_memory:
            mx.clear_cache()
        return report

    @property
    def apc_stats(self):
        with self._apc_lock:
            stats = dict(self._apc_stats)
            stats["clears"] = self._apc_clears
            stats["lifetime"] = dict(self._apc_lifetime)
            for key in self._STAT_KEYS:
                stats["lifetime"][key] += self._apc_stats[key]
            stats["cow_enabled"] = self._cow_branching
            stats["cow"] = self._cow_telemetry.snapshot()
            return stats

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        with self._apc_lock:
            return self._trim_to_locked(
                n_sequences=n_sequences, n_bytes=n_bytes
            )

    def _trim_to_locked(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        before = {id(entry): entry for entry in _iter_trie_entries(self._trie)}
        super().trim_to(n_sequences=n_sequences, n_bytes=n_bytes)
        live = {id(entry) for entry in _iter_trie_entries(self._trie)}
        for ident, entry in before.items():
            if ident not in live and isinstance(
                entry.prompt_cache, COWFrozenPromptCache
            ):
                entry.prompt_cache.close()


# Short public spelling for server integrations.
APC = AutomaticPrefixCache


__all__ = [
    "APC",
    "APCCapabilities",
    "APCKey",
    "APCLookup",
    "AutomaticPrefixCache",
    "MTPAPCSidecar",
    "inspect_apc_capabilities",
]
