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
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, List, Optional

import mlx.core as mx

from .cache_capsule import CacheCapsuleGeneration
from .cache_planes import (
    CompiledScheduleMetadata,
    PLEResidencyHints,
    PromptHostPlane,
)
from .cow_cache import (
    COWCacheStale,
    COWCacheTelemetry,
    COWFrozenPromptCache,
    COWPromptCacheBranch,
    cow_cache_enabled,
    freeze_prompt_cache,
)
from .models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    LRUPromptCache,
    PromptTrie,
    RotatingKVCache,
    _copy_prompt_cache_for_restore,
    _mark_prompt_cache_restored,
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
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
    capsule_generation: Optional[int] = None
    segment_manifest: Any = None


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
    _DISK_STAT_KEYS = (
        "idle_spills",
        "pressure_spills",
        "restores",
        "restore_failures",
        "spill_failures",
        "disk_evictions",
        "bytes_written",
        "bytes_read",
    )

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        max_tokens: Optional[int] = None,
        *,
        cow_branching: Optional[bool] = None,
        idle_disk_seconds: float = 0.0,
        idle_disk_dir: Optional[str] = None,
        idle_disk_max_bytes: int = 1 << 63,
        now_fn=time.monotonic,
    ):
        super().__init__(
            max_size=max_size, max_bytes=max_bytes, max_tokens=max_tokens
        )
        self._apc_lock = threading.RLock()
        self._cow_branching = cow_cache_enabled(cow_branching)
        self._cow_telemetry = COWCacheTelemetry()
        self._apc_stats = {key: 0 for key in self._STAT_KEYS}
        # Totals for the whole process. ``_apc_stats`` counts only the entries
        # the live cache could still serve, so it restarts at every clear.
        self._apc_lifetime = {key: 0 for key in self._STAT_KEYS}
        self._apc_clears = 0
        self._capsule_generation = CacheCapsuleGeneration()
        self._idle_disk_seconds = max(0.0, float(idle_disk_seconds))
        if self._idle_disk_seconds > 0 and not idle_disk_dir:
            raise ValueError(
                "idle_disk_dir is required when idle_disk_seconds is enabled"
            )
        self._idle_disk_dir = (
            Path(idle_disk_dir).expanduser().resolve()
            if idle_disk_dir and self._idle_disk_seconds > 0
            else None
        )
        self._idle_disk_max_bytes = max(0, int(idle_disk_max_bytes))
        self._now = now_fn
        self._last_idle_scan = 0.0
        self._disk_stats = {key: 0 for key in self._DISK_STAT_KEYS}
        self._disk_bytes = 0
        if self._idle_disk_dir is not None:
            self._idle_disk_dir.mkdir(parents=True, exist_ok=True)
            # This tier preserves idle state only within one process. A stale
            # file has no live APC identity/compatibility owner, so fail closed
            # across restarts and remove only our namespaced files.
            for path in self._idle_disk_dir.glob("apc-idle-*.safetensors"):
                try:
                    path.unlink()
                except OSError:
                    pass

    @property
    def capsule_generation(self) -> CacheCapsuleGeneration:
        """Generation authority for work captured at an APC lookup boundary."""

        return self._capsule_generation

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

    def _entry_records_locked(self):
        seen = set()
        for cache_type in self._lru._ordering:
            for key, tokens in tuple(self._lru._lrus[cache_type]):
                entry = self._trie.get(key, tokens)
                if entry is not None and id(entry) not in seen:
                    seen.add(id(entry))
                    yield key, list(tokens), entry

    @staticmethod
    def _entry_pinned(entry) -> bool:
        cache = entry.prompt_cache
        if not isinstance(cache, COWFrozenPromptCache):
            return False
        return int(getattr(cache.cow_owner, "pin_count", 0) or 0) > 0

    @staticmethod
    def _atomic_save_cache(path: Path, cache: List[Any]) -> None:
        temporary = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors"
        )
        try:
            save_prompt_cache(str(temporary), list(cache))
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _atomic_save_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
        temporary = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors"
        )
        try:
            mx.save_safetensors(str(temporary), arrays)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _disk_paths(entry) -> tuple[Path, ...]:
        disk = getattr(entry, "_apc_disk", None) or {}
        return tuple(
            Path(path)
            for path in (
                disk.get("target"),
                disk.get("draft"),
                disk.get("aux"),
            )
            if path
        )

    def _remove_disk_files_locked(self, entry) -> None:
        for path in self._disk_paths(entry):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            try:
                path.unlink()
            except OSError:
                continue
            self._disk_bytes = max(0, self._disk_bytes - int(size))
        entry._apc_disk = None

    def _drop_entry_locked(self, key, tokens, entry) -> None:
        current = self._trie.pop(key, tokens)
        if current is None:
            return
        self._lru.remove(key, tokens)
        self._n_bytes = max(0, self._n_bytes - int(current.nbytes))
        self._n_bytes_by_type[current.cache_type] = max(
            0,
            self._n_bytes_by_type[current.cache_type] - int(current.nbytes),
        )
        if isinstance(current.prompt_cache, COWFrozenPromptCache):
            current.prompt_cache.close()
        self._remove_disk_files_locked(current)
        current.prompt_cache = []
        current.sidecar = None
        current.nbytes = 0
        self._capsule_generation.advance()

    def _spill_entry_locked(self, key, tokens, entry, *, reason: str) -> bool:
        if self._idle_disk_dir is None or not entry.prompt_cache:
            return False
        if self._entry_pinned(entry):
            return False

        disk = getattr(entry, "_apc_disk", None)
        if not disk:
            stem = f"apc-idle-{uuid.uuid4().hex}"
            target = self._idle_disk_dir / f"{stem}.target.safetensors"
            draft = self._idle_disk_dir / f"{stem}.draft.safetensors"
            aux = self._idle_disk_dir / f"{stem}.aux.safetensors"
            created = []
            try:
                self._atomic_save_cache(target, entry.prompt_cache)
                created.append(target)
                sidecar = entry.sidecar
                sidecar_info = None
                if sidecar is not None:
                    draft_cache, tail_hidden = sidecar.state
                    self._atomic_save_cache(draft, draft_cache)
                    created.append(draft)
                    arrays = {}
                    if tail_hidden is not None:
                        arrays["tail_hidden"] = tail_hidden
                    if sidecar.rng_key is not None:
                        arrays["rng_key"] = sidecar.rng_key
                    if arrays:
                        self._atomic_save_arrays(aux, arrays)
                        created.append(aux)
                    sidecar_info = {
                        "covered_tokens": int(sidecar.covered_tokens),
                        "rng_draws": int(sidecar.rng_draws),
                    }
                metadata = getattr(
                    getattr(entry.prompt_cache, "cow_owner", None),
                    "metadata",
                    None,
                )
                disk = {
                    "target": str(target),
                    "draft": str(draft) if draft in created else None,
                    "aux": str(aux) if aux in created else None,
                    "sidecar": sidecar_info,
                    "cow_metadata": metadata,
                    "resident_nbytes": int(entry.nbytes),
                }
                entry._apc_disk = disk
                written = sum(path.stat().st_size for path in created)
                self._disk_bytes += int(written)
                self._disk_stats["bytes_written"] += int(written)
            except Exception:
                self._disk_stats["spill_failures"] += 1
                for path in created:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                return False

        resident_nbytes = int(entry.nbytes)
        if isinstance(entry.prompt_cache, COWFrozenPromptCache):
            entry.prompt_cache.close()
        entry.prompt_cache = []
        entry.sidecar = None
        entry.nbytes = 0
        self._n_bytes = max(0, self._n_bytes - resident_nbytes)
        self._n_bytes_by_type[entry.cache_type] = max(
            0, self._n_bytes_by_type[entry.cache_type] - resident_nbytes
        )
        self._disk_stats[
            "idle_spills" if reason == "idle" else "pressure_spills"
        ] += 1
        self._capsule_generation.advance()
        return True

    def _restore_entry_locked(self, key, tokens, entry) -> bool:
        disk = getattr(entry, "_apc_disk", None) or {}
        target = disk.get("target")
        if not target:
            return False
        try:
            # Make room before materializing file-backed arrays. The requested
            # entry may temporarily exceed the resident ceiling by itself, but
            # unrelated idle sources should not crowd its restore.
            expected = int(disk.get("resident_nbytes", 0) or 0)
            original_limit = self.max_bytes
            self.max_bytes = max(0, original_limit - expected)
            try:
                self._spill_resident_budget_locked()
            finally:
                self.max_bytes = original_limit
            cache = load_prompt_cache(target)
            sidecar = None
            sidecar_info = disk.get("sidecar")
            if sidecar_info is not None:
                draft = load_prompt_cache(disk["draft"])
                arrays = mx.load(disk["aux"]) if disk.get("aux") else {}
                sidecar = MTPAPCSidecar(
                    (draft, arrays.get("tail_hidden")),
                    covered_tokens=int(sidecar_info["covered_tokens"]),
                    rng_key=arrays.get("rng_key"),
                    rng_draws=int(sidecar_info.get("rng_draws", 0)),
                )
            mx.eval([item.state for item in cache])
            if sidecar is not None:
                mx.eval(
                    [item.state for item in sidecar.state[0]],
                    *(
                        [sidecar.state[1]]
                        if sidecar.state[1] is not None
                        else []
                    ),
                    *([sidecar.rng_key] if sidecar.rng_key is not None else []),
                )
            metadata = disk.get("cow_metadata")
            if self._cow_branching:
                cache, sidecar = freeze_prompt_cache(
                    cache,
                    key=key,
                    tokens=tokens,
                    cache_type=entry.cache_type,
                    sidecar=sidecar,
                    prompt_host=getattr(metadata, "prompt_host", None),
                    ple_hints=getattr(metadata, "ple_hints", None),
                    compiled_schedule=getattr(metadata, "compiled_schedule", None),
                    telemetry=self._cow_telemetry,
                    layer_segments=getattr(self, "_layer_segments", False),
                )
            entry.prompt_cache = cache
            entry.sidecar = sidecar
            entry.nbytes = sum(int(item.nbytes) for item in cache) + int(
                getattr(sidecar, "nbytes", 0)
            )
            self._n_bytes += int(entry.nbytes)
            self._n_bytes_by_type[entry.cache_type] += int(entry.nbytes)
            self._disk_stats["restores"] += 1
            self._disk_stats["bytes_read"] += sum(
                path.stat().st_size for path in self._disk_paths(entry)
            )
            entry._apc_last_access_at = self._now()
            return True
        except Exception:
            self._disk_stats["restore_failures"] += 1
            return False

    def _enforce_disk_limit_locked(self) -> None:
        if self._disk_bytes <= self._idle_disk_max_bytes:
            return
        records = sorted(
            (
                (float(getattr(entry, "_apc_last_access_at", 0.0)), key, tokens, entry)
                for key, tokens, entry in self._entry_records_locked()
                if getattr(entry, "_apc_disk", None)
            ),
            key=lambda row: row[0],
        )
        for _last_access, key, tokens, entry in records:
            if self._disk_bytes <= self._idle_disk_max_bytes:
                break
            if entry.prompt_cache:
                self._remove_disk_files_locked(entry)
            else:
                self._drop_entry_locked(key, tokens, entry)
            self._disk_stats["disk_evictions"] += 1

    def _spill_resident_budget_locked(self, *, exclude=None) -> int:
        if self._idle_disk_dir is None or self._n_bytes <= self.max_bytes:
            return 0
        spilled = 0
        while self._n_bytes > self.max_bytes:
            records = sorted(
                (
                    (
                        float(getattr(entry, "_apc_last_access_at", 0.0)),
                        key,
                        tokens,
                        entry,
                    )
                    for key, tokens, entry in self._entry_records_locked()
                    if entry.prompt_cache
                    and not self._entry_pinned(entry)
                    and (entry is not exclude or self._n_bytes == entry.nbytes)
                ),
                key=lambda row: row[0],
            )
            if not records:
                break
            _last_access, key, tokens, entry = records[0]
            if not self._spill_entry_locked(key, tokens, entry, reason="pressure"):
                break
            spilled += 1
        if spilled:
            mx.clear_cache()
            self._enforce_disk_limit_locked()
        return spilled

    def spill_idle_entries(self, *, now: Optional[float] = None) -> int:
        """Move unpinned APC entries idle past the configured age to disk."""

        if self._idle_disk_dir is None or self._idle_disk_seconds <= 0:
            return 0
        now = self._now() if now is None else float(now)
        scan_interval = 1.0
        with self._apc_lock:
            if now - self._last_idle_scan < scan_interval:
                return 0
            self._last_idle_scan = now
            spilled = 0
            for key, tokens, entry in tuple(self._entry_records_locked()):
                last_access = float(getattr(entry, "_apc_last_access_at", now))
                if (
                    entry.prompt_cache
                    and now - last_access >= self._idle_disk_seconds
                    and self._spill_entry_locked(key, tokens, entry, reason="idle")
                ):
                    spilled += 1
            if spilled:
                mx.clear_cache()
                self._enforce_disk_limit_locked()
            return spilled

    def lookup(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        # Search, candidate selection, restoration, and hit accounting must
        # observe one trie generation. clear()/trim_to()/store() use the same
        # lock, so a live reader gets either the old hit or the new miss.
        with self._apc_lock:
            return self._lookup_locked(key, tokens)

    def _lookup_locked(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        tokens = [int(token) for token in tokens]
        # A disk-only entry keeps its radix identity but no device arrays.
        # Hydrate only the exact/nearest candidates before the established APC
        # selection logic inspects their offsets and trim capabilities.
        while True:
            trie_result = self._trie.search(key, tokens)
            retry = False
            seen = set()
            for path in (
                trie_result.exact,
                trie_result.longer,
                trie_result.shorter,
            ):
                if path is None or tuple(path) in seen:
                    continue
                seen.add(tuple(path))
                entry = self._trie.get(trie_result.model, path)
                if entry is None:
                    continue
                if getattr(entry, "_apc_disk", None) and not entry.prompt_cache:
                    if not self._restore_entry_locked(trie_result.model, path, entry):
                        self._drop_entry_locked(trie_result.model, path, entry)
                        retry = True
                        break
                entry._apc_last_access_at = self._now()
            if not retry:
                break
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
                    capsule_generation=self._capsule_generation.current,
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
                    capsule_generation=self._capsule_generation.current,
                    segment_manifest=getattr(
                        restored_cache, "cow_segment_stats", None
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
            capsule_generation=self._capsule_generation.current,
            segment_manifest=getattr(cache, "cow_segment_stats", None),
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
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            self.overlength_rejections += 1
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
                    layer_segments=getattr(self, "_layer_segments", False),
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
            if self._cow_branching or self._idle_disk_dir is not None
            else {}
        )
        resident_limit = self.max_bytes
        if self._idle_disk_dir is not None:
            # Preserve over-budget entries in the disk tier instead of letting
            # the base LRU delete them before APC can serialize them.
            self.max_bytes = 1 << 63
        try:
            super().insert_cache(
                key,
                tokens,
                prompt_cache,
                cache_type=cache_type,
                sidecar=sidecar,
            )
        finally:
            self.max_bytes = resident_limit
        self._capsule_generation.advance()
        if before or cow_source is not None:
            live_entries = list(_iter_trie_entries(self._trie))
            live = {id(entry) for entry in live_entries}
            for ident, entry in before.items():
                if ident not in live:
                    if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                        entry.prompt_cache.close()
                    self._remove_disk_files_locked(entry)
            if cow_source is not None and not any(
                entry.prompt_cache is cow_source for entry in live_entries
            ):
                cow_source.close()
        stored_entry = self._trie.get(key, tokens)
        if stored_entry is not None:
            stored_entry._apc_last_access_at = self._now()
            self._spill_resident_budget_locked(exclude=stored_entry)
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
            self._remove_disk_files_locked(entry)

        # Release the arrays the detached entries still hold. Rebind, never
        # mutate in place: a caller may hold the same list.
        for entry in entries:
            entry.prompt_cache = []
            entry.sidecar = None
            entry.nbytes = 0
        entries.clear()
        if report["entries"]:
            self._capsule_generation.advance()

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
            stats["max_tokens"] = self.max_tokens
            stats["max_entry_tokens"] = self.max_entry_tokens
            stats["overlength_rejections"] = self.overlength_rejections
            stats["idle_disk"] = {
                "enabled": self._idle_disk_dir is not None,
                "idle_seconds": self._idle_disk_seconds,
                "resident_bytes": int(self._n_bytes),
                "disk_bytes": int(self._disk_bytes),
                "disk_max_bytes": int(self._idle_disk_max_bytes),
                "disk_entries": sum(
                    1
                    for entry in _iter_trie_entries(self._trie)
                    if getattr(entry, "_apc_disk", None)
                ),
                **dict(self._disk_stats),
            }
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
        if live != set(before):
            self._capsule_generation.advance()
        for ident, entry in before.items():
            if ident not in live:
                if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                    entry.prompt_cache.close()
                self._remove_disk_files_locked(entry)


class AutomaticPrefixCacheV2(AutomaticPrefixCache):
    """Model-declared APC with atomic layer/segment descriptor ownership."""

    schema_version = 2

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        max_tokens: Optional[int] = None,
        *,
        layout_name: str,
        idle_disk_seconds: float = 0.0,
        idle_disk_dir: Optional[str] = None,
        idle_disk_max_bytes: int = 1 << 63,
        now_fn=time.monotonic,
    ) -> None:
        if not layout_name:
            raise ValueError("APCv2 requires a model cache-layout declaration")
        super().__init__(
            max_size=max_size,
            max_bytes=max_bytes,
            max_tokens=max_tokens,
            cow_branching=True,
            idle_disk_seconds=idle_disk_seconds,
            idle_disk_dir=idle_disk_dir,
            idle_disk_max_bytes=idle_disk_max_bytes,
            now_fn=now_fn,
        )
        self._layer_segments = True
        self.layout_name = str(layout_name)

    @property
    def apc_stats(self):
        with self._apc_lock:
            stats = super().apc_stats
            aggregate = {
                "schema": "apcv2.layer-segments.v1",
                "entries": 0,
                "fallback_entries": 0,
                "layers": 0,
                "plane_layers": 0,
                "segments": 0,
                "logical_bytes": 0,
                "by_plane": {},
            }
            for entry in _iter_trie_entries(self._trie):
                frozen = entry.prompt_cache
                if not isinstance(frozen, COWFrozenPromptCache):
                    aggregate["fallback_entries"] += 1
                    continue
                summary = frozen.cow_owner.segment_stats()
                aggregate["entries"] += 1
                for key in ("layers", "plane_layers", "segments"):
                    aggregate[key] += int(summary.get(key, 0))
                for plane, values in summary.get("by_plane", {}).items():
                    combined = aggregate["by_plane"].setdefault(
                        plane,
                        {
                            "layers": 0,
                            "segments": 0,
                            "logical_bytes": 0,
                            "invalid": 0,
                        },
                    )
                    for key in combined:
                        combined[key] += int(values.get(key, 0))
                    aggregate["logical_bytes"] += int(
                        values.get("logical_bytes", 0)
                    )
            stats["version"] = 2
            stats["layout_name"] = self.layout_name
            stats["layer_segments"] = aggregate
            return stats


# Short public spelling for server integrations.
APC = AutomaticPrefixCache


__all__ = [
    "APC",
    "APCCapabilities",
    "APCKey",
    "APCLookup",
    "AutomaticPrefixCache",
    "AutomaticPrefixCacheV2",
    "MTPAPCSidecar",
    "inspect_apc_capabilities",
]
