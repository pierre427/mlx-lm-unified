"""Adapters from specialized prefix-cache engines to the shared APC contract.

The adapters intentionally retain each backend's native result. Snapshot
callers keep lifecycle and persistence metadata; block callers keep the
ref-counted Muse plan and must release or commit its acquired blocks.
"""

from __future__ import annotations

import json
from typing import Any, Hashable, Iterable, Optional

from .apc import APCCapabilities, APCKey, APCLookup


def _stable_text(value: Optional[Hashable]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _snapshot_identity(key: Hashable) -> dict[str, Optional[str]]:
    if not isinstance(key, APCKey):
        return {
            "model_id": _stable_text(key),
            "tokenizer_fingerprint": None,
            "cache_layout_fingerprint": None,
            "cache_salt": None,
        }
    model_identity = {
        "model": key.model,
        "revision": key.revision,
        "adapter": key.adapter,
    }
    return {
        "model_id": _stable_text(model_identity),
        "tokenizer_fingerprint": _stable_text(key.tokenizer_fingerprint),
        "cache_layout_fingerprint": _stable_text(key.cache_layout_fingerprint),
        "cache_salt": _stable_text(key.semantic_fingerprint),
    }


class SnapshotAPCAdapter:
    """Unified APC facade for ``SnapshotPrefixCache``-style backends."""

    def __init__(self, backend: Any):
        self.backend = backend

    def __getattr__(self, name: str):
        # Preserve prepare_cache, prewarm, session, manifest, and telemetry APIs.
        return getattr(self.backend, name)

    def __len__(self):
        return len(self.backend)

    def lookup(
        self,
        key: Hashable,
        tokens: Iterable[int],
        *,
        cache_key: Optional[str] = None,
    ) -> APCLookup:
        tokens = [int(token) for token in tokens]
        native = self.backend.lookup_result(
            tokens, cache_key=cache_key, **_snapshot_identity(key)
        )
        hit = native.hit
        if hit is None:
            return APCLookup(
                cache=None,
                remaining_tokens=tokens,
                cached_tokens=0,
                hit=False,
                hit_kind=None,
                miss_reason=native.miss_reason,
                native=native,
            )
        return APCLookup(
            cache=hit.cache,
            remaining_tokens=list(hit.tail_ids),
            cached_tokens=int(hit.reused_tokens),
            hit=True,
            hit_kind=hit.kind,
            miss_reason=None,
            native=native,
        )

    def store(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: Any,
        *,
        cache_key: Optional[str] = None,
        **save_kwargs,
    ) -> APCCapabilities:
        path = self.backend.save(
            [int(token) for token in tokens],
            prompt_cache,
            cache_key=cache_key,
            **_snapshot_identity(key),
            **save_kwargs,
        )
        return APCCapabilities(
            topology="persistent_snapshot",
            exact_prefix=True,
            arbitrary_branch=bool(self.backend.trim_prompt_cache),
            stored=path is not None,
            native=path,
        )

    @property
    def apc_stats(self):
        return {
            "backend": "snapshot",
            "entries": len(self.backend),
            "bytes": int(getattr(self.backend, "total_bytes", 0)),
            "restore_status": getattr(self.backend, "restore_status", None),
            "restored_entries": int(
                getattr(self.backend, "restored_entries", 0) or 0
            ),
        }


class BlockAPCAdapter:
    """Unified facade for Muse/MLX-VLM's block and exact APC manager.

    ``apc_module`` is injected to keep mlx-lm independent of mlx-vlm. It must
    expose ``apc_lookup_plan`` and ``commit_prefix_blocks``.
    """

    def __init__(
        self,
        manager: Any,
        apc_module: Any,
        *,
        mode: str = "block",
        safe_lookup_min: int = 0,
        suffix_is_text_only=None,
        prefix_has_media=None,
    ):
        if mode not in ("block", "exact"):
            raise ValueError("mode must be 'block' or 'exact'")
        self.manager = manager
        self.apc_module = apc_module
        self.mode = mode
        self.safe_lookup_min = max(0, int(safe_lookup_min))
        self.suffix_is_text_only = suffix_is_text_only or (lambda _n: True)
        self.prefix_has_media = prefix_has_media or (lambda _n: False)

    def __getattr__(self, name: str):
        return getattr(self.manager, name)

    @staticmethod
    def _extra_hash(key: Hashable) -> Optional[int]:
        value = key.semantic_fingerprint if isinstance(key, APCKey) else key
        if value is None:
            return 0
        if isinstance(value, bool):
            return None
        try:
            return int(value.__index__())
        except (AttributeError, TypeError, ValueError):
            return None

    def lookup(self, key: Hashable, tokens: Iterable[int]) -> APCLookup:
        tokens = [int(token) for token in tokens]
        extra_hash = self._extra_hash(key)
        if extra_hash is None:
            return APCLookup(
                None,
                tokens,
                0,
                False,
                None,
                "semantic_fingerprint_must_be_integer",
            )
        plan = self.apc_module.apc_lookup_plan(
            self.manager,
            tokens,
            extra_hash=extra_hash,
            apc_mode=self.mode,
            safe_lookup_min=self.safe_lookup_min,
            suffix_is_text_only=self.suffix_is_text_only,
            prefix_has_media=self.prefix_has_media,
        )
        if plan is None:
            return APCLookup(
                None, tokens, 0, False, None, "no_compatible_prefix"
            )
        prefix_len = int(plan.get("prefix_len", 0) or 0)
        return APCLookup(
            cache=plan.get("warm_cache"),
            remaining_tokens=tokens[prefix_len:],
            cached_tokens=prefix_len,
            hit=0 < prefix_len < len(tokens),
            hit_kind=self.mode,
            miss_reason=None,
            native=plan,
        )

    def release(self, lookup: APCLookup) -> None:
        plan = lookup.native if isinstance(lookup.native, dict) else {}
        blocks = list(plan.get("matched_blocks") or [])
        if blocks:
            self.manager.release(blocks)
            plan["matched_blocks"] = []

    def store(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: Any,
        *,
        batch_idx: Optional[int] = None,
        skip_first_n_tokens: int = 0,
        blocks_in_use=(),
        clone: bool = True,
        disk: bool = True,
    ) -> APCCapabilities:
        tokens = [int(token) for token in tokens]
        extra_hash = self._extra_hash(key)
        if extra_hash is None:
            return APCCapabilities(
                topology=f"mlx_vlm_{self.mode}",
                exact_prefix=False,
                arbitrary_branch=False,
                reason="semantic_fingerprint_must_be_integer",
                stored=False,
            )
        if self.mode == "exact":
            stored = bool(
                self.manager.store_exact_cache(
                    tokens,
                    prompt_cache,
                    extra_hash=extra_hash,
                    clone=clone,
                    disk=disk,
                )
            )
            native = None
        else:
            native = self.apc_module.commit_prefix_blocks(
                self.manager,
                prompt_cache,
                tokens,
                batch_idx=batch_idx,
                extra_hash=extra_hash,
                skip_first_n_tokens=skip_first_n_tokens,
                blocks_in_use=blocks_in_use,
            )
            stored = bool(native)
        return APCCapabilities(
            topology=f"mlx_vlm_{self.mode}",
            exact_prefix=self.mode == "exact",
            arbitrary_branch=self.mode == "block",
            stored=stored,
            native=native,
        )

    @property
    def apc_stats(self):
        stats = getattr(self.manager, "stats", None)
        if stats is None:
            native = None
        elif hasattr(stats, "__dict__"):
            native = dict(stats.__dict__)
        else:
            native = stats
        return {"backend": f"mlx_vlm_{self.mode}", "native": native}


__all__ = ["BlockAPCAdapter", "SnapshotAPCAdapter"]
