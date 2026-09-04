"""Loader-time weight packing for the Qwen4 per-token decode megakernel.

Metal binds 31 buffers.  A 48-layer persistent kernel cannot bind a buffer per
projection -- Qwen3.8-Flash-Next's decode path alone is 3,000-odd tensors -- so
every decode weight is packed into a handful of large ``uint32`` buffers and
addressed through an offset table the kernel indexes by (layer, projection).

Two properties this module is responsible for:

**Bounded packing memory.** Packing is a copy, and the decode path is ~67 GiB
here, so a naive pack would need a second copy of the model resident.  MLX
gives contiguous 1-D slices, reshapes and ``mx.view`` for free -- measured
2026-09-03 on ``0.32.2.dev20260829``: a 64 MiB slice of a 256 MiB parent moves
``get_active_memory`` by 0.0 MiB, while ``mx.contiguous`` of the same slice
moves it by 64 MiB.  So the pack is built one GROUP at a time and each source
module parameter is then rebound to a zero-copy view of the packed buffer.
That does not prove the old allocation was released: sibling views or graph
owners can retain it. The decoder checks measured allocator and host memory
before each group, reserves two groups for output plus staging, and refuses
further submissions on budget or swap growth. Never call ``mx.contiguous`` on
a rebound weight view -- that is what re-materialises it.

**Round-trip validation.**  Every packed entry is checked bit-for-bit against
the array it came from, at pack time, before the source is released.  A pack
that cannot prove that is not usable.

Layout of one quantized entry, matching what the spike's ``qdot4_rows`` and
``qdot8_rows`` read (``results/qwen4-megakernel-spike-20260903_stage2.py``):

* ``payload`` -- the ``.weight`` ``uint32`` array, flat, row-major, untouched.
* ``sb`` -- ``scales`` and ``biases`` fused into one array, flat,
  reinterpreted as ``uint32`` words.  The fusion is not cosmetic: the naive
  layout needed 31 bindings for two layers.  Two layouts exist and the
  default is the tuning spec's:

  ``"interleaved"`` (DEFAULT, spec Sec. 6) -- ``[..., rows, 2, ngroups]``, a
  row's scales and biases adjacent, which is what the spike measured and what
  ``qdot4_rows`` reads as ``sb[soff + g]`` and ``sb[soff + ng + g]``.  One
  stream, best locality.  Its cost is on the OTHER side: both halves are
  strided, so ``rebind`` hands the source module lazy strided views that
  materialise the first time a stock forward touches them.  That is free while
  the megakernel serves the token and a per-call copy if it falls back, which
  is the right way round.

  ``"split"`` -- ``[2, ..., rows, ngroups]``, scales then biases, each half a
  contiguous slice and so rebindable zero-copy.  Costs one extra scalar in the
  kernel (bias base ``sb_off + n_sb/2``) and reads two streams.  Kept for a
  fallback-heavy configuration; not the shipped layout.

Both are aligned to ``_ALIGN`` words so a ``uint4``/``float4`` cast inside the
kernel is legal at every entry base.
"""

from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import mlx.core as mx

# 16 uint32 words = 64 B.  Metal buffer bases are 256-B aligned, so an entry at
# a 64-B aligned word offset is safe for every vector width the kernel casts to.
_ALIGN = 16

# Table stride, in uint32 words.  Fixed so the kernel indexes it as
# ``table[entry * MEGA_TABLE_STRIDE + field]``.
TABLE_STRIDE = 12

# Field order inside one table row.  Mirrored by the ``TBL_*`` constants the
# kernel source defines; keep the two in step.
TABLE_FIELDS = (
    "group",       # 0: which packed buffer holds this entry
    "kind",        # 1: KIND_QUANT | KIND_DENSE
    "rows",        # 2: output rows (per expert, for a table)
    "cols",        # 3: input features
    "experts",     # 4: 0 for a plain projection, E for a switch_mlp table
    "bits",        # 5: 4 or 8 (0 for dense)
    "group_size",  # 6: quantization group size (0 for dense)
    "w_off",       # 7: payload offset, uint32 words from the group base
    "sb_off",      # 8: scale/bias offset, uint32 words from the group base
    "n_w",         # 9: payload length in uint32 words
    "n_sb",        # 10: scale/bias length in uint32 words
    "flags",       # 11: reserved
)
assert len(TABLE_FIELDS) == TABLE_STRIDE

KIND_QUANT = 0
KIND_DENSE = 1


class PackError(RuntimeError):
    """Raised when a pack cannot be built or cannot prove itself."""


@dataclass(frozen=True)
class PackEntry:
    """One packed tensor: where it lives and how to read it."""

    key: str
    index: int
    group: int
    kind: int
    rows: int
    cols: int
    experts: int
    bits: int
    group_size: int
    w_off: int
    sb_off: int
    n_w: int
    n_sb: int
    shape: tuple[int, ...]

    def row(self) -> list[int]:
        return [
            self.group, self.kind, self.rows, self.cols, self.experts,
            self.bits, self.group_size, self.w_off, self.sb_off,
            self.n_w, self.n_sb, 0,
        ]


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _words(array: mx.array) -> int:
    """Length of ``array`` in uint32 words once flattened into a pack group."""
    itemsize = array.dtype.size
    total = array.size * itemsize
    if total % 4:
        raise PackError(f"tensor of {total} bytes is not a whole number of words")
    return total // 4


def _as_words(array: mx.array) -> mx.array:
    """Flatten any array to a 1-D ``uint32`` word view, without copying data.

    ``mx.view`` reinterprets the dtype in place; the reshape that precedes it
    is a view for a contiguous array.  Neither allocates.
    """
    flat = array.reshape(-1)
    if flat.dtype == mx.uint32:
        return flat
    return mx.view(flat, mx.uint32)


SB_INTERLEAVED = "interleaved"
SB_SPLIT = "split"
DEFAULT_SB_LAYOUT = SB_INTERLEAVED


def _scale_bias_bytes(scales, biases) -> int:
    """Metadata-only size for the kernel's fixed bfloat16 scale/bias format."""
    if biases is None or scales.shape != biases.shape:
        raise PackError("invalid scale/bias shape pair")
    if scales.dtype != mx.bfloat16 or biases.dtype != mx.bfloat16:
        raise PackError("scale/bias tensors must both be bfloat16")
    return 2 * int(scales.size) * int(scales.dtype.size)


def fuse_scales_biases(
    scales: mx.array, biases: mx.array, layout: str = DEFAULT_SB_LAYOUT
) -> mx.array:
    """Fuse a projection's scales and biases into one array.

    ``interleaved`` -> ``[..., rows, 2, ngroups]`` (the shipped layout).
    ``split`` -> ``[2, ..., rows, ngroups]``.
    """
    _scale_bias_bytes(scales, biases)
    if layout == SB_INTERLEAVED:
        return mx.stack([scales, biases], axis=-2)
    if layout == SB_SPLIT:
        return mx.stack([scales, biases], axis=0)
    raise PackError(f"unknown scale/bias layout {layout!r}")


# --------------------------------------------------------------------- sources


@dataclass
class SourceTensor:
    """One logical projection, resolved lazily so nothing is held twice."""

    key: str
    fetch: Callable[[], dict[str, mx.array]]
    quantized: bool
    bits: int = 0
    group_size: int = 0


# Gains the checkpoint ships ZERO-centred.  ``TextModel.sanitize`` adds 1.0 to
# each of these on load, so a pack built from the shards that skipped the fold
# would load cleanly and produce deterministic garbage -- the mlx-vlm
# #2041/#2045 class.  Kept in step with ``qwen4_exp.TextModel.sanitize``.
ZERO_CENTRED_SUFFIXES = (
    ".hc_norm.weight",
    ".norm_key.weight",
    ".norm_query.weight",
    ".norm_conv.weight",
    ".q_layernorm.weight",
    ".k_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "hyper_connection_mixer.hc_norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


class SafetensorsSource:
    """Reads packable tensors straight out of the shards, AS THE MODEL SEES THEM.

    Used by the CPU-side contract tests: it never instantiates the model, so a
    two-layer round trip costs two layers of memory rather than 104 GiB.  Two
    load-time transforms are reproduced here, because a pack that skipped them
    would disagree with the live module and the disagreement is silent:

    * ``sanitize``'s zero-centred gain fold and the conv1d axis move;
    * ``transform_moe_weights``' gate/up fusion, so ``switch_mlp.gate_up_proj``
      resolves whether the checkpoint ships it fused or split.  The fused
      layout is also what the kernel's expert phase wants, so on a live model
      the pack reads it with no concatenation at all.
    """

    def __init__(self, path: str):
        self.path = path
        index = json.load(open(os.path.join(path, "model.safetensors.index.json")))
        self.weight_map: dict[str, str] = index["weight_map"]
        self.config = json.load(open(os.path.join(path, "config.json")))
        self._quant = self.config.get("quantization", {})
        self._default_bits = int(self._quant.get("bits", 4))
        self._default_gs = int(self._quant.get("group_size", 64))
        self._cache: dict[str, dict[str, mx.array]] = {}

    def quant_spec(self, key: str) -> tuple[int, int]:
        """Per-tensor bits/group_size, honouring the config's overrides.

        ``mlp.gate`` and ``mlp.shared_expert_gate`` are 8-bit in this
        checkpoint; everything else on the decode path is 4-bit/64.
        """
        override = self._quant.get(key)
        if isinstance(override, dict):
            return int(override["bits"]), int(override["group_size"])
        return self._default_bits, self._default_gs

    def _shard(self, shard: str) -> dict[str, mx.array]:
        if shard not in self._cache:
            # One shard at a time; the caller drops it when the group closes.
            self._cache = {shard: mx.load(os.path.join(self.path, shard))}
        return self._cache[shard]

    def release(self) -> None:
        self._cache = {}

    def _raw_available(self, key: str) -> bool:
        return key in self.weight_map or f"{key}.weight" in self.weight_map

    def _split_pair(self, key: str) -> Optional[tuple[str, str]]:
        """``...gate_up_proj`` -> the split spelling, when only that exists."""
        if not key.endswith("gate_up_proj"):
            return None
        stem = key[: -len("gate_up_proj")]
        gate, up = f"{stem}gate_proj", f"{stem}up_proj"
        if self._raw_available(gate) and self._raw_available(up):
            return gate, up
        return None

    def has(self, key: str) -> bool:
        return self._raw_available(key) or self._split_pair(key) is not None

    def _fetch_raw(self, key: str) -> dict[str, mx.array]:
        if f"{key}.weight" in self.weight_map:
            out = {}
            for part in ("weight", "scales", "biases"):
                name = f"{key}.{part}"
                if name in self.weight_map:
                    out[part] = self._shard(self.weight_map[name])[name]
            return out
        name = key if key in self.weight_map else f"{key}.weight"
        return {"weight": self._shard(self.weight_map[name])[name]}

    def _sanitize(self, key: str, parts: dict[str, mx.array]) -> dict[str, mx.array]:
        weight = parts["weight"]
        if "conv1d.weight" in key and weight.ndim == 3 and weight.shape[-1] != 1:
            parts = dict(parts)
            parts["weight"] = mx.contiguous(weight.moveaxis(2, 1))
            return parts
        if any(key.endswith(suffix) for suffix in ZERO_CENTRED_SUFFIXES):
            parts = dict(parts)
            parts["weight"] = weight + 1.0
        return parts

    def fetch(self, key: str) -> dict[str, mx.array]:
        pair = None if self._raw_available(key) else self._split_pair(key)
        if pair is None:
            return self._sanitize(key, self._fetch_raw(key))
        # ``transform_moe_weights`` concatenates on axis -2, gate rows first.
        gate, up = (self._fetch_raw(name) for name in pair)
        return {
            part: mx.contiguous(
                mx.concatenate([gate[part], up[part]], axis=-2)
            )
            for part in gate
        }


class ModuleSource:
    """Reads packable tensors off a live ``nn.Module`` tree, and can rebind."""

    def __init__(self, root):
        self.root = root

    def _resolve(self, key: str):
        node = self.root
        for part in key.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        return node

    def has(self, key: str) -> bool:
        try:
            self._resolve(key)
        except (AttributeError, IndexError, KeyError):
            return False
        return True

    def quant_spec(self, key: str) -> tuple[int, int]:
        module = self._resolve(key)
        return int(getattr(module, "bits", 4)), int(getattr(module, "group_size", 64))

    def fetch(self, key: str) -> dict[str, mx.array]:
        module = self._resolve(key)
        if isinstance(module, mx.array):
            return {"weight": module}
        out = {}
        for part in ("weight", "scales", "biases"):
            value = getattr(module, part, None)
            if isinstance(value, mx.array):
                out[part] = value
        return out

    def rebind(self, key: str, parts: dict[str, mx.array]) -> None:
        module = self._resolve(key)
        if isinstance(module, mx.array):
            # A bare parameter -- ``q_norm.weight``, ``A_log``, ``conv1d.weight``
            # -- resolves to the ARRAY, so the attribute has to be set on its
            # parent.  Setting it on the array itself raises, which is what a
            # full-model rebind found the first time one was tried.
            parent, _, leaf = key.rpartition(".")
            setattr(self._resolve(parent), leaf, parts["weight"])
            return
        for part, value in parts.items():
            setattr(module, part, value)

    def release(self) -> None:  # pragma: no cover - symmetry with the other source
        return None


# ------------------------------------------------------------------- the plan


def decode_path_keys(
    *,
    num_layers: int,
    layer_types: list[str],
    ple_layer_ids: Iterable[int] = (),
    include_mtp: bool = True,
    include_experts: bool = True,
    include_lm_head: bool = True,
    fuse_gate_up: bool = True,
    layers: Optional[Iterable[int]] = None,
) -> list[tuple[str, str]]:
    """Every tensor the decode megakernel reads, as ``(key, group_role)``.

    The embedding is deliberately absent: PLE and ``embed_tokens`` are a host
    file read and a gather that depend only on the input token, so they are
    hoisted before the launch and the embedding arrives as a kernel input.
    """
    wanted = set(range(num_layers)) if layers is None else set(layers)
    ple = set(int(i) - 1 for i in ple_layer_ids)
    # The live module fuses the routed gate and up projections into one
    # ``gate_up_proj`` table at load time, and that IS the layout the kernel's
    # first expert phase wants -- so on a live model the pack reads it with no
    # concatenation.  The split spelling stays reachable for a raw checkpoint.
    expert_names = (
        ("gate_up_proj", "down_proj") if fuse_gate_up
        else ("gate_proj", "up_proj", "down_proj")
    )
    out: list[tuple[str, str]] = []

    def add(key: str, role: str = "main") -> None:
        out.append((key, role))

    for index in range(num_layers):
        if index not in wanted:
            continue
        base = f"language_model.model.layers.{index}"
        for hyper in ("attn_hyper_connection", "mlp_hyper_connection"):
            add(f"{base}.{hyper}.hc_norm.weight")
            add(f"{base}.{hyper}.input_mix_weight_down")
            add(f"{base}.{hyper}.input_mix_weight_up")
            add(f"{base}.{hyper}.block_inject_weight")
        if layer_types[index] == "linear_attention":
            attn = f"{base}.linear_attn"
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
                         "out_proj"):
                add(f"{attn}.{name}")
            for name in ("A_log", "dt_bias", "conv1d.weight", "norm.weight"):
                add(f"{attn}.{name}")
        else:
            attn = f"{base}.self_attn"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add(f"{attn}.{name}")
            for name in ("q_norm.weight", "k_norm.weight"):
                add(f"{attn}.{name}")
            add(f"{attn}.indexer.index_qk_proj")
            add(f"{attn}.indexer.q_layernorm.weight")
            add(f"{attn}.indexer.k_layernorm.weight")
        if index in ple:
            # The n-gram gather is the host hoist; the rest of the PLE chain
            # -- projections, norms, the dilated depthwise conv -- is device
            # work and is packed with its layer.
            add(f"{base}.ple.key_proj")
            add(f"{base}.ple.value_proj")
            add(f"{base}.ple.conv1d.weight")
            add(f"{base}.ple.norm_conv.weight")
            add(f"{base}.ple.norm_key.weight")
            add(f"{base}.ple.norm_query.weight")
        add(f"{base}.mlp.gate")
        add(f"{base}.mlp.shared_expert_gate")
        for name in ("gate_proj", "up_proj", "down_proj"):
            add(f"{base}.mlp.shared_expert.{name}")
        if include_experts:
            for name in expert_names:
                add(f"{base}.mlp.switch_mlp.{name}", "experts")

    mixer = "language_model.model.hyper_connection_mixer"
    add(f"{mixer}.hc_norm.weight")
    add(f"{mixer}.input_mix_weight_down")
    add(f"{mixer}.input_mix_weight_up")
    if include_lm_head:
        add("language_model.lm_head", "lm_head")

    if include_mtp:
        add("mtp.pre_fc_norm_embedding.weight")
        add("mtp.pre_fc_norm_hidden.weight")
        add("mtp.fc_embedding")
        add("mtp.fc_hidden")
        add("mtp.hyper_connection_mixer.hc_norm.weight")
        add("mtp.hyper_connection_mixer.input_mix_weight_down")
        add("mtp.hyper_connection_mixer.input_mix_weight_up")
        base = "mtp.layers.0"
        for hyper in ("attn_hyper_connection", "mlp_hyper_connection"):
            add(f"{base}.{hyper}.hc_norm.weight")
            add(f"{base}.{hyper}.input_mix_weight_down")
            add(f"{base}.{hyper}.input_mix_weight_up")
            add(f"{base}.{hyper}.block_inject_weight")
        attn = f"{base}.self_attn"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            add(f"{attn}.{name}")
        add(f"{attn}.q_norm.weight")
        add(f"{attn}.k_norm.weight")
        add(f"{attn}.indexer.index_qk_proj")
        add(f"{attn}.indexer.q_layernorm.weight")
        add(f"{attn}.indexer.k_layernorm.weight")
        add(f"{base}.mlp.gate")
        add(f"{base}.mlp.shared_expert_gate")
        for name in ("gate_proj", "up_proj", "down_proj"):
            add(f"{base}.mlp.shared_expert.{name}")
        if include_experts:
            for name in expert_names:
                add(f"{base}.mlp.switch_mlp.{name}", "experts")
    return out


# -------------------------------------------------------------------- the pack


@dataclass
class MegaWeightPack:
    """Packed decode weights plus the offset table that addresses them."""

    buffers: list[mx.array] = field(default_factory=list)
    group_roles: list[str] = field(default_factory=list)
    entries: dict[str, PackEntry] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    table: Optional[mx.array] = None
    sb_layout: str = DEFAULT_SB_LAYOUT
    stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ readback
    def payload(self, key: str) -> mx.array:
        entry = self.entries[key]
        buf = self.buffers[entry.group]
        return buf[entry.w_off: entry.w_off + entry.n_w]

    def scales_biases(self, key: str) -> Optional[mx.array]:
        entry = self.entries[key]
        if entry.n_sb == 0:
            return None
        buf = self.buffers[entry.group]
        words = buf[entry.sb_off: entry.sb_off + entry.n_sb]
        return mx.view(words, mx.bfloat16)

    def views(self, key: str) -> dict[str, mx.array]:
        """The entry as the source module would hold it -- zero-copy."""
        entry = self.entries[key]
        if entry.kind == KIND_DENSE:
            words = self.payload(key)
            dense = mx.view(words, mx.bfloat16)[: _dense_elems(entry)]
            return {"weight": dense.reshape(entry.shape)}
        weight = self.payload(key).reshape(entry.shape)
        sb = self.scales_biases(key)
        ngroups = entry.cols // entry.group_size
        sb_shape = (*entry.shape[:-1], ngroups)
        if self.sb_layout == SB_SPLIT:
            half = sb.size // 2
            return {
                "weight": weight,
                "scales": sb[:half].reshape(sb_shape),
                "biases": sb[half:].reshape(sb_shape),
            }
        # Interleaved: both halves are strided views.  MLX keeps them lazy, so
        # nothing is copied until a stock forward actually reads one.
        sb = sb.reshape(*entry.shape[:-1], 2, ngroups)
        return {
            "weight": weight,
            "scales": sb[..., 0, :],
            "biases": sb[..., 1, :],
        }

    def table_row(self, key: str) -> dict[str, int]:
        return dict(zip(TABLE_FIELDS, self.entries[key].row()))

    def summary(self) -> dict[str, Any]:
        return {
            "entries": len(self.entries),
            "groups": [
                {
                    "index": i,
                    "role": self.group_roles[i],
                    "bytes": int(buf.size) * 4,
                }
                for i, buf in enumerate(self.buffers)
            ],
            "bound_buffers": len(self.buffers) + 1,  # + the offset table
            "table_stride": TABLE_STRIDE,
            **self.stats,
        }


def _dense_elems(entry: PackEntry) -> int:
    total = 1
    for dim in entry.shape:
        total *= dim
    return total


def _plan_groups(
    sizes: list[tuple[str, str, int]], max_group_bytes: int
) -> list[tuple[str, list[str]]]:
    """Greedy split of ``(key, role, words)`` into buffers under the cap.

    Roles never share a buffer: the kernel binds them by name, so a layout that
    let ``experts`` spill into ``main`` would change the binding set with the
    checkpoint.  Within a role, entries stay in consumption order.

    An entry cannot straddle two bindings without another table field and a
    branch in every read, so an entry larger than the cap is refused before
    any group allocation. The cap is also the runtime's transient-memory
    reservation and therefore cannot be treated as a soft grouping hint.
    """
    if (
        isinstance(max_group_bytes, bool)
        or not isinstance(max_group_bytes, int)
        or max_group_bytes <= 0
    ):
        raise PackError("max_group_bytes must be a positive integer")
    cap_words = max_group_bytes // 4
    if cap_words < 1:
        raise PackError("max_group_bytes is smaller than one uint32 word")
    groups: list[tuple[str, list[str]]] = []
    current_role: Optional[str] = None
    current: list[str] = []
    used = 0
    # Stable sort by role: the plan interleaves roles layer by layer, and a
    # role change closes a buffer, so leaving it interleaved would emit one
    # tiny buffer per layer and blow the 31-binding limit it exists to respect.
    by_role: dict[str, list[tuple[str, str, int]]] = {}
    for item in sizes:
        by_role.setdefault(item[1], []).append(item)
    ordered = [item for role in by_role for item in by_role[role]]
    for key, role, words in ordered:
        need = _align(words)
        if need > cap_words:
            raise PackError(
                f"{key}: aligned entry is {need * 4} bytes, over the "
                f"{max_group_bytes}-byte group cap"
            )
        if current_role != role or (current and used + need > cap_words):
            if current:
                groups.append((current_role, current))
            current_role, current, used = role, [], 0
        current.append(key)
        used += need
    if current:
        groups.append((current_role, current))
    return groups


def estimate_pack(source, plan: list[tuple[str, str]], *,
                  max_group_bytes: int = 8 << 30) -> dict[str, Any]:
    """Compute the exact packed layout without allocating or rebinding.

    This mirrors build_pack's first pass so admission can reject an oversized
    full pack (when ``rebind=False``), an oversized transient group, or too
    many Metal bindings before the first large buffer is created.
    """
    missing = [key for key, _ in plan if not source.has(key)]
    if missing:
        raise PackError(f"{len(missing)} planned tensors absent, first: {missing[0]}")
    sizes: list[tuple[str, str, int]] = []
    scale_bias_bytes = 0
    try:
        for key, role in plan:
            parts = source.fetch(key)
            weight = parts["weight"]
            n_w = _words(weight)
            n_sb = 0
            if "scales" in parts:
                scales, biases = parts["scales"], parts.get("biases")
                sb_bytes = _scale_bias_bytes(scales, biases)
                n_sb = sb_bytes // 4
                scale_bias_bytes += sb_bytes
            sizes.append((key, role, _align(n_w) + _align(n_sb)))
    finally:
        source.release()
    groups = _plan_groups(sizes, max_group_bytes)
    words_by_key = {key: words for key, _, words in sizes}
    group_bytes = [
        sum(words_by_key[key] for key in keys) * 4 for _, keys in groups
    ]
    return {
        "entries": len(sizes),
        "groups": len(groups),
        "group_bytes": group_bytes,
        "packed_bytes": sum(group_bytes),
        "largest_group_bytes": max(group_bytes, default=0),
        "scale_bias_bytes": scale_bias_bytes,
        "max_group_bytes": int(max_group_bytes),
    }


def build_pack(
    source,
    plan: list[tuple[str, str]],
    *,
    max_group_bytes: int = 8 << 30,
    validate: bool = True,
    rebind: bool = False,
    sb_layout: str = DEFAULT_SB_LAYOUT,
    quant_spec: Optional[Callable[[str], tuple[int, int]]] = None,
    memory_guard: Optional[Callable[..., None]] = None,
) -> MegaWeightPack:
    """Pack ``plan``'s tensors into group buffers and prove the round trip.

    ``validate`` compares every packed entry, bit for bit, against the array it
    came from, while that array is still resident.  ``rebind`` then replaces the
    source module's parameter with a zero-copy view of the pack, which is what
    keeps the pack from doubling the model's footprint.

    ``memory_guard`` runs before any group staging and after its evaluated
    output is rebound and temporary references are released. It may refuse
    the next submission; the caller must not reuse a partially rebound model.
    """
    quant_spec = quant_spec or source.quant_spec
    missing = [key for key, _ in plan if not source.has(key)]
    if missing:
        raise PackError(f"{len(missing)} planned tensors absent, first: {missing[0]}")
    device_cap = int(mx.device_info().get("max_buffer_length", 0)) or None

    # Pass 1: sizes only, so group boundaries are known before anything is
    # materialised.  Sizes come from the shard header, not from a load.
    sizes: list[tuple[str, str, int]] = []
    meta: dict[str, dict[str, Any]] = {}
    for key, role in plan:
        parts = source.fetch(key)
        weight = parts["weight"]
        n_w = _words(weight)
        n_sb = 0
        sb_shape = None
        if "scales" in parts:
            scales, biases = parts["scales"], parts.get("biases")
            sb_bytes = _scale_bias_bytes(scales, biases)
            n_sb = sb_bytes // 4
            sb_shape = tuple(parts["scales"].shape)
            del scales, biases
        meta[key] = {
            "role": role,
            "n_w": n_w,
            "n_sb": n_sb,
            "shape": tuple(weight.shape),
            "sb_shape": sb_shape,
            "quantized": "scales" in parts,
        }
        sizes.append((key, role, _align(n_w) + _align(n_sb)))
        del parts, weight
    source.release()

    groups = _plan_groups(sizes, max_group_bytes)
    pack = MegaWeightPack(sb_layout=sb_layout)
    source_bytes = 0
    packed_bytes = 0
    peak_transient = 0
    validated = 0
    words_by_key = {key: words for key, _, words in sizes}

    for group_index, (role, keys) in enumerate(groups):
        group_bytes = sum(words_by_key[key] for key in keys) * 4
        if memory_guard is not None:
            memory_guard(stage="before_group", group_index=group_index,
                         group_bytes=group_bytes)
        chunks: list[mx.array] = []
        used = 0
        for key in keys:
            info = meta[key]
            parts = source.fetch(key)
            weight = parts["weight"]
            source_bytes += weight.size * weight.dtype.size
            entry_w_off = used
            words = _as_words(weight)
            chunks.append(words)
            used += info["n_w"]
            pad = _align(used) - used
            if pad:
                chunks.append(mx.zeros((pad,), mx.uint32))
                used += pad
            entry_sb_off = used
            if info["quantized"]:
                fused = fuse_scales_biases(parts["scales"], parts["biases"],
                                           sb_layout)
                source_bytes += (
                    parts["scales"].size * parts["scales"].dtype.size
                    + parts["biases"].size * parts["biases"].dtype.size
                )
                chunks.append(_as_words(fused))
                del fused
                used += info["n_sb"]
                pad = _align(used) - used
                if pad:
                    chunks.append(mx.zeros((pad,), mx.uint32))
                    used += pad
            bits, group_size = (
                quant_spec(key) if info["quantized"] else (0, 0)
            )
            cols = info["shape"][-1]
            if info["quantized"]:
                # The payload's last axis is packed, so the logical input width
                # comes off the scale table, which is one value per group.
                cols = info["sb_shape"][-1] * group_size
            entry = PackEntry(
                key=key,
                index=len(pack.order),
                group=group_index,
                kind=KIND_QUANT if info["quantized"] else KIND_DENSE,
                rows=info["shape"][-2] if len(info["shape"]) >= 2 else 1,
                cols=cols,
                experts=info["shape"][0] if len(info["shape"]) == 3 else 0,
                bits=bits,
                group_size=group_size,
                w_off=entry_w_off,
                sb_off=entry_sb_off if info["quantized"] else 0,
                n_w=info["n_w"],
                n_sb=info["n_sb"],
                shape=info["shape"],
            )
            pack.entries[key] = entry
            pack.order.append(key)
            del parts, weight, words

        if device_cap is not None and used * 4 > device_cap:
            raise PackError(
                f"group {group_index} ({role}) is {used * 4} bytes, over the "
                f"device max_buffer_length of {device_cap}"
            )
        buf = mx.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        mx.eval(buf)
        packed_bytes += int(buf.size) * 4
        peak_transient = max(peak_transient, int(buf.size) * 4)
        pack.buffers.append(buf)
        pack.group_roles.append(role)
        del chunks

        if validate or rebind:
            for key in keys:
                if validate:
                    _validate_entry(pack, source, key)
                    validated += 1
                if rebind and hasattr(source, "rebind"):
                    source.rebind(key, pack.views(key))
            source.release()
        if rebind:
            # Evaluated packed outputs no longer need source graph temporaries.
            gc.collect()
            mx.clear_cache()
        if memory_guard is not None:
            memory_guard(stage="after_group", group_index=group_index,
                         group_bytes=0)

    table = mx.array(
        [value for key in pack.order for value in pack.entries[key].row()],
        mx.uint32,
    )
    mx.eval(table)
    pack.table = table
    pack.stats = {
        "source_bytes": source_bytes,
        "packed_bytes": packed_bytes,
        "extra_bytes": packed_bytes - source_bytes,
        "peak_transient_bytes": peak_transient,
        "rebound": bool(rebind and hasattr(source, "rebind")),
        "validated_entries": validated,
        "max_group_bytes": max_group_bytes,
        "sb_layout": sb_layout,
    }
    return pack


def _validate_entry(pack: MegaWeightPack, source, key: str) -> None:
    """Bit-for-bit round trip of one packed entry against its source."""
    entry = pack.entries[key]
    parts = source.fetch(key)
    got = pack.views(key)
    weight = parts["weight"]
    if tuple(got["weight"].shape) != tuple(weight.shape):
        raise PackError(
            f"{key}: shape {tuple(got['weight'].shape)} != {tuple(weight.shape)}"
        )
    if not _bit_equal(got["weight"], weight):
        raise PackError(f"{key}: payload round trip differs")
    if entry.kind == KIND_QUANT:
        for part in ("scales", "biases"):
            if not _bit_equal(got[part], parts[part]):
                raise PackError(f"{key}: {part} round trip differs")


def _bit_equal(a: mx.array, b: mx.array) -> bool:
    """Equality on the bit pattern, so a NaN or a -0.0 cannot pass as equal."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    width = a.dtype.size
    as_int = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}[width]
    left = mx.view(mx.contiguous(a).reshape(-1), as_int)
    right = mx.view(mx.contiguous(b).reshape(-1), as_int)
    return bool(mx.array_equal(left, right).item())
