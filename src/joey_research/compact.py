"""Offline physical text-stack compaction. No model code or pickle is loaded."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, BinaryIO


class CompactError(ValueError):
    """Unsupported or inconsistent input; no compact checkpoint was published."""


_WIDTH = {"F16": 2, "BF16": 2, "F32": 4}
_LIMIT = 100_000_000  # safetensors header limit
_COPY_BYTES = 4 * 1024 * 1024
_MERGE_ELEMENTS = 4096


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompactError(message)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse(raw: bytes) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise CompactError(f"Non-finite JSON number: {token}")

    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompactError("Invalid UTF-8 JSON") from exc
    _require(isinstance(obj, dict), "Expected a JSON object")
    return obj


def _encode(obj: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _regular(path: Path) -> Path:
    _require(not path.is_symlink() and path.is_file(), f"Expected a local regular file: {path.name}")
    return path


def _json_file(path: Path) -> dict[str, Any]:
    _require(_regular(path).stat().st_size <= _LIMIT, f"Oversize JSON: {path.name}")
    return _parse(path.read_bytes())


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    _require(len(data) == size, "Truncated safetensors file")
    return data


def _chunks(stream: BinaryIO, length: int, cap: int) -> Iterator[bytes]:
    while length:
        data = _read(stream, min(length, cap))
        yield data
        length -= len(data)


@dataclass(frozen=True)
class Tensor:
    file: Path
    offset: int  # absolute byte offset
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @property
    def numel(self) -> int:
        return self.nbytes // _WIDTH[self.dtype]

    def open_at(self, stack: ExitStack) -> BinaryIO:
        stream = stack.enter_context(self.file.open("rb"))
        stream.seek(self.offset)
        return stream


def _shard(path: Path) -> tuple[dict[str, Tensor], dict[str, str]]:
    with _regular(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        (n,) = struct.unpack("<Q", _read(stream, 8))
        _require(0 < n <= min(_LIMIT, size - 8), "Invalid safetensors header size")
        raw = _read(stream, n)
    _require(raw.startswith(b"{"), "Safetensors header must start with an object")
    header = _parse(raw)
    metadata = header.pop("__metadata__", {})
    _require(isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values()),
             "Invalid safetensors metadata")
    tensors = {}
    ranges = []
    for key, record in header.items():
        _require(key and isinstance(record, dict) and set(record) == {"dtype", "shape", "data_offsets"},
                 f"Invalid tensor record: {key}")
        dtype, shape, offsets = record["dtype"], record["shape"], record["data_offsets"]
        _require(isinstance(dtype, str) and dtype in _WIDTH, f"Unsupported storage dtype: {dtype}")
        _require(isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape),
                 f"Invalid shape: {key}")
        _require(isinstance(offsets, list) and len(offsets) == 2 and
                 all(type(x) is int and x >= 0 for x in offsets), f"Invalid offsets: {key}")
        start, end = offsets
        _require(end <= size - 8 - n and end - start == math.prod(shape) * _WIDTH[dtype],
                 f"Shape/offset mismatch: {key}")
        ranges.append((start, end))
        tensors[key] = Tensor(path, 8 + n + start, dtype, tuple(shape), end - start)
    cursor = 0
    for start, end in sorted(ranges):
        _require(start == cursor, "Overlapping or non-contiguous safetensors data")
        cursor = end
    _require(cursor == size - 8 - n, "Unindexed safetensors data")
    return tensors, metadata


@dataclass
class Checkpoint:
    tensors: dict[str, Tensor]
    aliases: dict[str, str]
    files: tuple[Path, ...]

    @classmethod
    def open(cls, directory: Path) -> Checkpoint:
        index_path = directory / "model.safetensors.index.json"
        single = directory / "model.safetensors"
        indexed = os.path.lexists(index_path)
        weight_map: dict[str, str] = {}
        if indexed:
            _require(not os.path.lexists(single), "Ambiguous single and indexed checkpoints")
            index = _json_file(index_path)
            weight_map = index.get("weight_map")
            _require(isinstance(weight_map, dict) and bool(weight_map), "Empty or invalid shard index")
            for name in weight_map.values():
                _require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors", name),
                         "Shard index must use local safetensors basenames")
            files = tuple(directory / name for name in sorted(set(weight_map.values())))
        else:
            files = (single,)
        _require(set(directory.glob("*.safetensors")) == set(files), "Missing or unindexed weight file")
        tensors: dict[str, Tensor] = {}
        aliases: dict[str, str] = {}
        for file in files:
            shard, metadata = _shard(file)
            _require(not (tensors.keys() & shard.keys()), "Duplicate tensor in multiple shards")
            for key in shard:
                _require(not indexed or weight_map.get(key) == file.name, f"Wrong or missing shard for {key}")
            tensors.update(shard)
            fmt = metadata.pop("format", "pt")
            _require(fmt == "pt", "Only PyTorch layout is supported")
            for alias, target in metadata.items():
                _require(alias not in aliases or aliases[alias] == target, "Inconsistent shard metadata")
                aliases[alias] = target
        _require(bool(tensors) and (not indexed or tensors.keys() == weight_map.keys()),
                 "Missing indexed tensors")
        return cls(tensors, aliases, files)


def _identical(a: Tensor, b: Tensor) -> bool:
    if (a.shape, a.dtype) != (b.shape, b.dtype):
        return False
    with ExitStack() as stack:
        left, right = a.open_at(stack), b.open_at(stack)
        return all(data == _read(right, len(data)) for data in _chunks(left, a.nbytes, _COPY_BYTES))


_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")


def _f32(x: float) -> float:
    return _F32.unpack(_F32.pack(x))[0]


def _values(data: bytes, dtype: str) -> Iterator[float]:
    if dtype == "BF16":
        return (_F32.unpack(_U32.pack(bits << 16))[0] for (bits,) in struct.iter_unpack("<H", data))
    return (x for (x,) in struct.iter_unpack({"F16": "<e", "F32": "<f"}[dtype], data))


def _cast(values: Sequence[float], dtype: str) -> bytes:
    if dtype == "BF16":
        ints = []
        for value in values:
            (bits,) = _U32.unpack(_F32.pack(value))
            rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16  # round-to-nearest-even
            _require((rounded & 0x7F80) != 0x7F80, "Non-finite BF16 merge result")
            ints.append(rounded)
        return struct.pack(f"<{len(ints)}H", *ints)
    return struct.pack(f"<{len(values)}" + {"F16": "e", "F32": "f"}[dtype], *values)


@dataclass(frozen=True)
class OutputTensor:
    inputs: tuple[Tensor, ...]
    coefficients: tuple[float, ...] = ()

    @property
    def anchor(self) -> Tensor:
        return self.inputs[0]

    def data(self) -> Iterator[bytes]:
        t = self.anchor
        with ExitStack() as stack:
            streams = [source.open_at(stack) for source in self.inputs]
            if not self.coefficients:
                yield from _chunks(streams[0], t.nbytes, _COPY_BYTES)
                return
            try:
                for pos in range(0, t.numel, _MERGE_ELEMENTS):
                    n = min(_MERGE_ELEMENTS, t.numel - pos)
                    accum = [0.0] * n
                    for stream, weight in zip(streams, self.coefficients, strict=True):
                        data = _read(stream, n * _WIDTH[t.dtype])
                        for i, x in enumerate(_values(data, t.dtype)):
                            accum[i] = _f32(accum[i] + _f32(x * weight))
                    _require(all(map(math.isfinite, accum)), "Non-finite projection in FP32 merge")
                    yield _cast(accum, t.dtype)
            except OverflowError as exc:
                raise CompactError("Overflow in FP32 merge or output cast") from exc


def _write_weights(path: Path, outputs: Mapping[str, OutputTensor], aliases: dict[str, str]) -> str:
    header: dict[str, Any] = {"__metadata__": {"format": "pt", **aliases}}
    cursor = 0
    names = sorted(outputs)
    for name in names:
        t = outputs[name].anchor
        header[name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [cursor, cursor + t.nbytes]}
        cursor += t.nbytes
    raw = _encode(header)
    raw += b" " * (-len(raw) % 8)
    _require(len(raw) <= _LIMIT, "Compact checkpoint exceeds the single-shard header limit")
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        def write(chunk: bytes) -> None:
            stream.write(chunk)
            digest.update(chunk)

        write(struct.pack("<Q", len(raw)) + raw)
        for name in names:
            for chunk in outputs[name].data():
                write(chunk)
        _require(stream.tell() == 8 + len(raw) + cursor, "Wrong compact payload length")
    return digest.hexdigest()


MODEL = "baselabs/joey-1"
REVISION = "db5c2a631f2da93c4173378438d8246966e25159"
_PIN_CONFIG = "8337ef3afdc5d188f1c673f833783ae5ff72641a0fce4885e6ffa83ce1c38640"
_PIN_WEIGHTS = "4c503aded23bae311dff80a63326503fb720572c178f07a2142332d55f5f3afc"
_FORMAT = "joey-compact/v1"
_CYCLE = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
_LANG = "model.language_model."
_LAYERS = _LANG + "layers."
_EMBED = _LANG + "embed_tokens.weight"
_NORM = _LANG + "norm.weight"
_HEAD = "lm_head.weight"
_VISION = "model.visual."
_MERGE_SUFFIXES = frozenset({"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"})
_SIDECARS = frozenset({
    "LICENSE", "NOTICE", "chat_template.jinja", "generation_config.json", "processor_config.json",
    "readout.json", "tokenizer.json", "tokenizer_config.json",
})


def _digest(path: Path, copy_to: BinaryIO | None = None) -> str:
    sha = hashlib.sha256()
    with _regular(path).open("rb") as stream:
        while data := stream.read(_COPY_BYTES):
            sha.update(data)
            if copy_to is not None:
                copy_to.write(data)
    return sha.hexdigest()


def _dimensions(text: dict, *keys: str) -> tuple[int, ...]:
    dims = tuple(text.get(key) for key in keys)
    _require(all(type(x) is int and x > 0 for x in dims), f"Expected positive dimensions: {keys}")
    return dims


def _schemas(text: dict) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    h, m, q, kv, d, lk, dk, lv, dv, kernel = _dimensions(
        text, "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
        "linear_value_head_dim", "linear_conv_kernel_dim",
    )
    qkv = 2 * lk * dk + lv * dv
    common = {
        "input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
        "mlp.up_proj.weight": (m, h), "mlp.gate_proj.weight": (m, h), "mlp.down_proj.weight": (h, m),
    }
    linear = {
        **common, "linear_attn.A_log": (lv,), "linear_attn.dt_bias": (lv,),
        "linear_attn.conv1d.weight": (qkv, 1, kernel), "linear_attn.norm.weight": (dv,),
        "linear_attn.in_proj_a.weight": (lv, h), "linear_attn.in_proj_b.weight": (lv, h),
        "linear_attn.in_proj_qkv.weight": (qkv, h), "linear_attn.in_proj_z.weight": (lv * dv, h),
        "linear_attn.out_proj.weight": (h, lv * dv),
    }
    full = {
        **common, "self_attn.q_norm.weight": (d,), "self_attn.k_norm.weight": (d,),
        "self_attn.q_proj.weight": (2 * q * d, h), "self_attn.k_proj.weight": (kv * d, h),
        "self_attn.v_proj.weight": (kv * d, h), "self_attn.o_proj.weight": (h, q * d),
    }
    return linear, full


def _layout(config: dict, checkpoint: Checkpoint) -> list[dict[str, Tensor]]:
    _require(config.get("model_type") == "qwen3_5" and
             config.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "Unsupported Joey layout")
    text = config.get("text_config")
    _require(isinstance(text, dict) and text.get("model_type") == "qwen3_5_text" and
             text.get("full_attention_interval") == 4 and text.get("mlp_only_layers") == [] and
             text.get("attn_output_gate") is True and text.get("attention_bias") is False,
             "Unsupported text configuration")
    n, h, vocab = _dimensions(text, "num_hidden_layers", "hidden_size", "vocab_size")
    _require(n % 4 == 0 and n <= 24 and text.get("layer_types") == list(_CYCLE) * (n // 4),
             "Expected complete [linear, linear, linear, full] cycles")
    tied = text.get("tie_word_embeddings")
    _require(type(tied) is bool and config.get("tie_word_embeddings") is tied, "Inconsistent tie flags")
    tensors = checkpoint.tensors
    _require(_EMBED in tensors and tensors[_EMBED].shape == (vocab, h) and
             _NORM in tensors and tensors[_NORM].shape == (h,), "Missing or incompatible embedding/final norm")
    _require((_HEAD in tensors and tensors[_HEAD].shape == (vocab, h)) or
             (_HEAD not in tensors and tied), "Missing or incompatible language head")
    _require(not checkpoint.aliases or (tied and _HEAD not in tensors and
             checkpoint.aliases == {_HEAD: _EMBED}), "Unsupported safetensors metadata or weight alias")
    _require(any(key.startswith(_VISION) for key in tensors), "Full bundle must include vision parameters")
    layers: list[dict[str, Tensor]] = [{} for _ in range(n)]
    linear, full = _schemas(text)
    for key, tensor in tensors.items():
        if key in {_HEAD, _EMBED, _NORM} or (key.startswith(_VISION) and key.endswith((".weight", ".bias"))):
            continue
        _require(key.startswith(_LAYERS), f"Unexpected tensor namespace: {key}")
        number, _, suffix = key.removeprefix(_LAYERS).partition(".")
        _require(number.isascii() and number.isdigit() and str(int(number)) == number and int(number) < n,
                 f"Invalid text-layer index: {key}")
        layers[int(number)][suffix] = tensor
    for i, layer in enumerate(layers):
        schema = full if i % 4 == 3 else linear
        _require(layer.keys() == schema.keys(), f"Wrong parameter set at text layer {i}")
        for key, shape in schema.items():
            _require(layer[key].shape == shape, f"Wrong shape at text layer {i}: {key}")
            _require(layer[key].dtype == layers[i % 4][key].dtype, f"Incompatible same-phase dtype: {key}")
    return layers


def _ties(config: dict, tensors: dict[str, Tensor]) -> dict:
    head_stored = _HEAD in tensors
    return {
        "embedding": _EMBED, "head": _HEAD,
        "configured_tied": config["text_config"]["tie_word_embeddings"],
        "both_separately_stored": head_stored,
        "stored_values_bit_identical": _identical(tensors[_EMBED], tensors[_HEAD]) if head_stored else None,
    }


@dataclass(frozen=True)
class Group:
    cycles: tuple[int, ...]
    coefficients: tuple[float, ...]
    anchor: int


def _plan(plan: dict, source_cycles: int) -> tuple[list[Group], tuple[int, int]]:
    _require(set(plan) == {"format", "groups", "repeat_cycles"} and plan["format"] == _FORMAT,
             "Expected a joey-compact/v1 plan")
    groups = plan["groups"]
    _require(isinstance(groups, list) and 0 < len(groups) < source_cycles, "Must physically remove cycles")
    checked = []
    previous = -1
    for group in groups:
        _require(isinstance(group, dict) and group.keys() <= {"cycles", "weights", "anchor"}, "Unknown plan field")
        cycles = group.get("cycles")
        _require(isinstance(cycles, list) and bool(cycles) and all(type(i) is int for i in cycles),
                 "Cycle IDs must be non-empty lists of integers")
        _require(previous < cycles[0] and cycles == sorted(set(cycles)) and cycles[-1] < source_cycles,
                 "Source cycles must be unique, in bounds and in depth order")
        previous = cycles[-1]
        anchor = group.get("anchor", cycles[0])
        _require(type(anchor) is int and anchor in cycles, "Anchor must be one of the contributing cycles")
        weights = group.get("weights")
        coefficients: tuple[float, ...] = ()
        if len(cycles) == 1:
            _require("weights" not in group, "Single source groups use byte-preserving selection")
        else:
            _require(isinstance(weights, list) and len(weights) == len(cycles) and
                     all(type(w) in (float, int) and 0 < w <= 1 for w in weights),
                     "Each merged source needs a positive finite coefficient")
            _require(abs(math.fsum(weights) - 1.0) <= 1e-7, "Merge coefficients must sum to one")
            coefficients = tuple(_f32(w) for w in weights)
            _require(all(w > 0 for w in coefficients), "Coefficient underflows FP32")
        checked.append(Group(tuple(cycles), coefficients, anchor))
    span = plan["repeat_cycles"]
    _require(isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span) and
             0 <= span[0] < span[1] <= len(checked), "Invalid half-open compact repeat-cycle range")
    return checked, (4 * span[0], 4 * span[1])


def _outputs(layers: list[dict[str, Tensor]], tensors: dict[str, Tensor], groups: list[Group]
             ) -> tuple[dict[str, OutputTensor], list[int]]:
    result = {key: OutputTensor((t,)) for key, t in tensors.items() if not key.startswith(_LAYERS)}
    anchors = []
    for gi, group in enumerate(groups):
        for phase in range(4):
            sources = [4 * cycle + phase for cycle in group.cycles]
            anchor = 4 * group.anchor + phase
            anchors.append(anchor)
            for suffix, tensor in layers[anchor].items():
                if group.coefficients and suffix in _MERGE_SUFFIXES:
                    output = OutputTensor(tuple(layers[i][suffix] for i in sources), group.coefficients)
                else:
                    output = OutputTensor((tensor,))
                result[f"{_LAYERS}{4 * gi + phase}.{suffix}"] = output
    return result, anchors


def _storage(tensors: Mapping[str, Tensor], weight_files: Sequence[Path], tied: bool) -> dict:
    def sum_storage(items: Sequence[Tensor]) -> dict[str, int]:
        return {"tensors": len(items), "stored_parameter_elements": sum(t.numel for t in items),
                "tensor_bytes": sum(t.nbytes for t in items)}

    text = sum_storage([t for k, t in tensors.items() if not k.startswith(_VISION)])
    vision = sum_storage([t for k, t in tensors.items() if k.startswith(_VISION)])
    text["parameter_elements_assuming_configured_ties"] = text["stored_parameter_elements"] - (
        tensors[_HEAD].numel if tied and _HEAD in tensors else 0
    )
    return {"weight_file_bytes": sum(p.stat().st_size for p in weight_files),
            "full": sum_storage(list(tensors.values())), "text": text, "vision": vision}


def layer_schedule(manifest: dict, k: int) -> tuple[int, ...]:
    """Physical layer IDs for a prefill: prefix, k visits to the body, suffix. No weights are cloned."""
    _require(type(k) is int and 1 <= k <= 32, "k must be an integer in [1,32]")
    _require(manifest.get("format") == _FORMAT, "Unknown manifest format")
    n = manifest.get("physical_depth")
    span = manifest.get("repeat_layers")
    _require(type(n) is int and 0 < n <= 24 and n % 4 == 0 and isinstance(span, list) and
             len(span) == 2 and all(type(i) is int and i % 4 == 0 for i in span) and
             0 <= span[0] < span[1] <= n, "Invalid compact schedule")
    lo, hi = span
    return tuple(range(lo)) + tuple(range(lo, hi)) * k + tuple(range(hi, n))


def measure_text_storage(tensors: Mapping[str, Any]) -> dict:
    """Count resident contiguous text parameters/views supplied by a loaded runner, including external readout.

    Values expose the PyTorch tensor storage interface. This measures backing storage, not allocator
    reservation, temporaries, or activation memory. Nothing is copied or moved.
    """
    _require(bool(tensors), "Pass the deployed text parameter and readout tensors")
    spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    layouts: dict[tuple[str, int], tuple[int, str, int]] = {}
    aliases: dict[tuple[str, int, int, int], list[str]] = {}
    named_elements = named_bytes = 0
    for name, tensor in tensors.items():
        _require(tensor.is_contiguous() and str(tensor.device) != "meta", "Expected a resident contiguous tensor")
        storage = tensor.untyped_storage()
        width, size, count = tensor.element_size(), storage.nbytes(), tensor.numel()
        start = tensor.storage_offset() * width
        end = start + count * width
        _require(width > 0 and 0 <= start <= end <= size, "Invalid tensor view extent")
        key = (str(tensor.device), storage.data_ptr())
        layout = (width, str(tensor.dtype), size)
        _require(layouts.setdefault(key, layout) == layout, "Mixed dtype or inconsistent shared storage")
        spans.setdefault(key, []).append((start, end))
        aliases.setdefault((*key, start, end), []).append(name)
        named_elements += count
        named_bytes += count * width
    elements = referenced = 0
    for key, ranges in spans.items():
        last = covered = 0
        for start, end in sorted(ranges):
            covered += max(0, end - max(last, start))
            last = max(last, end)
        referenced += covered
        elements += covered // layouts[key][0]
    return {
        "named_elements": named_elements, "named_bytes": named_bytes, "unique_referenced_elements": elements,
        "unique_referenced_bytes": referenced, "backing_storage_bytes": sum(t[2] for t in layouts.values()),
        "exact_alias_groups": sorted(sorted(names) for names in aliases.values() if len(names) > 1),
    }


def _source_versions(root: Path) -> dict:
    paths = set(root.glob("*.safetensors")) | {root / "config.json"}
    paths.update(root / name for name in _SIDECARS | {"model.safetensors.index.json"} if os.path.lexists(root / name))
    versions = {}
    for path in paths:
        s = _regular(path).stat()
        versions[path.name] = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    return versions


def export_compact(source_dir: str | Path, output_dir: str | Path, plan: str | Path | dict, *,
                   fixture: bool = False) -> dict:
    """Create one independent compact bundle. Real input must match the pinned Joey-1 bundle weights/config."""
    source = Path(source_dir).resolve(strict=True)
    dest_arg = Path(output_dir)
    _require(not os.path.lexists(dest_arg), "Destination must be a new directory")
    dest = dest_arg.parent.resolve(strict=True) / dest_arg.name
    _require(source.is_dir() and not dest.is_relative_to(source) and not source.is_relative_to(dest),
             "Source and destination must be disjoint directories")
    versions = _source_versions(source)
    config = _json_file(source / "config.json")
    config_hash = hashlib.sha256(_encode(config)).hexdigest()
    if fixture:
        _require(config.get("joey_generated_fixture") is True, "Fixture mode requires a marked generated config")
    else:
        _require(config_hash == _PIN_CONFIG, "Not the pinned Joey-1 config")
    checkpoint = Checkpoint.open(source)
    hashes = {p.name: _digest(p) for p in checkpoint.files}
    if not fixture:
        _require(hashes == {"model.safetensors": _PIN_WEIGHTS}, "Not the pinned Joey-1 weights")
    layers = _layout(config, checkpoint)
    instructions = copy.deepcopy(plan) if isinstance(plan, dict) else _json_file(Path(plan))
    groups, span = _plan(instructions, len(layers) // 4)
    outputs, anchors = _outputs(layers, checkpoint.tensors, groups)
    ties = _ties(config, checkpoint.tensors)
    original_storage = _storage(checkpoint.tensors, checkpoint.files, ties["configured_tied"])
    compact_config = copy.deepcopy(config)
    text = compact_config["text_config"]
    text["num_hidden_layers"] = len(anchors)
    text["layer_types"] = [text["layer_types"][i] for i in anchors]
    # The temporary directory lives on the destination filesystem; failed validation leaves no export.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".joey-export-") as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        weight_file = staged / "model.safetensors"
        weight_hash = _write_weights(weight_file, outputs, checkpoint.aliases)
        config_bytes = _encode(compact_config, pretty=True)
        (staged / "config.json").write_bytes(config_bytes)
        copied = {}
        for name in sorted(_SIDECARS):
            if os.path.lexists(source / name):
                with (staged / name).open("xb") as target:
                    copied[name] = _digest(source / name, target)
        saved = Checkpoint.open(staged)
        _layout(compact_config, saved)
        storage = _storage(saved.tensors, saved.files, ties["configured_tied"])
        _require(storage["weight_file_bytes"] < original_storage["weight_file_bytes"] and
                 storage["full"]["stored_parameter_elements"] < original_storage["full"]["stored_parameter_elements"],
                 "Export did not physically reduce the checkpoint")
        manifest = {
            "format": _FORMAT, "source": {
                "kind": "generated-fixture" if fixture else "pinned-joey-1",
                "model": None if fixture else MODEL, "revision": None if fixture else REVISION,
                "canonical_config_sha256": config_hash, "weight_file_sha256": hashes,
                "depth": len(layers), "storage": original_storage,
            },
            "plan": instructions, "physical_depth": len(anchors), "anchor_source_layers": anchors,
            "repeat_layers": list(span), "merge_suffixes": sorted(_MERGE_SUFFIXES),
            "ties": ties, "storage": storage, "weights_sha256": weight_hash,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(), "sidecar_sha256": copied,
        }
        (staged / "compact.json").write_bytes(_encode(manifest, pretty=True))
        _require(_source_versions(source) == versions, "Source changed during export")
        _require(not os.path.lexists(dest), "Destination already exists")
        staged.rename(dest)
    return manifest


def verify_compact(directory: str | Path) -> dict:
    """Reopen a local export, checking hashes, topology, ties, storage and its schedule (not runner parity)."""
    root = Path(directory)
    manifest = _json_file(root / "compact.json")
    _require(manifest.get("format") == _FORMAT, "Unknown compact manifest format")
    config = _json_file(root / "config.json")
    _require(_digest(root / "config.json") == manifest.get("config_sha256"), "Compact config digest mismatch")
    checkpoint = Checkpoint.open(root)
    _require(checkpoint.files == (root / "model.safetensors",) and
             _digest(checkpoint.files[0]) == manifest.get("weights_sha256"), "Compact weight digest mismatch")
    layers = _layout(config, checkpoint)
    _require(len(layers) == manifest.get("physical_depth"), "Compact depth mismatch")
    source = manifest.get("source")
    _require(isinstance(source, dict) and type(source.get("depth")) is int and
             0 < source["depth"] <= 24 and source["depth"] % 4 == 0, "Invalid source depth")
    if source.get("kind") == "generated-fixture":
        _require(config.get("joey_generated_fixture") is True, "Incorrect fixture provenance")
    else:
        _require(source.get("kind") == "pinned-joey-1" and source.get("model") == MODEL and
                 source.get("revision") == REVISION and source.get("canonical_config_sha256") == _PIN_CONFIG and
                 source.get("weight_file_sha256") == {"model.safetensors": _PIN_WEIGHTS},
                 "Incorrect source provenance")
        restored = copy.deepcopy(config)
        restored["text_config"].update(num_hidden_layers=source["depth"], layer_types=list(_CYCLE) * (source["depth"] // 4))
        _require(hashlib.sha256(_encode(restored)).hexdigest() == _PIN_CONFIG, "Incorrect source configuration")
    _require(isinstance(manifest.get("plan"), dict), "Missing compact plan")
    groups, span = _plan(manifest["plan"], source["depth"] // 4)
    anchors = [4 * group.anchor + phase for group in groups for phase in range(4)]
    _require(anchors == manifest.get("anchor_source_layers") and len(anchors) == len(layers) and
             list(span) == manifest.get("repeat_layers") and manifest.get("merge_suffixes") == sorted(_MERGE_SUFFIXES),
             "Manifest topology does not match plan")
    ties = _ties(config, checkpoint.tensors)
    _require(manifest.get("ties") == ties, "Compact tie data mismatch")
    _require(manifest.get("storage") == _storage(checkpoint.tensors, checkpoint.files, ties["configured_tied"]),
             "Compact storage mismatch")
    sidecars = manifest.get("sidecar_sha256")
    _require(isinstance(sidecars, dict) and sidecars.keys() == {n for n in _SIDECARS if os.path.lexists(root / n)},
             "Compact sidecar set mismatch")
    for name, digest in sidecars.items():
        _require(_digest(root / name) == digest, f"Compact sidecar digest mismatch: {name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("source")
    export.add_argument("destination")
    export.add_argument("plan")
    export.add_argument("--fixture", action="store_true", help="require a marked synthetic source instead of the pin")
    verify = commands.add_parser("verify")
    verify.add_argument("directory")
    verify.add_argument("--k", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_compact(args.source, args.destination, args.plan, fixture=args.fixture)
            print(_encode(result, pretty=True).decode(), end="")
        else:
            result = verify_compact(args.directory)
            print(json.dumps({"weights_sha256": result["weights_sha256"], "physical_ids": layer_schedule(result, args.k)}))
    except (CompactError, OSError) as exc:
        parser.exit(1, f"compact: {exc}\n")


if __name__ == "__main__":
    main()
