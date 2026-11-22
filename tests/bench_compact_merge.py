"""Benchmark the production scalar merge on generated BF16 files, not a model.

Run: PYTHONPATH=src python -B tests/bench_compact_merge.py
Prints JSON. No checkpoint, package install, device, or network is required.
"""

import argparse
import gc
import json
from pathlib import Path
from statistics import median
import struct
import sys
import tempfile
from time import perf_counter
import tracemalloc

from joey_research.compact import OutputTensor, Tensor, _MERGE_ELEMENTS, _write_weights


def source(path, rows, cols, seed):
    words = []
    for i in range(4096):
        sign = ((i + seed) & 1) << 15
        exponent = (113 + (37*i + 53*seed) % 27) << 7
        words.append(sign | exponent | ((7*i + seed) % 128))
    tile = struct.pack("<4096H", *words)
    total = 2 * rows * cols
    with path.open("xb") as stream:
        for pos in range(0, total, len(tile)):
            stream.write(tile[:min(len(tile), total - pos)])
    return Tensor(path, 0, "BF16", (rows, cols), total)


def measure(directory, shape, trials):
    rows, cols = shape
    count = rows * cols
    inputs = tuple(source(directory / f"source-{i}.bin", rows, cols, i) for i in range(2))
    outputs = {"matrix": OutputTensor(inputs, (0.25, 0.75))}
    path = directory / "generated.safetensors"
    digest = _write_weights(path, outputs, {})  # warm input pages and the interpreter
    path.unlink()

    def once():
        start = perf_counter()
        observed = _write_weights(path, outputs, {})
        seconds = perf_counter() - start
        if observed != digest:
            raise RuntimeError("Inconsistent output across trials")
        path.unlink()
        return seconds

    seconds = [once() for _ in range(trials)]
    gc.collect()
    tracemalloc.start()
    try:
        once()  # separate, untimed allocation probe
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "shape": shape, "outputs": count, "seconds": seconds,
        "median_outputs_per_s": count / median(seconds), "slowest_outputs_per_s": count / max(seconds),
        "peak_traced_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, choices=range(3, 10), default=5)
    args = parser.parse_args()
    results = []
    for shape in [(8193, 31), (128, 2048), (43, 6144), (512, 2048)]:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1], prefix=".bench-compact-") as tmp:
            results.append(measure(Path(tmp), shape, args.trials))
    print(json.dumps({
        "python": sys.version.split()[0], "chunk_elements": _MERGE_ELEMENTS, "input_dtype": "BF16",
        "contributors": 2, "cases": results, "sample_floor_outputs_per_s": min(
            r["slowest_outputs_per_s"] for r in results),
        "scope": "Single matrix read, scalar merge, safetensors write and SHA-256. Warm pages, no fsync. "
                 "Timing excludes fixture generation and tracing. Peak is traced Python bytes, not RSS.",
    }, indent=2))


if __name__ == "__main__":
    main()
