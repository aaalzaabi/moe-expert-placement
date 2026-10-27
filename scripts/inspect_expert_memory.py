
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/phase4f/memory/expert_tensor_metadata.csv', 'outputs/phase4f/memory/expert_memory_summary.csv'])
# ---------------------------------------------------------------------------------------

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from safetensors import safe_open

MODEL_DIR = Path(
    "models/OLMoE-1B-7B-0125-Instruct"
)

INDEX = MODEL_DIR / "model.safetensors.index.json"

# Bytes for common safetensors dtypes
DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result

with open(INDEX) as f:
    index = json.load(f)

shards = sorted(set(index["weight_map"].values()))

records = []

print("Inspecting OLMoE checkpoint metadata...")

for shard in shards:
    with safe_open(
        str(MODEL_DIR / shard),
        framework="pt",
        device="cpu",
    ) as reader:

        for name in reader.keys():

            if "experts" not in name.lower():
                continue

            match = re.search(
                r"layers\.(\d+)\.",
                name
            )

            if not match:
                continue

            layer_id = int(match.group(1))

            # Reads metadata without materializing weights.
            tensor_slice = reader.get_slice(name)

            shape = tensor_slice.get_shape()
            dtype = tensor_slice.get_dtype()

            if dtype not in DTYPE_BYTES:
                raise ValueError(
                    f"Unsupported dtype {dtype}: {name}"
                )

            nbytes = (
                product(shape) * DTYPE_BYTES[dtype]
            )

            records.append({
                "layer_id": layer_id,
                "tensor_name": name,
                "shape": str(shape),
                "dtype": dtype,
                "bytes": nbytes,
                "size_mb": nbytes / 1e6,
            })

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No expert tensors found. Inspect checkpoint names."
    )

layer_summary = (
    df.groupby("layer_id", as_index=False)
      .agg(
          expert_tensors=("tensor_name", "count"),
          expert_bytes=("bytes", "sum")
      )
)

layer_summary["expert_mb"] = (
    layer_summary["expert_bytes"] / 1e6
)

# Estimate based on 64 equal-sized experts per layer.
layer_summary["per_expert_mb_estimate"] = (
    layer_summary["expert_mb"] / 64
)

total_bytes = df["bytes"].sum()

print("\nEXPERT MEMORY SUMMARY")
print(layer_summary.to_string(index=False))

print("\nTotal stored expert weights:")
print(f"{total_bytes / 1e9:.3f} GB")

print("\nAverage stored bytes per expert slot:")
print(f"{total_bytes / 1024 / 1e6:.3f} MB")

print("\nLayers found:", df.layer_id.nunique())

assert df.layer_id.nunique() == 16, (
    "Expected 16 layers; inspect tensor naming."
)

OUT = Path("outputs/phase4f/memory")
OUT.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUT / "expert_tensor_metadata.csv",
    index=False,
)

layer_summary.to_csv(
    OUT / "expert_memory_summary.csv",
    index=False,
)

print("\nFiles saved successfully.")

