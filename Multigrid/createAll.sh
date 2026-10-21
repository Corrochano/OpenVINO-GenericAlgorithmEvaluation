#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

sizes=(256 512 1024 2048 4096 8192)
datatypes=(fp32 fp16)
iterations=100

for size in "${sizes[@]}"; do
    for datatype in "${datatypes[@]}"; do
        printf '\n[INFO] Creating Multigrid model %s (%s)\n' "$size" "$datatype"
        python3 "$script_dir/jacobiConversion.py" "$size" "$iterations" "$datatype"
    done
done
