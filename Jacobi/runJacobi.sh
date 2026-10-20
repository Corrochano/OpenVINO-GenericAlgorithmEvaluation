#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

runs="${RUNS:-1}"
iterations=100
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer.\n' >&2
    exit 2
fi

sizes=(256 512 1024 2048 4096 8192)
datatypes=(fp32 fp16)
devices=(cpu gpu npu)
failures=0

for size in "${sizes[@]}"; do
    for datatype in "${datatypes[@]}"; do
        model_file="$script_dir/jacobi${size}_model_${datatype}.xml"
        if [[ ! -f "$model_file" ]]; then
            printf '\n[SKIP] Missing Jacobi model: %s\n' "$model_file"
            continue
        fi
        for device in "${devices[@]}"; do
            printf '\n[INFO] Running %s Jacobi %s (%s runs)\n' "$device" "$size" "$datatype" "$runs"
            if ! python3 "$script_dir/executeOpenVINO.py" "$size" "$iterations" "$runs" "$datatype" "$device"; then
                printf '[ERROR] Failed: %s %s %s\n' "$device" "$size" "$datatype" >&2
                failures=$((failures + 1))
            fi
        done
    done
done

if (( failures > 0 )); then
    printf '\n[ERROR] %d Jacobi run(s) failed.\n' "$failures" >&2
    exit 1
fi

printf '\n+++ All Jacobi runs completed +++\n'