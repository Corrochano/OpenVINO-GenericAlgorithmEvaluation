#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

runs="${RUNS:-1}"
iterations=100
measure_energy="${MEASURE_ENERGY:-0}"
devices_text="${DEVICES:-cpu gpu}"
read -r -a devices <<< "$devices_text"
if [[ ! "$runs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'RUNS must be a positive integer.\n' >&2
    exit 2
fi
if [[ "$measure_energy" != 0 && "$measure_energy" != 1 ]]; then
    printf 'MEASURE_ENERGY must be 0 or 1.\n' >&2
    exit 2
fi
if (( ${#devices[@]} == 0 )); then
    printf 'DEVICES must contain at least one device.\n' >&2
    exit 2
fi

if [[ "$measure_energy" == 1 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
        printf 'sudo is required when MEASURE_ENERGY=1.\n' >&2
        exit 2
    fi
    sudo -v || exit 2
fi

sizes=(256 512 1024 2048 4096 8192)
datatypes=(fp32 fp16)
failures=0

for size in "${sizes[@]}"; do
    for datatype in "${datatypes[@]}"; do
        model_file="$script_dir/multigrid${size}_model_${datatype}.xml"
        if [[ ! -f "$model_file" ]]; then
            printf '\n[SKIP] Missing Multigrid model: %s\n' "$model_file"
            continue
        fi
        for device in "${devices[@]}"; do
            printf '\n[INFO] Running %s Multigrid %s (%s, %s runs)\n' "$device" "$size" "$datatype" "$runs"
            if [[ "$measure_energy" == 1 ]] && ! sudo -v; then
                printf '[ERROR] Could not refresh sudo authorization for RAPL reads.\n' >&2
                failures=$((failures + 1))
                continue
            fi
            if ! python3 "$script_dir/executeOpenVINO.py" "$size" "$iterations" "$runs" "$datatype" "$device"; then
                printf '[ERROR] Failed: %s %s %s\n' "$device" "$size" "$datatype" >&2
                failures=$((failures + 1))
            fi
        done
    done
done

if (( failures > 0 )); then
    printf '\n[ERROR] %d Multigrid run(s) failed.\n' "$failures" >&2
    exit 1
fi

printf '\n+++ All Multigrid runs completed +++\n'
