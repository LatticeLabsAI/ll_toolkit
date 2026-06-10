#!/usr/bin/env bash
# gpu-smi.sh — an nvidia-smi-style live GPU monitor for Apple Silicon (no sudo).
#
# macOS has no nvidia-smi, but the Apple GPU (AGXAccelerator) exposes the same
# live counters through ioreg. This prints GPU utilization %, GPU (unified)
# memory in use, and the busiest python/torch process, refreshing on an interval.
#
# Usage:
#   scripts/gpu-smi.sh           # refresh every 1s
#   scripts/gpu-smi.sh 2         # refresh every 2s
#
# For a richer TUI (GPU/CPU/ANE/power) install/run `sudo mactop`, or
# `sudo powermetrics --samplers gpu_power -i 1000`.
set -euo pipefail

INTERVAL="${1:-1}"

read_stat() {
    # $1 = ioreg PerformanceStatistics key; echoes the integer value (or empty).
    ioreg -r -c AGXAccelerator -d 1 -w 0 2>/dev/null \
        | grep -ao "\"$1\"=[0-9]*" | head -1 | cut -d= -f2
}

while true; do
    util="$(read_stat 'Device Utilization %')"
    mem_bytes="$(read_stat 'In use system memory')"
    alloc_bytes="$(read_stat 'Alloc system memory')"
    mem_gb="n/a"; alloc_gb="n/a"
    [ -n "${mem_bytes:-}" ] && mem_gb="$(awk -v b="$mem_bytes" 'BEGIN{printf "%.1f", b/1073741824}')"
    [ -n "${alloc_bytes:-}" ] && alloc_gb="$(awk -v b="$alloc_bytes" 'BEGIN{printf "%.1f", b/1073741824}')"

    # busiest python/torch process (training/inference)
    proc="$(ps aux | grep -iE 'python|torch' | grep -v grep \
            | sort -k3 -rn | head -1 \
            | awk '{printf "pid=%s cpu=%s%% mem=%s%%", $2, $3, $4}')"

    printf '\033[2J\033[H'  # clear screen
    printf 'gpu-smi (Apple Silicon AGXAccelerator)        %s\n' "$(date '+%H:%M:%S')"
    printf '%s\n' '--------------------------------------------------------------'
    printf '  GPU util        : %s%%\n' "${util:-?}"
    printf '  GPU mem in use  : %s GB\n' "$mem_gb"
    printf '  GPU mem alloc   : %s GB\n' "$alloc_gb"
    printf '  top proc        : %s\n' "${proc:-none}"
    printf '%s\n' '--------------------------------------------------------------'
    printf '  refresh %ss · Ctrl-C to exit · richer view: sudo mactop\n' "$INTERVAL"
    sleep "$INTERVAL"
done
