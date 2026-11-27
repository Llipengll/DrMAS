#!/usr/bin/env bash
# CPU-only DrMAS checks on the Python 3.12 Ascend 950 development image.
set -euo pipefail
if (( $# == 0 )); then
    echo 'Usage: scripts/run_950dt_cpu.sh COMMAND [ARG ...]' >&2
    exit 2
fi
if compgen -G '/dev/*davinci*' >/dev/null; then
    echo 'Run this CPU profile in a container with no NPU devices mounted.' >&2
    exit 1
fi
devlib=/usr/local/Ascend/cann-9.1.0/aarch64-linux/devlib
if [[ ! -r "$devlib/libascend_hal.so" ]]; then
    echo "Missing CANN development library: $devlib/libascend_hal.so" >&2
    exit 1
fi
# Transformers 4.53 probes torch_npu while importing its CPU model classes.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$devlib"
exec "$@"
