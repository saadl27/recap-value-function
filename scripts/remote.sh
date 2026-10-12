#!/usr/bin/env bash
# Optional laptop → workstation helper for launching and monitoring full runs. Requires key-based SSH.
# Code reaches the workstation through GitHub (`git pull`), never by copying files.
#
#   scripts/remote.sh update               git pull --ff-only in the workstation clone
#   scripts/remote.sh run  "<cmd>"         run <cmd> in the workstation repo and wait (short jobs)
#   scripts/remote.sh bg   <name> "<cmd>"  start <cmd> detached; log: runs/logs/<name>.log, pid: runs/logs/<name>.pid
#   scripts/remote.sh log  <name> [n]      tail the last n (default 50) lines of a background job's log
#   scripts/remote.sh jobs                 list background jobs started with `bg` and whether they still run
#   scripts/remote.sh gpu                  GPU memory/utilisation and every process using it (shared GPU!)
#   scripts/remote.sh fetch-reports        copy reports/ from the workstation into the laptop repo
#
# Target: RECAP_REMOTE_HOST (default saad@curiosity.local), RECAP_REMOTE_DIR (default DreamMachines, under $HOME).
set -euo pipefail

HOST="${RECAP_REMOTE_HOST:-saad@curiosity.local}"
DIR="${RECAP_REMOTE_DIR:-DreamMachines}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST")

# Non-interactive SSH shells do not source ~/.bashrc: put uv on PATH explicitly.
PREAMBLE="export PATH=\$HOME/.local/bin:\$PATH; cd ~/$DIR"

cmd="${1:-}"
shift || true
case "$cmd" in
    update)
        "${SSH[@]}" "$PREAMBLE && git pull --ff-only"
        ;;
    run)
        "${SSH[@]}" "$PREAMBLE && $1"
        ;;
    bg)
        name="$1"
        job="$2"
        "${SSH[@]}" "$PREAMBLE && mkdir -p runs/logs && \
            nohup bash -c $(printf %q "$job") > runs/logs/$name.log 2>&1 < /dev/null & \
            echo \$! > runs/logs/$name.pid && echo started $name pid \$(cat runs/logs/$name.pid)"
        ;;
    log)
        "${SSH[@]}" "$PREAMBLE && tail -n ${2:-50} runs/logs/$1.log"
        ;;
    jobs)
        "${SSH[@]}" "$PREAMBLE && for p in runs/logs/*.pid; do [ -e \"\$p\" ] || continue; \
            n=\$(basename \$p .pid); pid=\$(cat \$p); \
            if kill -0 \$pid 2>/dev/null; then s=running; else s=finished; fi; echo \"\$n \$pid \$s\"; done"
        ;;
    gpu)
        "${SSH[@]}" "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; \
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; \
            ps -o user=,pid=,cmd= -p \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -) 2>/dev/null || true"
        ;;
    fetch-reports)
        rsync -az "$HOST:$DIR/reports/" "$REPO/reports/"
        ;;
    *)
        sed -n '2,13p' "$0"
        exit 1
        ;;
esac
