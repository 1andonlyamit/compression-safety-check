#!/usr/bin/env bash
# Wait for a Kaggle kernel to finish, surviving network drops, then download its output.
# Usage: scripts/kaggle_wait.sh <slug> <dest_dir> [max_minutes]
slug="$1"; dest="$2"; max="${3:-720}"; user="${KAGGLE_USER:-amitmakar}"
mkdir -p "$dest"
for i in $(seq 1 "$max"); do
  s=$(python -m kaggle kernels status "$user/$slug" 2>&1 | tail -1)
  echo "$(date +%H:%M) $s" >> "$dest/poll.log"
  # Only Kaggle's own final states count; network errors (e.g. "ConnectTimeoutError") must not.
  echo "$s" | grep -qE 'KernelWorkerStatus\.(COMPLETE|ERROR|CANCEL)' && break
  sleep 60
done
for t in 1 2 3 4 5 6; do
  python -m kaggle kernels output "$user/$slug" -p "$dest" -o > "$dest/download.log" 2>&1 && break
  sleep 60
done
tail -1 "$dest/poll.log"
ls "$dest"
