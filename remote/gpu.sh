#!/usr/bin/env bash
# Isolated GPU training for SAGA on a remote NVIDIA box.
#
# Everything on the remote lives in $REMOTE_DIR (own venv, own HF cache, own runs/).
# Nothing outside that folder is read or written. Results come back into
# remote_runs/<run>/ locally, separate from the Mac's checkpoints/ and results/.
#
#   remote/gpu.sh check            # GPU, disk, and whether $REMOTE_DIR already exists
#   remote/gpu.sh push             # copy training code + labelled data (train/val only)
#   remote/gpu.sh setup            # venv + CUDA PyTorch + model weights, then a GPU smoke test
#   remote/gpu.sh train [run] [args...]   # start training in the background (survives disconnect)
#   remote/gpu.sh status [run]     # tail the training log
#   remote/gpu.sh eval [run]       # test-set evaluation of the run's best checkpoint
#   remote/gpu.sh pull [run]       # bring back best checkpoint, logs and metrics
set -euo pipefail

HOST=${AG_GPU_HOST:?set AG_GPU_HOST=user@gpu-host}
PORT=${AG_GPU_PORT:-22}
KEY=${AG_GPU_KEY:-$HOME/.ssh/id_ed25519}
REMOTE_DIR=${AG_REMOTE_DIR:-/root/saga-train}
LOCAL=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ServerAliveInterval=30 "$HOST")
RSYNC=(rsync -az --stats -e "ssh -p $PORT -i $KEY -o IdentitiesOnly=yes")  # macOS openrsync-compatible

cmd=${1:-help}; shift || true
run=${1:-b2_8k}; [[ $# -gt 0 ]] && shift

case "$cmd" in
check)
  "${SSH[@]}" "set -e
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version --format=csv
    echo 'cpus:' \$(nproc); free -g | sed -n 2p; df -h / | tail -1
    python3 --version; command -v uv || true
    echo 'other GPU jobs:'; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    if [ -e '$REMOTE_DIR' ]; then echo 'WARNING: $REMOTE_DIR already exists:'; ls -la '$REMOTE_DIR'; else echo '$REMOTE_DIR is free'; fi"
  ;;
push)
  "${SSH[@]}" "mkdir -p '$REMOTE_DIR/data/10k' '$REMOTE_DIR/data/bdd100k_seg_maps'"
  printf '%s\n' ag/__init__.py ag/classes.py ag/data.py ag/model.py ag/audit.py \
    scripts/train.py scripts/eval_seg.py scripts/eval_audit.py data/splits.json remote/requirements.txt \
    > "${TMPDIR:-/tmp}/ag_push_list"
  "${RSYNC[@]}" --files-from="${TMPDIR:-/tmp}/ag_push_list" "$LOCAL/" "$HOST:$REMOTE_DIR/"
  # labelled images only: official train (train + holdout) and official val (our test)
  "${RSYNC[@]}" "$LOCAL/data/10k/train" "$LOCAL/data/10k/val" "$HOST:$REMOTE_DIR/data/10k/"
  "${RSYNC[@]}" "$LOCAL/data/bdd100k_seg_maps/labels" "$HOST:$REMOTE_DIR/data/bdd100k_seg_maps/"
  "${SSH[@]}" "cd '$REMOTE_DIR' && echo files: \$(find data -type f | wc -l) && du -sh ."
  ;;
setup)
  "${SSH[@]}" "set -e; cd '$REMOTE_DIR'
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\$HOME/.local/bin:\$PATH
    uv venv -q --python 3.12 .venv
    uv pip install -q --python .venv/bin/python -r remote/requirements.txt
    export HF_HOME=\$PWD/.cache/hf
    .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(\"torch\", torch.__version__, torch.cuda.get_device_name(0))'
    .venv/bin/python -c 'import sys; sys.path.insert(0, \".\"); from ag.model import load; m = load(); print(\"weights ok on\", next(m.parameters()).device)'"
  ;;
train)
  # default: the same recipe as the Mac run, fresh from Cityscapes weights, bf16 on CUDA
  args=${*:---iters 8000 --bs 8 --crop 512 --amp}
  "${SSH[@]}" "set -e; cd '$REMOTE_DIR'; mkdir -p runs/$run
    # pid file, not pgrep: pgrep -f would match this very ssh command line
    if [ -f runs/$run/pid ] && kill -0 \$(cat runs/$run/pid) 2>/dev/null; then echo 'run $run is already training'; exit 1; fi
    [ -e runs/$run/last ] && { echo 'runs/$run already has checkpoints; pick a new run name'; exit 1; }
    export HF_HOME=\$PWD/.cache/hf
    W=\$(( \$(nproc) > 16 ? 16 : \$(nproc) - 1 ))
    nohup setsid .venv/bin/python scripts/train.py --out runs/$run --workers \$W $args > runs/$run/train.out 2>&1 &
    echo \$! > runs/$run/pid
    echo started run $run with \$W loader workers: $args"
  ;;
status)
  "${SSH[@]}" "cd '$REMOTE_DIR'; grep -h eval runs/$run/log.jsonl | tail -3; tail -2 runs/$run/log.jsonl; grep -iE 'error|traceback' runs/$run/train.out | tail -3 || true
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
  ;;
eval)
  "${SSH[@]}" "set -e; cd '$REMOTE_DIR'; export HF_HOME=\$PWD/.cache/hf; mkdir -p results
    .venv/bin/python scripts/eval_seg.py --model runs/$run/best --name $run --tta --save-preds
    .venv/bin/python scripts/eval_audit.py --name $run"
  ;;
pull)
  mkdir -p "$LOCAL/remote_runs/$run"
  "${RSYNC[@]}" "$HOST:$REMOTE_DIR/runs/$run/best" "$HOST:$REMOTE_DIR/runs/$run/log.jsonl" \
    "$HOST:$REMOTE_DIR/runs/$run/train.out" "$LOCAL/remote_runs/$run/"
  "${RSYNC[@]}" "$HOST:$REMOTE_DIR/results/seg_${run}_test.json" "$HOST:$REMOTE_DIR/results/audit_${run}_test.json" \
    "$LOCAL/remote_runs/$run/" || echo "(no test metrics yet; run: remote/gpu.sh eval $run)"
  ls -la "$LOCAL/remote_runs/$run"
  ;;
*)
  sed -n '2,15p' "$0"
  ;;
esac
