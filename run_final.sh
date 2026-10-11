#!/bin/zsh
# Final evaluation + platform data, run after training finishes.
set -e
cd "$(dirname "$0")"
export HF_HOME=$PWD/.cache/hf
PY=.venv/bin/python
CKPT=checkpoints/b2_bdd/best
$PY scripts/eval_seg.py --model $CKPT --name finetuned --tta --save-preds
$PY scripts/eval_seg.py --name baseline --split holdout
$PY scripts/eval_audit.py --name baseline
$PY scripts/eval_audit.py --name finetuned
$PY scripts/make_gallery.py --name finetuned
$PY scripts/predict_map.py --model $CKPT --tta
