#!/usr/bin/env bash
set -euo pipefail

# Example: self-distillation on top of slime (Megatron + SGLang, colocate).
# Update all paths and model arguments before running.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXT_DIR="${SCRIPT_DIR}"

export PYTHONPATH="${SLIME_DIR}:${EXT_DIR}:${PYTHONPATH:-}"

cd "${SLIME_DIR}"

python train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --num-gpus-per-node 4 \
  --colocate \
  --train-backend megatron \
  --num-rollout 100 \
  --rollout-num-gpus 4 \
  --rollout-num-gpus-per-engine 4 \
  --n-samples-per-prompt 8 \
  --rollout-batch-size 32 \
  --micro-batch-size 1 \
  --global-batch-size 32 \
  --advantage-estimator grpo \
  --use-opd \
  --opd-type megatron \
  --opd-kl-coef 0.0 \
  --opd-teacher-load /path/to/teacher_megatron_ckpt \
  --loss-type custom_loss \
  --custom-loss-function-path slime_self_distill.custom_loss.self_distill_loss \
  --custom-convert-samples-to-train-data-path slime_self_distill.custom_convert.convert_samples_to_train_data \
  --data-source-path slime_self_distill.data_source.SelfDistillRolloutDataSource \
  --self-distill-kl-scope full_vocab \
  --self-distill-teacher-prompt-key teacher_prompt \
  --self-distill-divergence alpha_js \
  --self-distill-alpha 0.0 \
  --self-distill-top-entropy-quantile 1.0 \
  --self-distill-num-loss-tokens-to-skip 0 \
  --prompt-data /path/to/train.jsonl \
  --input-key prompt \
  --metadata-key metadata \
  --save /path/to/save_ckpt \
  --load /path/to/student_megatron_ckpt
