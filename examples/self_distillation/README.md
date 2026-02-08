# Self-Distillation Example

This example adds a self-distillation workflow on top of `slime`, with minimal core changes.

## What it provides

- A rollout data source that keeps student prompt for rollout and injects teacher prompt metadata for training-time distillation.
- A custom sample-to-train-data converter that builds both:
  - `tokens` (student prompt + completion)
  - `teacher_tokens` (teacher prompt + completion)
- A custom distillation loss for `slime` `custom_loss` mode.
- A dataset preparation utility and a runnable command template.

## Scope

The implementation is designed to preserve existing `slime` training/serving optimizations (Megatron + SGLang + colocate path) while adding self-distillation behavior through an example package (idanshen/Self-Distillation from paper "Self-Distillation Enables Continual Learning").

The custom loss supports two divergence scopes on completion tokens:
- `full_vocab`: full-distribution KL/alpha-JS (strict mode).
- `sampled_token`: sampled-token KL/alpha-JS using teacher token log-probs.
Both support optional entropy filtering and rollout/train mismatch correction.

## Dataset format

The data source expects each sample to include teacher prompt in metadata:

```json
{
  "prompt": "...",
  "metadata": {
    "teacher_prompt": "..."
  }
}
```

If your source data has `teacher_prompt` as a top-level field, use `slime_self_distill.prepare_dataset` to convert it.

## Usage

1. Put this directory on `PYTHONPATH` together with `slime`.
2. Use:
   - `--data-source-path slime_self_distill.data_source.SelfDistillRolloutDataSource`
   - `--custom-convert-samples-to-train-data-path slime_self_distill.custom_convert.convert_samples_to_train_data`
   - `--loss-type custom_loss`
   - `--custom-loss-function-path slime_self_distill.custom_loss.self_distill_loss`
3. Enable teacher model loading through existing OPD Megatron path:
   - `--use-opd --opd-type megatron --opd-teacher-load ...`
   - Set `--opd-kl-coef 0.0` if you only want custom self-distillation loss and no OPD advantage penalty.
4. For strict DistilTrainer-style defaults, use:
   - `--self-distill-kl-scope full_vocab`
   - `--self-distill-divergence alpha_js --self-distill-alpha 0.0` (forward-KL default behavior)
5. For sampled-token divergence mode, use:
   - `--self-distill-kl-scope sampled_token`
6. See `run_self_distill_slime.sh`.
