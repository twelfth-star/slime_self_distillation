from __future__ import annotations

import math

import torch
import torch.distributed as dist

from slime.backends.megatron_utils.loss import distributed_log_softmax, get_responses
from slime.utils.ppo_utils import calculate_log_probs_and_entropy, compute_entropy_from_logits


def _safe_crop(*tensors: torch.Tensor):
    min_len = min(t.size(0) for t in tensors)
    return tuple(t[:min_len] for t in tensors)


def _get_entropy_threshold(entropy_list, mask_list, top_entropy_quantile: float):
    if top_entropy_quantile >= 1.0:
        return None

    valid = []
    for entropy, mask in zip(entropy_list, mask_list, strict=False):
        picked = entropy[mask > 0]
        if picked.numel() > 0:
            valid.append(picked)
    device = entropy_list[0].device if entropy_list else torch.device("cpu")
    if valid:
        gathered = torch.cat(valid, dim=0)
    else:
        gathered = torch.empty(0, dtype=torch.float32, device=device)

    # Keep entropy threshold consistent across data-parallel ranks.
    if dist.is_initialized():
        from megatron.core import mpu

        try:
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        except TypeError:
            dp_group = mpu.get_data_parallel_group()

        world_size = dist.get_world_size(group=dp_group)
        if world_size > 1:
            gathered_list = [None] * world_size
            dist.all_gather_object(gathered_list, gathered.detach().float().cpu(), group=dp_group)
            non_empty = [x for x in gathered_list if isinstance(x, torch.Tensor) and x.numel() > 0]
            if non_empty:
                gathered = torch.cat(non_empty, dim=0).to(device=device, dtype=torch.float32)
            else:
                gathered = torch.empty(0, dtype=torch.float32, device=device)

    if gathered.numel() == 0:
        return None

    return torch.quantile(gathered, 1.0 - top_entropy_quantile)


def _distribution_token_divergence(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    args,
    tp_group,
) -> torch.Tensor:
    """
    Arguments:
        student_log_probs: [seq_len, vocab_size] log probabilities from student model
        teacher_log_probs: [seq_len, vocab_size] log probabilities from teacher model
        args: training arguments with self_distill_divergence and self_distill_alpha
        tp_group: tensor parallel group for all-reduce
    """
    mode = getattr(args, "self_distill_divergence", "alpha_js")
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    if mode == "reverse_kl":
        token_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    elif mode == "forward_kl":
        token_div = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    elif mode == "alpha_js":
        alpha = float(getattr(args, "self_distill_alpha", 0.0))
        if alpha <= 0.0:
            token_div = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
        elif alpha >= 1.0:
            token_div = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_log_probs + math.log(1.0 - alpha),
                        teacher_log_probs + math.log(alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            # Match Self-Distillation: alpha * KL(teacher || mixture) + (1-alpha) * KL(student || mixture).
            kl_teacher = (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
            kl_student = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)
            token_div = alpha * kl_teacher + (1.0 - alpha) * kl_student
    else:
        raise ValueError(f"Unsupported self_distill_divergence: {mode}")

    # each TP rank holds a vocab shard; sum token divergence contributions across TP ranks
    if dist.is_initialized():
        dist.all_reduce(token_div, op=dist.ReduceOp.SUM, group=tp_group)
    return token_div


def _sampled_token_divergence(
    student_selected_log_probs: torch.Tensor,
    teacher_selected_log_probs: torch.Tensor,
    args,
) -> torch.Tensor:
    mode = getattr(args, "self_distill_divergence", "alpha_js")
    if mode == "reverse_kl":
        token_div = student_selected_log_probs - teacher_selected_log_probs
    elif mode == "forward_kl":
        token_div = teacher_selected_log_probs - student_selected_log_probs
    elif mode == "alpha_js":
        alpha = float(getattr(args, "self_distill_alpha", 0.0))
        if alpha <= 0.0:
            token_div = teacher_selected_log_probs - student_selected_log_probs
        elif alpha >= 1.0:
            token_div = student_selected_log_probs - teacher_selected_log_probs
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        student_selected_log_probs + math.log(1.0 - alpha),
                        teacher_selected_log_probs + math.log(alpha),
                    ],
                    dim=0,
                ),
                dim=0,
            )
            token_div = alpha * (teacher_selected_log_probs - mixture_log_probs) + (1.0 - alpha) * (
                student_selected_log_probs - mixture_log_probs
            )
    else:
        raise ValueError(f"Unsupported self_distill_divergence: {mode}")
    return token_div


def self_distill_loss(args, batch, logits, sum_of_sample_mean):
    del sum_of_sample_mean

    kl_scope = getattr(args, "self_distill_kl_scope", "off")
    if kl_scope == "off":
        raise ValueError("self_distill_kl_scope=off disables self-distillation; do not use self_distill_loss in this mode.")
    if kl_scope not in ["full_vocab", "sampled_token"]:
        raise ValueError(f"Unsupported self_distill_kl_scope: {kl_scope}")

    from megatron.core import mpu

    tp_group = mpu.get_tensor_model_parallel_group()

    student_selected_log_probs_list = []
    teacher_selected_log_probs_list = []
    student_full_log_probs_list = []
    entropy_list = []
    teacher_full_log_probs_list = []
    teacher_token_log_probs_list = batch.get("teacher_log_probs", None)
    if kl_scope == "full_vocab" and ("teacher_full_log_probs" not in batch or batch["teacher_full_log_probs"] is None):
        raise ValueError("full_vocab self-distillation requires teacher_full_log_probs in batch.")
    if kl_scope == "sampled_token" and teacher_token_log_probs_list is None:
        raise ValueError("sampled_token self-distillation requires teacher_log_probs in batch.")

    responses = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
        )
    )
    if kl_scope == "full_vocab" and len(batch["teacher_full_log_probs"]) != len(responses):
        raise ValueError("teacher_full_log_probs count does not match response count.")
    if kl_scope == "sampled_token" and len(teacher_token_log_probs_list) != len(responses):
        raise ValueError("teacher_log_probs count does not match response count.")

    for idx, (logits_chunk, tokens_chunk) in enumerate(responses):
        student_full_log_probs = distributed_log_softmax(logits_chunk, tp_group)
        student_selected_log_probs, _ = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            tp_group,
            with_entropy=False,
            chunk_size=args.log_probs_chunk_size,
        )
        student_selected_log_probs = student_selected_log_probs.squeeze(-1)
        min_len = student_selected_log_probs.size(0)
        teacher_full_log_probs = None
        if kl_scope == "full_vocab":
            teacher_full_log_probs = batch["teacher_full_log_probs"][idx].to(
                device=student_full_log_probs.device,
                dtype=student_full_log_probs.dtype,
            )
            student_full_log_probs, teacher_full_log_probs, student_selected_log_probs = _safe_crop(
                student_full_log_probs, teacher_full_log_probs, student_selected_log_probs
            )
            min_len = student_full_log_probs.size(0)

        entropy = compute_entropy_from_logits(logits_chunk[:min_len].clone(), tp_group)

        if teacher_token_log_probs_list is not None and idx < len(teacher_token_log_probs_list):
            teacher_selected_log_probs = teacher_token_log_probs_list[idx]
            if teacher_selected_log_probs is not None:
                teacher_selected_log_probs = teacher_selected_log_probs.to(
                    device=student_selected_log_probs.device,
                    dtype=student_selected_log_probs.dtype,
                )[:min_len]
            else:
                if kl_scope == "sampled_token":
                    raise ValueError("sampled_token self-distillation requires non-empty teacher_log_probs per sample.")
                teacher_selected_log_probs = torch.zeros_like(student_selected_log_probs)
        else:
            if kl_scope == "sampled_token":
                raise ValueError("sampled_token self-distillation requires teacher_log_probs for each sample.")
            teacher_selected_log_probs = torch.zeros_like(student_selected_log_probs)

        student_full_log_probs_list.append(student_full_log_probs)
        if teacher_full_log_probs is None:
            teacher_full_log_probs_list.append(student_full_log_probs.new_zeros((student_full_log_probs.size(0), 0)))
        else:
            teacher_full_log_probs_list.append(teacher_full_log_probs)
        student_selected_log_probs_list.append(student_selected_log_probs)
        teacher_selected_log_probs_list.append(teacher_selected_log_probs)
        entropy_list.append(entropy)

    rollout_log_probs_list = batch.get("rollout_log_probs", None)

    skip_tokens = int(getattr(args, "self_distill_num_loss_tokens_to_skip", 0))
    top_entropy_quantile = float(getattr(args, "self_distill_top_entropy_quantile", 1.0))
    use_importance_correction = bool(getattr(args, "self_distill_importance_correction", True))
    importance_cap = float(getattr(args, "self_distill_importance_cap", 2.0))

    mask_list = []
    for loss_mask in batch["loss_masks"]:
        mask = loss_mask.to(device=student_selected_log_probs_list[0].device, dtype=torch.float32).clone()
        if skip_tokens > 0:
            mask[: min(skip_tokens, mask.numel())] = 0.0
        mask_list.append(mask)

    entropy_threshold = _get_entropy_threshold(entropy_list, mask_list, top_entropy_quantile)

    sample_losses = []
    sample_entropies = []
    sample_logprob_gaps = []
    for idx, (
        student_selected_log_probs,
        teacher_selected_log_probs,
        student_full_log_probs,
        teacher_full_log_probs,
        entropy,
        mask,
    ) in enumerate(
        zip(
            student_selected_log_probs_list,
            teacher_selected_log_probs_list,
            student_full_log_probs_list,
            teacher_full_log_probs_list,
            entropy_list,
            mask_list,
            strict=False,
        )
    ):
        has_rollout_log_probs = (
            rollout_log_probs_list is not None
            and idx < len(rollout_log_probs_list)
            and rollout_log_probs_list[idx] is not None
        )
        if has_rollout_log_probs:
            rollout_log_probs = rollout_log_probs_list[idx].to(
                device=student_selected_log_probs.device,
                dtype=student_selected_log_probs.dtype,
            )
        else:
            rollout_log_probs = None

        (
            student_selected_log_probs,
            teacher_selected_log_probs,
            entropy,
            mask,
        ) = _safe_crop(
            student_selected_log_probs,
            teacher_selected_log_probs,
            entropy,
            mask,
        )
        if kl_scope == "full_vocab":
            student_full_log_probs = student_full_log_probs[: student_selected_log_probs.numel()]
            teacher_full_log_probs = teacher_full_log_probs[: student_selected_log_probs.numel()]
        if rollout_log_probs is not None:
            rollout_log_probs = rollout_log_probs[: student_selected_log_probs.numel()]

        if entropy_threshold is not None:
            mask = mask * (entropy >= entropy_threshold).to(mask.dtype)

        if kl_scope == "full_vocab":
            token_loss = _distribution_token_divergence(student_full_log_probs, teacher_full_log_probs, args, tp_group)
        else:
            token_loss = _sampled_token_divergence(student_selected_log_probs, teacher_selected_log_probs, args)
        if use_importance_correction and rollout_log_probs is not None:
            ratio = torch.exp(student_selected_log_probs - rollout_log_probs)
            ratio = torch.clamp(ratio, max=importance_cap)
            importance_weight = (ratio * mask).sum() / mask.sum().clamp(min=1.0)
            token_loss = token_loss * importance_weight

        denom = torch.clamp_min(mask.sum(), 1.0)
        sample_losses.append((token_loss * mask).sum() / denom)
        sample_entropies.append((entropy * mask).sum() / denom)
        sample_logprob_gaps.append(((student_selected_log_probs - teacher_selected_log_probs) * mask).sum() / denom)

    if not sample_losses:
        zero = logits.sum() * 0.0
        return zero, {"distill_loss": zero.detach()}

    loss = torch.stack(sample_losses).mean()
    metrics = {
        "distill_loss": loss.detach(),
        "distill_entropy": torch.stack(sample_entropies).mean().detach(),
        "distill_logprob_gap": torch.stack(sample_logprob_gaps).mean().detach(),
    }
    return loss, metrics
