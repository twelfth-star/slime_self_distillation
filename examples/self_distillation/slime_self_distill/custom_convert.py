from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from slime.utils.types import Sample

_TOKENIZER = None


def _flatten_samples(samples: list[Sample] | list[list[Sample]]) -> list[Sample]:
    if not samples:
        return []
    if isinstance(samples[0], list):
        flat_samples = []
        for group in samples:
            flat_samples.extend(group)
        return flat_samples
    return samples


def _get_tokenizer(args):
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return _TOKENIZER


def _prompt_to_ids(prompt: Any, tokenizer, tools: Any = None) -> list[int]:
    if isinstance(prompt, str):
        prompt_text = prompt
    elif isinstance(prompt, list):
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Prompt is chat messages but tokenizer does not support apply_chat_template.")
        prompt_text = tokenizer.apply_chat_template(
            prompt,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    return tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def _truncate_prompt_ids(prompt_ids: list[int], args) -> list[int]:
    max_prompt_len = getattr(args, "self_distill_max_prompt_length", None)
    if max_prompt_len is None:
        max_prompt_len = getattr(args, "rollout_max_prompt_len", None)
    if max_prompt_len is None:
        return prompt_ids
    max_prompt_len = int(max_prompt_len)
    if max_prompt_len <= 0:
        return prompt_ids
    return prompt_ids[-max_prompt_len:]


def convert_samples_to_train_data(args, samples: list[Sample] | list[list[Sample]]) -> dict[str, Any]:
    samples = _flatten_samples(samples)
    if not samples:
        return {}

    tokenizer = _get_tokenizer(args)

    student_tokens_list: list[list[int]] = []
    teacher_tokens_list: list[list[int]] = []
    response_lengths: list[int] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    truncated: list[int] = []
    sample_indices: list[int] = []
    loss_masks: list[list[int]] = []
    prompts: list[Any] = []

    for sample in samples:
        metadata = sample.metadata or {}
        tools = metadata.get("tools", None)

        student_prompt = metadata.get("_student_prompt", sample.prompt)
        teacher_prompt = metadata.get("_teacher_prompt", student_prompt)

        response_length = int(sample.response_length)
        if response_length < 0 or response_length > len(sample.tokens):
            raise ValueError(
                f"Invalid response_length={response_length} for sample {sample.index} with total tokens={len(sample.tokens)}"
            )
        completion_tokens = sample.tokens[-response_length:] if response_length > 0 else []

        student_prompt_ids = _truncate_prompt_ids(_prompt_to_ids(student_prompt, tokenizer, tools=tools), args)
        student_tokens = student_prompt_ids + completion_tokens

        teacher_prompt_ids = _truncate_prompt_ids(_prompt_to_ids(teacher_prompt, tokenizer, tools=tools), args)
        teacher_tokens = teacher_prompt_ids + completion_tokens

        student_tokens_list.append(student_tokens)
        teacher_tokens_list.append(teacher_tokens)
        response_lengths.append(response_length)
        rewards.append(0.0)
        raw_rewards.append(0.0)
        truncated.append(1 if sample.status == Sample.Status.TRUNCATED else 0)
        sample_indices.append(sample.index)
        prompts.append(student_prompt)

        if sample.loss_mask is None:
            sample_loss_mask = [1] * response_length
        else:
            sample_loss_mask = list(sample.loss_mask)
        if len(sample_loss_mask) != response_length:
            raise ValueError(
                f"loss mask length {len(sample_loss_mask)} != response length {response_length} for sample {sample.index}"
            )
        if sample.remove_sample:
            sample_loss_mask = [0] * response_length
        loss_masks.append(sample_loss_mask)

    train_data: dict[str, Any] = {
        "tokens": student_tokens_list,
        "teacher_tokens": teacher_tokens_list,
        "teacher_total_lengths": [len(x) for x in teacher_tokens_list],
        "response_lengths": response_lengths,
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": truncated,
        "sample_indices": sample_indices,
        "loss_masks": loss_masks,
        "prompt": prompts,
    }

    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]
    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]
    if samples[0].multimodal_train_inputs is not None:
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]
    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    return train_data
