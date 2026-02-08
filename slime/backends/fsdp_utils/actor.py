import logging
import math
import os
import random
from argparse import Namespace
from itertools import accumulate

import ray
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig

from slime.ray.train_actor import TrainRayActor
from slime.utils import logging_utils, train_dump_utils, train_metric_utils
from slime.utils.data import get_minimum_num_micro_batch_size, process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.metric_utils import compute_rollout_step
from slime.utils.misc import Box
from slime.utils.ppo_utils import compute_approx_kl, compute_gspo_kl, compute_opsm_mask, compute_policy_loss
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.profile_utils import TrainProfiler
from slime.utils.timer import Timer, inverse_timer, timer, with_defer

from . import checkpoint
from .data_packing import pack_sequences, unpack_sequences
from .lr_scheduler import get_lr_scheduler
from .update_weight_utils import UpdateWeightFromDistributed, UpdateWeightFromTensor

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TrainRayActor):
    """Simplified TrainRayActor for pure HF+FSDP training.

    Responsibilities:
      * Initialize model/tokenizer on rank0 sequentially to avoid race on cache
      * Wrap model with FSDP
      * Provide minimal train / save / update_weights hooks compatible with existing RayTrainGroup

    Weight update strategy:
      * Rank0 gathers state_dict (full) and broadcasts tensor-by-tensor.
      * For small models this is fine; for larger models consider sharded state_dict type.
    """

    @with_defer(lambda: Timer().start("train_wait"))
    def init(self, args: Namespace, role: str, with_ref: bool = False, with_opd_teacher: bool = False) -> int:  # type: ignore[override]
        if with_opd_teacher:
            raise NotImplementedError(
                "On-policy distillation (OPD) with Megatron teacher is not supported in FSDP backend. "
                "Please use the Megatron backend for OPD, or use --opd-type=sglang with an external teacher server."
            )
        super().init(args, role, with_ref, with_opd_teacher)

        # Setup device mesh for data parallelism
        self._setup_device_mesh()
        torch.manual_seed(args.seed)

        self.train_parallel_config = {
            "dp_size": self.dp_size,
        }

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        self._enable_true_on_policy_optimizations(args)
        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        for i in range(dist.get_world_size()):
            if i == dist.get_rank():
                self.hf_config = AutoConfig.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = load_tokenizer(self.args.hf_checkpoint, trust_remote_code=True)
                # Vision models have `vision_config` in the config
                if hasattr(self.hf_config, "vision_config"):
                    self.processor = load_processor(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        init_context = self._get_init_weight_context_manager()

        with init_context():
            model = self.get_model_cls().from_pretrained(
                self.args.hf_checkpoint,
                trust_remote_code=True,
                attn_implementation=self.args.attn_implementation,
            )

        model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(model, mesh=self.dp_mesh, cpu_offload=self.fsdp_cpu_offload, args=self.args)

        model = self._fsdp2_load_full_state_dict(
            model, full_state, self.dp_mesh, cpu_offload=True if self.fsdp_cpu_offload else None
        )

        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)

        self.weight_updater = (
            UpdateWeightFromTensor(self.args, self.model)
            if self.args.colocate
            else UpdateWeightFromDistributed(self.args, self.model)
        )

        checkpoint.finalize_load(self, checkpoint_payload)

        # Initialize data packing parameters
        self.max_tokens_per_gpu = args.max_tokens_per_gpu  # From main arguments

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def get_model_cls(self):
        # Vision models have `vision_config` in the config
        if hasattr(self.hf_config, "vision_config"):
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM

            return AutoModelForCausalLM

    def _enable_true_on_policy_optimizations(self, args):
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            from .models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

            apply_true_on_policy_patch_for_qwen3_moe()
        else:
            from .models.qwen3_moe_hf import apply_fsdp_moe_patch

            apply_fsdp_moe_patch()

    def _setup_device_mesh(self) -> None:
        """Setup device mesh for data parallelism."""
        from torch.distributed.device_mesh import init_device_mesh

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Pure data parallelism
        self.dp_size = world_size
        self.dp_rank = rank

        # Create 1D device mesh for data parallelism
        self.mesh = init_device_mesh("cuda", mesh_shape=(self.dp_size,), mesh_dim_names=("dp",))
        self.dp_group = self.mesh.get_group("dp")
        self.dp_mesh = self.mesh

        logger.info(f"[Rank {rank}] Device mesh (1D): world_size={world_size}, dp_size={self.dp_size}")

    def _get_init_weight_context_manager(self):
        """Get context manager for model initialization.

        Returns a callable that creates a context manager.
        Uses meta device (no memory allocation) for non-rank-0 processes,
        UNLESS tie_word_embeddings=True (which causes hangs with meta tensors).

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        NOTE: tie_word_embedding causes meta_tensor init to hang
        """
        from accelerate import init_empty_weights

        # Check if model uses tied word embeddings (which doesn't work with meta tensors)
        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        """Load full state dict into FSDP2 model with efficient broadcast from rank 0.

        This function loads weights from rank 0 and broadcasts to all other ranks,
        avoiding the need for each rank to load the full model from disk.

        Args:
            model: FSDP2-wrapped model
            full_state: State dict (only rank 0 has real weights, others have empty dict)
            device_mesh: Device mesh for FSDP
            cpu_offload: If not None, enables StateDictOptions cpu_offload

        Ref:verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    @timer
    def sleep(self) -> None:
        """Pause CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        print_memory("before offload model")

        self.model.cpu()
        move_torch_optimizer(self.optimizer, "cpu")
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        """Resume CUDA memory for all tracked tensors."""
        if not self.args.offload_train:
            return

        self.model.cuda()
        move_torch_optimizer(self.optimizer, "cuda")
        dist.barrier(group=get_gloo_group())
        print_memory("after wake_up model")

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        """Delegate checkpoint saving to the shared checkpoint utilities."""
        if self.args.debug_rollout_only or self.args.save is None:
            return

        assert not self.args.async_save, "FSDPTrainRayActor does not support async_save yet."
        checkpoint.save(self, rollout_id)

    def _compute_log_prob(
        self,
        model_tag: str,
        packed_batches: list[dict[str, torch.Tensor]],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        """Compute token log-probabilities for a list of packed batches.

        Parameters:
            model_tag: Which parameters to use, e.g. "actor" or "ref".
            packed_batches: A list of packed batch dictionaries produced by
                `pack_sequences`, each containing at least `tokens` and
                `position_ids`; may also include multimodal keys like `pixel_values`.
            store_prefix: Prefix to use for keys in outputs (e.g., "ref_").

        Returns:
            A lightweight dictionary keyed by f"{store_prefix}log_probs". The
            actual per-sequence results are written in-place into each element of
            `packed_batches` under the same key and can be read back by callers.

        Note:
            Uses separate ref model when model_tag == "ref". The ref model is
            loaded from CPU to GPU on-demand and offloaded back after use.
        """
        # Select which model to use
        if model_tag == "ref" and self.ref_model is not None:
            if not self.fsdp_cpu_offload:
                self.model.cpu()
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

            active_model = self.ref_model
            active_model.eval()
        else:
            active_model = self.model

        try:
            rollout_data = {f"{store_prefix}log_probs": []}
            with timer(f"{store_prefix}log_probs"), torch.no_grad():
                for batch in self.prof.iterate_train_log_probs(
                    tqdm(packed_batches, desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                ):
                    model_args = self._get_model_inputs_args(batch)
                    logits = active_model(**model_args).logits.squeeze(0).float()
                    log_probs_result, entropy_result = get_logprob_and_entropy(
                        logits=logits,
                        target_tokens=batch["tokens"],
                        allow_compile=not self.args.true_on_policy_mode,
                        temperature=self.args.rollout_temperature,
                    )
                    batch[f"{store_prefix}log_probs"] = log_probs_result
                    if store_prefix == "":
                        batch["entropy"] = entropy_result
            return rollout_data

        finally:
            # Restore actor model if it was offloaded
            if model_tag == "ref" and self.ref_model is not None:
                torch.cuda.empty_cache()
                dist.barrier(group=get_gloo_group())

                if not self.fsdp_cpu_offload:
                    self.model.cuda()
                    dist.barrier(group=get_gloo_group())

    def _packed_data(
        self, rollout_data: dict[str, list[torch.Tensor]]
    ) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
        """Pack variable-length sequences for efficient processing.

        Parameters:
            rollout_data: Dictionary of lists containing sequence-level tensors
                such as `tokens`, `loss_masks`, `rewards`, `response_lengths`,
                `advantages`, `returns`, and optional `rollout_log_probs`.

        Returns:
            A pair `(packed_batches, grad_accum)` where `packed_batches` is a list
            of packed batch dictionaries and `grad_accum` lists the micro-batch
            indices at which to perform optimizer steps.
        """
        # Pack sequences efficiently
        tokens = rollout_data["tokens"]

        packed_batches = []
        mbs_size_list = []
        local_batch_size = self.args.global_batch_size // self.dp_size
        assert (
            self.args.global_batch_size % self.dp_size == 0
        ), f"global_batch_size {self.args.global_batch_size} is not divisible by dp_world_size {self.dp_size}"
        # Use global_batch_size for splitting when max_tokens_per_gpu is enabled
        if self.args.use_dynamic_batch_size:
            max_tokens = self.args.max_tokens_per_gpu

            for i in range(0, len(tokens), local_batch_size):
                mbs_size_list.append(
                    get_minimum_num_micro_batch_size(
                        [len(t) for t in rollout_data["tokens"][i : i + local_batch_size]],
                        max_tokens,
                    )
                )
            num_microbatches = torch.tensor(mbs_size_list, dtype=torch.int, device=torch.cuda.current_device())
            dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=self.dp_group)
            num_microbatches = num_microbatches.tolist()
        else:
            num_microbatches = [self.args.global_batch_size // (self.args.micro_batch_size * self.dp_size)] * (
                len(tokens) // local_batch_size
            )

        start = 0
        for mbs_size in num_microbatches:
            end = start + local_batch_size
            packed_batches.extend(
                pack_sequences(
                    rollout_data["tokens"][start:end],
                    rollout_data["loss_masks"][start:end],
                    rollout_data["rewards"][start:end],
                    rollout_data["raw_reward"][start:end],
                    rollout_data["response_lengths"][start:end],
                    rollout_data["advantages"][start:end],
                    rollout_data["returns"][start:end],
                    teacher_tokens=(rollout_data["teacher_tokens"][start:end] if "teacher_tokens" in rollout_data else None),
                    teacher_log_probs=(
                        rollout_data["teacher_log_probs"][start:end] if "teacher_log_probs" in rollout_data else None
                    ),
                    rollout_log_probs=(
                        rollout_data["rollout_log_probs"][start:end] if "rollout_log_probs" in rollout_data else None
                    ),
                    multimodal_train_inputs=(
                        rollout_data["multimodal_train_inputs"][start:end]
                        if "multimodal_train_inputs" in rollout_data
                        else None
                    ),
                    num_packs=mbs_size,
                )
            )
            start = end
        grad_accum = list(accumulate(num_microbatches))

        return packed_batches, grad_accum

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        """Run one training update over a rollout batch.

        Parameters:
            rollout_id: Monotonic id for logging.
            rollout_data_ref: A Box handle wrapping a Ray object reference to a
                dictionary with rollout tensors and metadata (e.g., `tokens`,
                `loss_masks`, `rewards`, `response_lengths`, optional
                `rollout_log_probs`, etc.). It will be fetched and partitioned
                by `process_rollout_data` based on data-parallel rank/size.
        """
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data = process_rollout_data(self.args, rollout_data_ref, self.dp_rank, self.dp_size)
            if self.args.debug_rollout_only:
                return
            self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        train_metric_utils.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=None,
        )

    def _log_rollout_data(self, rollout_id: int, rollout_data, packed_batches):
        log_dict = {}
        if "raw_reward" in rollout_data and dist.get_rank() == 0:
            raw_reward_list = rollout_data["raw_reward"]
            if raw_reward_list:
                log_dict["rollout/raw_reward"] = sum(raw_reward_list) / len(raw_reward_list)

        for metric_key in ["log_probs", "rollout_log_probs", "ref_log_probs", "advantages", "returns"]:
            if metric_key not in packed_batches[0]:
                continue
            val = torch.tensor([0.0], device=torch.cuda.current_device())
            for _mbs_id, batches in enumerate(packed_batches):
                unpacked_batches = unpack_sequences(batches)
                for unpacked_batch in unpacked_batches:
                    if isinstance(unpacked_batch[metric_key], torch.Tensor):
                        loss_masks_tensor = unpacked_batch["loss_masks"].to(device=torch.cuda.current_device())
                        metric_tensor = unpacked_batch[metric_key].to(device=torch.cuda.current_device())
                        val += (metric_tensor * loss_masks_tensor).sum() / loss_masks_tensor.sum().clamp_min(1)
                    else:
                        val += unpacked_batch[metric_key]
            dist.all_reduce(val, op=dist.ReduceOp.SUM, group=self.dp_group)
            log_dict[f"rollout/{metric_key}"] = (
                val / (self.args.n_samples_per_prompt * self.args.rollout_batch_size)
            ).item()
        if dist.get_rank() == 0:
            logger.info(f"rollout {rollout_id}: {log_dict}")
            log_dict["rollout/step"] = compute_rollout_step(self.args, rollout_id)
            logging_utils.log(self.args, log_dict, step_key="rollout/step")

        if self.args.ci_test and self.args.true_on_policy_mode:
            assert log_dict["rollout/log_probs"] == log_dict["rollout/rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['rollout/log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout/rollout_log_probs']})"
            )

    def _train_core(self, rollout_id: int, rollout_data) -> None:
        if self.args.advantage_estimator in ["grpo", "gspo"]:
            rollout_data["advantages"] = rollout_data["returns"] = [
                torch.tensor([rollout_data["rewards"][i]] * rollout_data["response_lengths"][i])
                for i in range(len(rollout_data["rewards"]))
            ]
        else:
            raise NotImplementedError(f"Unsupported advantage_estimator {self.args.advantage_estimator}")

        packed_batches, grad_accum = self._packed_data(rollout_data)

        assert (
            len(grad_accum) > 0
        ), f"Invalid grad_accum {grad_accum} for micro_batch_size {self.args.micro_batch_size} and global_batch_size {self.args.global_batch_size}"

        if self.ref_model is not None:
            self._compute_log_prob("ref", packed_batches, store_prefix="ref_")

        self._compute_log_prob("actor", packed_batches)
        self._log_rollout_data(rollout_id, rollout_data, packed_batches)

        with timer("actor_train"):
            reported_accum: dict[str, list[torch.Tensor]] = {}
            self.optimizer.zero_grad(set_to_none=True)
            for mbs_id, packed_batch in self.prof.iterate_train_actor(
                enumerate(tqdm(packed_batches, desc="actor_train", disable=dist.get_rank() != 0))
            ):
                self._train_step(
                    packed_batch=packed_batch,
                    reported_accum=reported_accum,
                    mbs_id=mbs_id,
                    grad_accum=grad_accum,
                )

        self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Update ref model if needed (copy actor weights to ref)
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            # Copy actor model state to ref model
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _extract_response_distributions(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        response_lengths: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        full_log_probs_list = []
        selected_log_probs_list = []
        entropy_list = []
        temperature = self.args.rollout_temperature

        for idx, response_length in enumerate(response_lengths):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            seq_len = end - start
            if response_length <= 0 or seq_len <= 1:
                empty = logits.new_zeros((0,), dtype=torch.float32)
                full_log_probs_list.append(logits.new_zeros((0, logits.size(-1)), dtype=torch.float32))
                selected_log_probs_list.append(empty)
                entropy_list.append(empty)
                continue

            # Predict token[t] from logits[t-1], and keep only response-token positions.
            response_start = seq_len - response_length
            logit_start = start + response_start - 1
            logit_end = end - 1
            logits_slice = logits[logit_start:logit_end]
            if temperature is not None:
                logits_slice = logits_slice / temperature
            full_log_probs = torch.log_softmax(logits_slice, dim=-1)
            target_tokens = tokens[start + response_start : end].to(device=logits.device)
            selected_log_probs = full_log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            probs = full_log_probs.exp()
            entropy = -(probs * full_log_probs).sum(dim=-1)

            full_log_probs_list.append(full_log_probs)
            selected_log_probs_list.append(selected_log_probs)
            entropy_list.append(entropy)

        return full_log_probs_list, selected_log_probs_list, entropy_list

    def _self_distill_loss(self, packed_batch, logits, unpacked_batches):
        kl_scope = getattr(self.args, "self_distill_kl_scope", "off")
        use_teacher_log_probs = isinstance(packed_batch.get("teacher_log_probs", None), torch.Tensor)

        if kl_scope not in ["full_vocab", "sampled_token"]:
            raise ValueError(f"Unsupported self_distill_kl_scope: {kl_scope}")
        if kl_scope == "full_vocab" and self.ref_model is None:
            raise ValueError("FSDP full_vocab self-distillation requires a ref model loaded via --ref-load.")
        if kl_scope == "sampled_token" and not use_teacher_log_probs and self.ref_model is None:
            raise ValueError(
                "FSDP sampled_token self-distillation requires teacher_log_probs in rollout data "
                "or a ref model via --ref-load."
            )

        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]
        student_full_log_probs_list, student_selected_log_probs_list, entropy_list = self._extract_response_distributions(
            logits=logits,
            tokens=packed_batch["tokens"],
            cu_seqlens=packed_batch["cu_seqlens"],
            response_lengths=response_lengths,
        )

        teacher_full_log_probs_list: list[torch.Tensor] = []
        teacher_selected_log_probs_list: list[torch.Tensor] = []
        need_teacher_forward = kl_scope == "full_vocab" or (kl_scope == "sampled_token" and not use_teacher_log_probs)
        if need_teacher_forward:
            if "teacher_tokens" not in packed_batch or "teacher_cu_seqlens" not in packed_batch:
                raise ValueError(
                    "Self-distillation requires teacher_tokens and teacher_cu_seqlens in packed batch "
                    "when teacher forward is needed."
                )
            with torch.no_grad():
                self.ref_model.eval()
                teacher_model_args = self._get_model_inputs_args(
                    packed_batch, token_key="teacher_tokens", position_ids_key="teacher_position_ids"
                )
                teacher_logits = self.ref_model(**teacher_model_args).logits.squeeze(0).float()

            teacher_full_log_probs_list, teacher_selected_log_probs_list, _ = self._extract_response_distributions(
                logits=teacher_logits,
                tokens=packed_batch["teacher_tokens"],
                cu_seqlens=packed_batch["teacher_cu_seqlens"],
                response_lengths=response_lengths,
            )
        else:
            teacher_log_probs = packed_batch["teacher_log_probs"]
            offset = 0
            for response_length in response_lengths:
                end = offset + int(response_length)
                if end > teacher_log_probs.numel():
                    raise ValueError(
                        "teacher_log_probs length is smaller than the total response length in packed batch."
                    )
                teacher_selected_log_probs_list.append(
                    teacher_log_probs[offset:end].to(
                        device=logits.device,
                        dtype=student_selected_log_probs_list[0].dtype if student_selected_log_probs_list else torch.float32,
                    )
                )
                teacher_full_log_probs_list.append(logits.new_zeros((int(response_length), 0), dtype=torch.float32))
                offset = end
            if offset != teacher_log_probs.numel():
                raise ValueError(
                    "teacher_log_probs length does not match total response length in packed batch."
                )

        skip_tokens = int(getattr(self.args, "self_distill_num_loss_tokens_to_skip", 0))
        top_entropy_quantile = float(getattr(self.args, "self_distill_top_entropy_quantile", 1.0))
        use_importance_correction = bool(getattr(self.args, "self_distill_importance_correction", True))
        importance_cap = float(getattr(self.args, "self_distill_importance_cap", 2.0))
        divergence_mode = getattr(self.args, "self_distill_divergence", "alpha_js")
        alpha = float(getattr(self.args, "self_distill_alpha", 0.0))

        loss_masks = []
        for batch in unpacked_batches:
            mask = batch["loss_masks"].to(device=logits.device, dtype=torch.float32).clone()
            if skip_tokens > 0:
                mask[: min(skip_tokens, mask.numel())] = 0.0
            loss_masks.append(mask)

        entropy_threshold = None
        if top_entropy_quantile < 1.0:
            valid_entropy = []
            for entropy, mask in zip(entropy_list, loss_masks, strict=False):
                picked = entropy[mask > 0]
                if picked.numel() > 0:
                    valid_entropy.append(picked)
            gathered = (
                torch.cat(valid_entropy, dim=0)
                if valid_entropy
                else torch.empty(0, device=logits.device, dtype=torch.float32)
            )
            if self.dp_size > 1:
                gathered_list = [None] * self.dp_size
                dist.all_gather_object(gathered_list, gathered.detach().float().cpu(), group=self.dp_group)
                non_empty = [x for x in gathered_list if isinstance(x, torch.Tensor) and x.numel() > 0]
                if non_empty:
                    gathered = torch.cat(non_empty, dim=0).to(device=logits.device, dtype=torch.float32)
                else:
                    gathered = torch.empty(0, device=logits.device, dtype=torch.float32)
            if gathered.numel() > 0:
                entropy_threshold = torch.quantile(gathered, 1.0 - top_entropy_quantile)

        sample_losses = []
        sample_entropies = []
        sample_logprob_gaps = []
        for idx, (
            student_full_log_probs,
            teacher_full_log_probs,
            student_selected_log_probs,
            teacher_selected_log_probs,
            entropy,
            mask,
        ) in enumerate(
            zip(
                student_full_log_probs_list,
                teacher_full_log_probs_list,
                student_selected_log_probs_list,
                teacher_selected_log_probs_list,
                entropy_list,
                loss_masks,
                strict=False,
            )
        ):
            min_len = min(
                student_full_log_probs.size(0),
                teacher_full_log_probs.size(0),
                student_selected_log_probs.size(0),
                teacher_selected_log_probs.size(0),
                entropy.size(0),
                mask.size(0),
            )
            student_full_log_probs = student_full_log_probs[:min_len]
            teacher_full_log_probs = teacher_full_log_probs[:min_len]
            student_selected_log_probs = student_selected_log_probs[:min_len]
            teacher_selected_log_probs = teacher_selected_log_probs[:min_len]
            entropy = entropy[:min_len]
            mask = mask[:min_len]

            if entropy_threshold is not None:
                mask = mask * (entropy >= entropy_threshold).to(mask.dtype)

            if kl_scope == "full_vocab":
                if divergence_mode == "reverse_kl":
                    student_probs = student_full_log_probs.exp()
                    token_loss = (student_probs * (student_full_log_probs - teacher_full_log_probs)).sum(dim=-1)
                elif divergence_mode == "forward_kl":
                    teacher_probs = teacher_full_log_probs.exp()
                    token_loss = (teacher_probs * (teacher_full_log_probs - student_full_log_probs)).sum(dim=-1)
                elif divergence_mode == "alpha_js":
                    if alpha <= 0.0:
                        teacher_probs = teacher_full_log_probs.exp()
                        token_loss = (teacher_probs * (teacher_full_log_probs - student_full_log_probs)).sum(dim=-1)
                    elif alpha >= 1.0:
                        student_probs = student_full_log_probs.exp()
                        token_loss = (student_probs * (student_full_log_probs - teacher_full_log_probs)).sum(dim=-1)
                    else:
                        mixture_log_probs = torch.logsumexp(
                            torch.stack(
                                [
                                    student_full_log_probs + math.log(1.0 - alpha),
                                    teacher_full_log_probs + math.log(alpha),
                                ],
                                dim=0,
                            ),
                            dim=0,
                        )
                        teacher_probs = teacher_full_log_probs.exp()
                        student_probs = student_full_log_probs.exp()
                        kl_teacher = (teacher_probs * (teacher_full_log_probs - mixture_log_probs)).sum(dim=-1)
                        kl_student = (student_probs * (student_full_log_probs - mixture_log_probs)).sum(dim=-1)
                        token_loss = alpha * kl_teacher + (1.0 - alpha) * kl_student
                else:
                    raise ValueError(f"Unsupported self_distill_divergence: {divergence_mode}")
            elif kl_scope == "sampled_token":
                if divergence_mode == "reverse_kl":
                    token_loss = student_selected_log_probs - teacher_selected_log_probs
                elif divergence_mode == "forward_kl":
                    token_loss = teacher_selected_log_probs - student_selected_log_probs
                elif divergence_mode == "alpha_js":
                    if alpha <= 0.0:
                        token_loss = teacher_selected_log_probs - student_selected_log_probs
                    elif alpha >= 1.0:
                        token_loss = student_selected_log_probs - teacher_selected_log_probs
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
                        token_loss = alpha * (teacher_selected_log_probs - mixture_log_probs) + (1.0 - alpha) * (
                            student_selected_log_probs - mixture_log_probs
                        )
                else:
                    raise ValueError(f"Unsupported self_distill_divergence: {divergence_mode}")
            else:
                raise ValueError(f"Unsupported self_distill_kl_scope: {kl_scope}")

            if use_importance_correction:
                rollout_log_probs = unpacked_batches[idx].get("rollout_log_probs", None)
                if isinstance(rollout_log_probs, torch.Tensor) and rollout_log_probs.numel() > 0:
                    rollout_log_probs = rollout_log_probs.to(
                        device=student_selected_log_probs.device,
                        dtype=student_selected_log_probs.dtype,
                    )[:min_len]
                    ratio = torch.exp(student_selected_log_probs - rollout_log_probs)
                    ratio = torch.clamp(ratio, max=importance_cap)
                    importance_weight = (ratio * mask).sum() / mask.sum().clamp(min=1.0)
                    token_loss = token_loss * importance_weight

            denom = mask.sum().clamp(min=1.0)
            sample_losses.append((token_loss * mask).sum() / denom)
            sample_entropies.append((entropy * mask).sum() / denom)
            sample_logprob_gaps.append(((student_selected_log_probs - teacher_selected_log_probs) * mask).sum() / denom)

        if not sample_losses:
            zero = logits.sum() * 0.0
            return zero, {"distill_loss": zero.detach()}

        loss = torch.stack(sample_losses).mean()
        reported = {
            "distill_loss": loss.detach(),
            "distill_entropy": torch.stack(sample_entropies).mean().detach(),
            "distill_logprob_gap": torch.stack(sample_logprob_gaps).mean().detach(),
        }
        return loss, reported

    def _train_step(self, packed_batch, reported_accum, mbs_id, grad_accum):
        # Prepare model inputs
        model_args = self._get_model_inputs_args(packed_batch)
        logits = self.model(**model_args).logits.squeeze(0).float()

        # Compute log probs and entropy
        log_probs, entropy_result = get_logprob_and_entropy(
            logits=logits,
            target_tokens=packed_batch["tokens"],
            allow_compile=not self.args.true_on_policy_mode,
            temperature=self.args.rollout_temperature,
        )
        packed_batch["cur_log_probs"] = log_probs
        packed_batch["entropy"] = entropy_result

        unpacked_batches = unpack_sequences(packed_batch)

        if self.args.loss_type == "custom_loss" and getattr(self.args, "self_distill_kl_scope", "off") != "off":
            loss, reported = self._self_distill_loss(packed_batch, logits, unpacked_batches)
            loss = loss * self.dp_size / self.args.global_batch_size
            loss.backward()

            for k, v in reported.items():
                reported_accum.setdefault(k, []).append(v)

            if (mbs_id + 1) in grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                grad_norm = float(grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
                reduced_aggregated = [None] * self.dp_size
                dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
                aggregated = {}
                for k in reported_accum.keys():
                    aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
                reported_accum.clear()
                if dist.get_rank() == 0:
                    log_dict = {
                        f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                    }
                    log_dict["train/grad_norm"] = grad_norm
                    lr_values = self.lr_scheduler.get_last_lr()
                    for gid, _group in enumerate(self.optimizer.param_groups):
                        log_dict[f"train/lr-pg_{gid}"] = lr_values[gid]
                    logger.info(f"step {self.global_step}: {log_dict}")
                    log_dict["train/step"] = self.global_step
                    logging_utils.log(self.args, log_dict, step_key="train/step")
                self.global_step += 1
            return

        old_log_prob_key = "rollout_log_probs" if self.args.use_rollout_logprobs else "log_probs"
        missing_old_log_probs = [
            idx
            for idx, batch in enumerate(unpacked_batches)
            if old_log_prob_key not in batch or not isinstance(batch[old_log_prob_key], torch.Tensor)
        ]
        if missing_old_log_probs:
            raise KeyError(
                f"{old_log_prob_key} must be provided as torch.Tensor for all microbatches when "
                f"use_rollout_logprobs is set to {self.args.use_rollout_logprobs}. Missing in batches: {missing_old_log_probs}"
            )
        old_log_probs = torch.cat([batch[old_log_prob_key] for batch in unpacked_batches], dim=0)
        log_probs = torch.cat([batch["cur_log_probs"] for batch in unpacked_batches], dim=0)
        advantages = torch.cat([batch["advantages"] for batch in unpacked_batches], dim=0)
        loss_masks = [batch["loss_masks"].to(device=log_probs.device) for batch in unpacked_batches]
        response_lengths = [batch["response_lengths"] for batch in unpacked_batches]

        advantages = advantages.to(device=log_probs.device)
        old_log_probs = old_log_probs.to(device=log_probs.device)
        ppo_kl = old_log_probs - log_probs

        if self.args.use_opsm:
            opsm_mask, opsm_clipfrac = compute_opsm_mask(
                args=self.args,
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                advantages=[batch["advantages"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        if self.args.advantage_estimator == "gspo":
            ppo_kl = compute_gspo_kl(
                full_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                full_old_log_probs=[batch[old_log_prob_key] for batch in unpacked_batches],
                local_log_probs=[batch["cur_log_probs"] for batch in unpacked_batches],
                loss_masks=loss_masks,
            )

        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, self.args.eps_clip, self.args.eps_clip_high)

        if self.args.use_opsm:
            pg_loss = pg_loss * opsm_mask

        def _has_rollout_log_probs(batch) -> bool:
            rollout_tensor = batch.get("rollout_log_probs")
            return isinstance(rollout_tensor, torch.Tensor) and rollout_tensor.numel() > 0

        has_rollout_log_probs = all(_has_rollout_log_probs(batch) for batch in unpacked_batches)
        rollout_log_probs = (
            torch.cat([batch["rollout_log_probs"] for batch in unpacked_batches], dim=0)
            if has_rollout_log_probs
            else None
        )

        if self.args.calculate_per_token_loss:
            pg_loss = sum_of_token(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_token(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_token(ppo_kl.abs(), response_lengths, loss_masks)
        else:
            pg_loss = sum_of_sample_mean(pg_loss, response_lengths, loss_masks)
            pg_clipfrac = sum_of_sample_mean(pg_clipfrac, response_lengths, loss_masks)
            ppo_kl = sum_of_sample_mean(ppo_kl.abs(), response_lengths, loss_masks)

        # Only compare rollout vs. train log probs when they originate from different stages.
        train_rollout_logprob_abs_diff = None
        if not self.args.use_rollout_logprobs and rollout_log_probs is not None:
            train_rollout_logprob_abs_diff = (old_log_probs - rollout_log_probs).abs()
            train_rollout_logprob_abs_diff = sum_of_sample_mean(
                train_rollout_logprob_abs_diff, response_lengths, loss_masks
            ).detach()

        entropy = torch.cat([batch["entropy"] for batch in unpacked_batches], dim=0)
        entropy_loss = sum_of_sample_mean(entropy, response_lengths, loss_masks)

        loss = pg_loss - self.args.entropy_coef * entropy_loss

        if self.args.use_kl_loss:
            ref_log_probs = torch.cat([batch["ref_log_probs"] for batch in unpacked_batches], dim=0)
            importance_ratio = None
            if self.args.use_unbiased_kl:
                importance_ratio = torch.exp(log_probs - old_log_probs)
            kl = compute_approx_kl(
                log_probs,
                ref_log_probs,
                kl_loss_type=self.args.kl_loss_type,
                importance_ratio=importance_ratio,
            )
            kl_loss = sum_of_sample_mean(kl, response_lengths, loss_masks)

            loss = loss + self.args.kl_loss_coef * kl_loss

        reported = {
            "loss": loss.detach(),
            "pg_loss": pg_loss.detach(),
            "pg_clipfrac": pg_clipfrac.detach(),
            "ppo_kl": ppo_kl.detach(),
            "entropy_loss": entropy_loss.detach(),
        }

        if train_rollout_logprob_abs_diff is not None:
            reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff

        if self.args.use_kl_loss:
            reported["kl_loss"] = kl_loss.detach()

        if self.args.use_opsm:
            reported["opsm_clipfrac"] = opsm_clipfrac

        # Scale loss for gradient accumulation
        loss = loss * self.dp_size / self.args.global_batch_size
        loss.backward()

        # Accumulate reported metrics (store tensors for later mean)
        for k, v in reported.items():
            reported_accum.setdefault(k, []).append(v)

        if (mbs_id + 1) in grad_accum:
            # TODO: check if the grad norm is global grad norm.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            # the grad norm used to be of DTensor
            grad_norm = float(grad_norm)

            self.optimizer.step()
            # Update learning rate
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Aggregate logs
            aggregated = {k: torch.stack(v).sum().item() for k, v in reported_accum.items()}
            # TODO: change this, this is slow.
            reduced_aggregated = [None] * self.dp_size
            dist.all_gather_object(reduced_aggregated, aggregated, group=self.dp_group)
            aggregated = {}
            for k in reported_accum.keys():
                aggregated[k] = sum([r[k] for r in reduced_aggregated]) / (self.args.global_batch_size)
            reported_accum.clear()
            if dist.get_rank() == 0:
                log_dict = {
                    f"train/{k}": (val.item() if torch.is_tensor(val) else val) for k, val in aggregated.items()
                }
                log_dict["train/grad_norm"] = grad_norm

                # Log learning rate per parameter group; use scheduler's last computed LRs
                lr_values = self.lr_scheduler.get_last_lr()
                for gid, _group in enumerate(self.optimizer.param_groups):
                    log_dict[f"train/lr-pg_{gid}"] = lr_values[gid]

                kl_info = ""
                if self.args.use_kl_loss and "kl_loss" in aggregated:
                    kl_info = f", kl_loss: {aggregated['kl_loss']:.4f}, kl_penalty: {aggregated['kl_loss'] * self.args.kl_loss_coef:.4f}"
                    logger.info(kl_info)
                logger.info(f"step {self.global_step}: {log_dict}")

                log_dict["train/step"] = self.global_step
                logging_utils.log(self.args, log_dict, step_key="train/step")
            self.global_step += 1

    @timer
    def update_weights(self) -> None:  # type: ignore[override]
        """Synchronize actor weights to rollout engines.

        Handles both colocated and distributed update modes. In offload mode,
        wakes up parameters as needed to perform the update.
        """
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote()
        )
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        self.weight_updater.update_weights()

        if self.args.ci_test and len(rollout_engines) > 0:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                )

        clear_memory()

    def _create_ref_model(self, ref_load_path: str | None):
        """Create and initialize a separate reference model with FSDP2 CPUOffloadPolicy.

        Parameters:
            ref_load_path: Path to a directory containing a HF checkpoint. If
                None, a ValueError is raised.

        Returns:
            FSDP2-wrapped ref model with CPU offload enabled

        Note:
            Creates a separate FSDP2 model instance for the reference model.
            ALWAYS uses CPUOffloadPolicy for the reference model to save memory,
            regardless of the actor model's CPU offload setting.
        """
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            with init_context():
                ref_model = self.get_model_cls().from_pretrained(
                    ref_load_path,
                    trust_remote_code=True,
                    attn_implementation=self.args.attn_implementation,
                )

            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(ref_model, mesh=self.dp_mesh, cpu_offload=True, args=self.args)
            ref_model = self._fsdp2_load_full_state_dict(ref_model, full_state, self.dp_mesh, cpu_offload=True)

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(
        self,
        packed_sequence: dict,
        *,
        token_key: str = "tokens",
        position_ids_key: str = "position_ids",
    ) -> dict:
        input_ids = packed_sequence[token_key].unsqueeze(0)
        position_ids = packed_sequence[position_ids_key].unsqueeze(0)

        model_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        if packed_sequence.get("multimodal_train_inputs"):
            model_args.update(packed_sequence["multimodal_train_inputs"])
        return model_args


def selective_log_softmax_raw(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Fused version of the common `log_softmax -> gather` operation.

    The fused version of this operation avoids the (potentially large) memory overhead
    of allocating a new tensor to store the full logprobs.

    Parameters:
        logits: Tensor of shape [..., V] containing model logits.
        input_ids: Tensor of shape [...] of token indices whose log-probabilities are gathered.

    Returns:
        Tensor of shape [...] containing the log-probabilities corresponding to `input_ids`.
    """
    logprobs = logits.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


selective_log_softmax_compiled = torch.compile(dynamic=True)(selective_log_softmax_raw)


def gather_log_probs_packed(
    shifted_logits: torch.Tensor,
    input_ids: torch.Tensor,
    allow_compile: bool,
    cu_seqlens: torch.Tensor | float | None = None,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather next-token log probabilities for packed sequences.

    Parameters:
        logits: Model logits of shape [B, T, V] or [T, V].
        input_ids: Token ids of shape [B, T] or [T].
        cu_seqlens: Optional cumulative sequence lengths (unused here). Present
            for API compatibility with callers.

    Returns:
        A tensor of shape [T-1] (or [B, T-1]) with log-probabilities of targets.
    """
    # Handle batch dimension - logits should be [batch_size, seq_len, vocab_size]
    if shifted_logits.dim() == 3:
        # Remove batch dimension for packed sequences
        shifted_logits = shifted_logits.squeeze(0)
        input_ids = input_ids.squeeze(0)

    if temperature is not None:
        shifted_logits = shifted_logits.div(temperature)

    targets = input_ids[1:].to(device=shifted_logits.device)

    # Gather log probs for targets
    selective_log_softmax = selective_log_softmax_compiled if allow_compile else selective_log_softmax_raw
    return selective_log_softmax(shifted_logits, targets)


def get_logprob_and_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    allow_compile: bool,
    temperature: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log probabilities and entropy.

    Parameters:
        logits: Model output logits with shape [seq_len, vocab_size]
        target_tokens: Target tokens with shape [seq_len]
        allow_compile: Whether to allow compilation
        temperature: Temperature parameter (optional)

    Returns:
        log_probs: Log probabilities with shape [seq_len - 1]
        entropy: Entropy with shape [seq_len - 1]
    """
    shifted_logits = logits[:-1, :]
    log_probs = gather_log_probs_packed(
        shifted_logits, target_tokens, allow_compile=allow_compile, temperature=temperature
    )
    log_probs_full = torch.log_softmax(shifted_logits, dim=-1)
    probs = torch.softmax(shifted_logits, dim=-1)
    entropy = -(probs * log_probs_full).sum(dim=-1)
    return log_probs, entropy


def sum_of_sample_mean(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    """Compute sum of per-sample means across variable-length responses.

    Parameters:
        x: Flat tensor containing concatenated per-token values across samples.
        response_lengths: Lengths of each sample's response segment in `x`.
        loss_masks: Per-sample masks aligned with `response_lengths`.

    Returns:
        A scalar tensor equal to the sum over samples of the mean value within
        each sample's response segment.
    """
    return sum(
        [
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )


@torch.no_grad()
def move_torch_optimizer(optimizer, device):
    """ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py"""
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=True)

    torch.cuda.synchronize()


def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None):
    """Apply FSDP v2 to the model.

    Args:
        model: The model to wrap with FSDP
        mesh: Optional DeviceMesh for FSDP. If None, uses all ranks.
        cpu_offload: If True, offload parameters, gradients, and optimizer states
            to CPU. The optimizer step will run on CPU. (Default: False)
        args: Arguments containing precision settings (fp16/bf16)

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and layer_cls_to_wrap[0] is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    # Determine precision policy based on args
    param_dtype = torch.bfloat16  # Default to bf16 as before
    reduce_dtype = torch.float32

    if args.fp16:
        param_dtype = torch.float16

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # Apply FSDP to each module (offload_policy=None is equivalent to not passing it)
    for module in modules:
        fully_shard(module, **fsdp_kwargs)

    # Apply FSDP to the top-level model
    fully_shard(model, **fsdp_kwargs)

    return model


def sum_of_token(x: torch.Tensor, response_lengths: list[int], loss_masks: list[torch.Tensor]) -> torch.Tensor:
    return sum(
        [
            (x_i * loss_mask_i).sum()
            for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
        ]
    )
