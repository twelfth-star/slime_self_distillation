from slime.rollout.data_source import RolloutDataSourceWithBuffer


class SelfDistillRolloutDataSource(RolloutDataSourceWithBuffer):
    """Rollout data source that augments samples with teacher prompt metadata.

    Teacher prompt is read from sample metadata using:
    args.self_distill_teacher_prompt_key (default: "teacher_prompt").
    """

    def get_samples(self, num_samples: int):
        samples = super().get_samples(num_samples)
        teacher_prompt_key = getattr(self.args, "self_distill_teacher_prompt_key", "teacher_prompt")

        for group in samples:
            for sample in group:
                metadata = sample.metadata or {}
                teacher_prompt = metadata.get(teacher_prompt_key)
                if teacher_prompt is None:
                    sample.metadata = metadata
                    continue

                metadata.setdefault("_student_prompt", sample.prompt)
                metadata["_teacher_prompt"] = teacher_prompt
                sample.metadata = metadata

        return samples
