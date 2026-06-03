from typing import Any, Dict, List

from evaluation.hallucination_study.hallu_utils.config_resolving import EvalLabelingConfig
from evaluation.hallucination_study.hallu_utils.JudgeProcessor import JudgeProcessor

class TrajectoryJudgeProcessor(JudgeProcessor):
    """
    Label full agent trajectories for explicit admission of user-provided hints.

    This processor evaluates each assistant turn in a trajectory and aggregates the
    step-level judgements into a trajectory-level summary used by the study.
    """
    def __init__(self, config: EvalLabelingConfig):
        super().__init__(config=config)

    def label_batch(self, trajectories: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """
        Return one result dict per trajectory.

        Each result dict contains:
        - admits_using_hint: bool
        - admits_using_hint_per_step: list of
          {"message_index": int, "admits_using_hint": 0 or 1}
        """
        label_prompts, label_routing = self.build_labelling_prompt_structs(trajectories=trajectories)

        if not label_prompts:
            return [
                {
                    "admits_using_hint": False,
                    "admits_using_hint_per_step": [],
                }
                for _ in trajectories
            ]

        label_sampling_params = self._init_label_sampling_params()

        if self.model_config.enable_thinking:
            label_outputs = self.llm.chat(
                label_prompts,
                sampling_params=label_sampling_params,
                chat_template_kwargs={"enable_thinking": self.model_config.enable_thinking},
            )
        else:
            label_outputs = self.llm.chat(
                label_prompts,
                sampling_params=label_sampling_params,
            )
        if self.config.run_config.judge_type == "strict":
            judgements = self.extract_step_judgements_from_json(
                label_routing=label_routing,
                label_outputs=label_outputs,
            )
        
        else:
            judgements = self.extract_step_judgements(
                label_routing=label_routing,
                label_outputs=label_outputs,
            )
        

        return self.aggregate_trajectory_judgements(
            judgements=judgements,
            num_trajectories=len(trajectories),
        )

    def aggregate_trajectory_judgements(
        self,
        judgements: Dict[int, Dict[int, bool]],
        num_trajectories: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for traj_index in range(num_trajectories):
            traj_judgements = judgements.get(traj_index, {})

            per_step = [
                {
                    "message_index": message_index,
                    "admits_using_hint": int(label),
                }
                for message_index, label in sorted(traj_judgements.items())
            ]

            results.append(
                {
                    "admits_using_hint": any(traj_judgements.values()),
                    "admits_using_hint_per_step": per_step,
                }
            )

        return results

    def build_labelling_prompt_structs(
        self,
        trajectories: List[List[Dict[str, str]]],
    ) -> tuple[List[List[Dict[str, str]]], List[tuple[int, int]]]:
        """
        Build one labelling prompt per assistant message.
        label_routing stores (trajectory_index, message_index).
        """
        label_prompts: List[List[Dict[str, str]]] = []
        label_routing: List[tuple[int, int]] = []

        for traj_index, trajectory_messages in enumerate(trajectories):
            hint_message = self._extract_first_user_message(trajectory_messages)
            if hint_message is None:
                continue

            for message_index, message in enumerate(trajectory_messages):
                if message.get("role") != "assistant":
                    continue

                assistant_content = message.get("content", "")
                traj_labelling_prompt = self._build_label_prompts(
                    hint_message=hint_message,
                    assistant_message=assistant_content,
                )
                label_prompts.append(traj_labelling_prompt)
                label_routing.append((traj_index, message_index))

        return label_prompts, label_routing
