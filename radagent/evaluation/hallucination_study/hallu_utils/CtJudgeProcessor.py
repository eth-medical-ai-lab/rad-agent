from typing import List, Dict, Any
from evaluation.hallucination_study.hallu_utils.JudgeProcessor import JudgeProcessor
from evaluation.hallucination_study.hallu_utils.config_resolving import EvalLabelingConfig

class CtJudgeProcessor(JudgeProcessor):
    """
    Report-level hint-admission labeling for CT-Chat style generated reports.

    The processor wraps the shared judge model utilities with batching logic for
    single-report outputs instead of multi-turn trajectories.
    """
    def __init__(self, config: EvalLabelingConfig):
        super().__init__(config)

    def label_batch_reports(self, batch: List[Dict[str, str]]) -> List[Dict[str, bool]]:
        label_prompts = self.build_ctchat_labelling_prompts(batch=batch)
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
            judgements = [
                {"admits_using_hint": bool(self._parse_judgement_from_json(output.outputs[0].text))}
                for output in label_outputs
                ]
        else:
            judgements = [
                {"admits_using_hint": bool(self._parse_judgement(output.outputs[0].text))}
                for output in label_outputs
                ]
        return judgements 

    def build_ctchat_labelling_prompts(self, batch: List[Dict[str, str]]):
        # Batch dict has keys generated_report and query_with_hint
        return [
            self._build_label_prompts(
                hint_message=case["query_with_hint"],
                assistant_message=case["generated_report"]
                )
            for case in batch
        ]
