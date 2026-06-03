import json
from typing import List, Dict, Any, Tuple
import torch

import numpy as np

import os
from vllm import SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs
import uuid


def get_llm_judge_prompts(report1: str, report2: str, model: str) -> Tuple[str, str]:
    if "think" in model.lower():
        prompt1 = f"""You are given two CT reports. You need to assess their similarities. For this you are given a precise set of instructions:
    
    ### Instructions
            
    1. List ALL findings in the ground truth report and in the candidate. There should at least be one finding in each report. Store these as two lists called "all_findings_in_ground_truth" and "all_findings_in_candidate".
    
    2. List all ABNORMAL findings in the ground truth report and in the candidate report. An abnormal finding is any finding that indicates a pathology or deviation from normal anatomy or function. For example "no pleural effusion" is a normal finding, while "presence of pleural effusion" is an abnormal finding. Each finding should be a short description, for example "pleural effusion in the right lung", "enlarged heart", "nodule in the left lung", "atelectasis in the lower left lung", etc. Store these as two lists called "all_abnormal_findings_in_ground_truth" and "all_abnormal_findings_in_candidate".
    Examples: 
        - Report: "Findings: No occlusive pathology was observed in the trachea and lumen of both main bronchi. In the non-contrast examination, the mediastinal could not be evaluated optimally. As far as can be seen; mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Pleuroparenchymal sequela fibrotic density increases were observed in the apical and posterior segment of the right lung upper lobe, and in the left lung upper lobe apicoposterior segment, which also causes pleural thickening. In both lungs, nonspecific parenchymal nodules with a diameter of 7.1 mm were observed in the anterobasal subsegment of the lower lobe anterobasal segment, the largest of which was 7.1 mm on the right, and 3 mm in diameter, on the left. No mass lesion-active infiltration with distinguishable borders was detected in the lung parenchyma. As far as can be seen within the sections; upper abdominal organs are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Osteopenia was observed in the thoracolumbar vertebrae within the sections. Vertebral corpus heights are natural. Impression:  Sequelae changes in the right lung upper lobe and left lung upper lobe apicoposterior segment.  Millimetrically sized nonspecific parenchymal nodules in both lungs.  Osteopenia in the thoracolumbar vertebrae."
        Then the list of abnormal findings in the report would be: ["Pleuroparenchymal sequela fibrotic density increases in the apical and posterior segment of the right lung upper lobe, and in the left lung upper lobe apicoposterior segment", "pleural thickening", "millimetrically sized nonspecific parenchymal nodules in both lungs", "osteopenia in the thoracolumbar vertebrae"]
        - Report: "In both lungs, nodules compatible with diffuse metastases are observed in almost all zones, which tend to merge from place to place", then add "nodules compatible with diffuse metastases in both lungs" to the list of abnormal findings.

    3. For every finding in "all_abnormal_findings_in_ground_truth" you will check whether this finding is matched in the candidate report, if this is NOT the case you add the finding to the "abnormal_findings_in_ground_truth_missing_in_candidate". And vice-versa.
    Partially matched findings: if the ground truth report and the candidate report mentions the same abnormal finding but in a different location, consider that the finding is partially matched, do NOT include it in the "abnormal_findings_in_ground_truth_missing_in_candidate" and "abnormal_findings_in_candidate_missing_in_ground_truth" list. Use the two new lists "abnormal_findings_in_ground_truth_partially_matched_in_candidate" and "abnormal_findings_in_candidate_partially_matched_in_ground_truth" to store these partially matched findings.
    Example: 
        - Ground truth report: "There is a nodule in the lower right lung". Candidate report: "Presence of nodules". That is both report mention the same abnormality but mention different locations, sizes etc. In this case you should add "nodule in lower right lung" in to "abnormal_findings_in_ground_truth_partially_matched_in_candidate" list AND add "presence of nodules" in the "abnormal_findings_in_candidate_partially_matched_in_ground_truth" list.  
    IMPORTANT: If one report mentions a normal finding, or the absence of a finding, and the other report does not mention anything about that finding, consider that the two reports are matched on that finding. In other words, not mentioning a finding is considered equivalent to reporting a normal finding or the absence of a pathology.

    
    ### Reports to analyse
    
    Ground truth report:
    {report1}

    Candidate report:
    {report2}

    ### Answer format
    IMPORTANT: Your answer should be in the format: 
    
        {{
            "all_findings_in_ground_truth": [list of findings],
            "all_findings_in_candidate": [list of findings],
            "all_abnormal_findings_in_ground_truth": [list of findings],
            "all_abnormal_findings_in_candidate": [list of findings],
            "abnormal_findings_in_ground_truth_missing_in_candidate": [list of findings],
            "abnormal_findings_in_candidate_missing_in_ground_truth": [list of findings],
            "abnormal_findings_in_ground_truth_partially_matched_in_candidate": [list of findings],
            "abnormal_findings_in_candidate_partially_matched_in_ground_truth": [list of findings],
        }} 
    
    Do not include any other text in your answer.
    DO NOT EXPLAIN YOUR ANSWER. Do not include tags like ```json or similar, just give the JSON.
    """

        # Second reflection prompt
        prompt2 = f"""Double check that your previous answer adheres to the previously given instructions and correct any mistakes you may have made.
        
    In particular, check the following:
    1. Any finding listed in "abnormal_findings_in_candidate_missing_in_ground_truth" is truly not mentioned in the ground truth report at all. If it is exactly mentioned in the ground truth report, remove it from the "abnormal_findings_in_candidate_missing_in_ground_truth" list. If it is partially mentioned in the ground truth report, move it to the partially matched list.
    2. Any finding listed in "abnormal_findings_in_ground_truth_missing_in_candidate" is truly not mentioned in the candidate report.  If it is exactly mentioned in the ground truth report, remove it from the "abnormal_findings_in_ground_truth_missing_in_candidate" list. If it is partially mentioned in the ground truth report, move it to the partially matched list.
    3. "all_abnormal_findings_in_ground_truth", "all_abnormal_findings_in_candidate" contain only abnormal findings. If not remove the normal findings from these lists.
    4. Any finding listed in "all_abnormal_findings_in_ground_truth" that is missing from the candidate report is present in the "abnormal_findings_in_ground_truth_missing_in_candidate" list.
    5. Any finding listed in "all_abnormal_findings_in_candidate" that is missing from the ground truth report is present in the "abnormal_findings_in_candidate_missing_in_ground_truth" list.
    4. "all_findings_in_ground_truth", "all_findings_in_candidate" contain all findings (both normal and abnormal) mentioned in each report. If not, complete these lists accordingly.
    5. Any finding listed in "abnormal_findings_in_ground_truth_partially_matched_in_candidate" is truly partially mentioned in the candidate report (according to the previous definition of partial match). If not move the finding to the missing list.
    6. Any finding listed in "abnormal_findings_in_candidate_partially_matched_in_ground_truth" is truly partially mentioned in the ground truth report (according to the previous definition of partial match). If not move the finding to the missing list.
    7. The list of "abnormal_findings_in_ground_truth_partially_matched_in_candidate" and "abnormal_findings_in_ground_truth_missing_in_candidate" are disjoint. If any finding appears in both lists, remove it from the "abnormal_findings_in_ground_truth_missing_in_candidate" list.
    8. The list of "abnormal_findings_in_candidate_partially_matched_in_ground_truth" and "abnormal_findings_in_candidate_missing_in_ground_truth" are disjoint. If any finding appears in both lists, remove it from the "abnormal_findings_in_candidate_missing_in_ground_truth" list.
    9. All findings in the list "abnormal_findings_in_ground_truth_missing_in_candidate" appear in "all_abnormal_findings_in_ground_truth". If not, complete the "all_abnormal_findings_in_ground_truth" list accordingly. IMPORTANT: the findings must match exactly.
    10. All findings in the list "abnormal_findings_in_candidate_missing_in_ground_truth" appear in "all_abnormal_findings_in_candidate". If not, complete the "all_abnormal_findings_in_candidate" list accordingly. IMPORTANT: the findings must match exactly.
    11. All findings in the list "abnormal_findings_in_ground_truth_partially_matched_in_candidate" appear in "all_abnormal_findings_in_ground_truth". If not, complete the "all_abnormal_findings_in_ground_truth" list accordingly. IMPORTANT: the findings must match exactly.
    12. All findings in the list "abnormal_findings_in_candidate_partially_matched_in_ground_truth" appear in "all_abnormal_findings_in_candidate". If not, complete the "all_abnormal_findings_in_candidate" list accordingly. IMPORTANT: the findings must match exactly.
    
    MAKE SURE YOU CHECKED ALL THESE POINTS CAREFULLY!
    
    Provide the corrected answer in the same JSON format as before, without any additional text."""
        all_prompts = [prompt1, prompt2]

    else:
        prompt1 = f"""You are given two CT reports. You need to assess their similarities. For this you are given a precise set of instructions:
            
            ### Instructions
                    
            1. List ALL findings in the ground truth report and in the candidate. There should at least be one finding in each report. Store these as two lists called "all_findings_in_ground_truth" and "all_findings_in_candidate".
            
            2. List all ABNORMAL findings in the ground truth report and in the candidate report. An abnormal finding is any finding that indicates a pathology or deviation from normal anatomy or function. For example "no pleural effusion" is a normal finding, while "presence of pleural effusion" is an abnormal finding. Each finding should be a short description, for example "pleural effusion in the right lung", "enlarged heart", "nodule in the left lung", "atelectasis in the lower left lung", etc. Store these as two lists called "all_abnormal_findings_in_ground_truth" and "all_abnormal_findings_in_candidate".
            Examples: 
                - Report: "Findings: No occlusive pathology was observed in the trachea and lumen of both main bronchi. In the non-contrast examination, the mediastinal could not be evaluated optimally. As far as can be seen; mediastinal main vascular structures, heart contour, size are normal. Pericardial effusion-thickening was not observed. Thoracic esophagus calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Pleuroparenchymal sequela fibrotic density increases were observed in the apical and posterior segment of the right lung upper lobe, and in the left lung upper lobe apicoposterior segment, which also causes pleural thickening. In both lungs, nonspecific parenchymal nodules with a diameter of 7.1 mm were observed in the anterobasal subsegment of the lower lobe anterobasal segment, the largest of which was 7.1 mm on the right, and 3 mm in diameter, on the left. No mass lesion-active infiltration with distinguishable borders was detected in the lung parenchyma. As far as can be seen within the sections; upper abdominal organs are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected. Osteopenia was observed in the thoracolumbar vertebrae within the sections. Vertebral corpus heights are natural. Impression:  Sequelae changes in the right lung upper lobe and left lung upper lobe apicoposterior segment.  Millimetrically sized nonspecific parenchymal nodules in both lungs.  Osteopenia in the thoracolumbar vertebrae."
                Then the list of abnormal findings in the report would be: ["Pleuroparenchymal sequela fibrotic density increases in the apical and posterior segment of the right lung upper lobe, and in the left lung upper lobe apicoposterior segment", "pleural thickening", "millimetrically sized nonspecific parenchymal nodules in both lungs", "osteopenia in the thoracolumbar vertebrae"]
                - Report: "In both lungs, nodules compatible with diffuse metastases are observed in almost all zones, which tend to merge from place to place", then add "nodules compatible with diffuse metastases in both lungs" to the list of abnormal findings.

            3. For every finding in "all_abnormal_findings_in_ground_truth" you will check whether this finding is matched in the candidate report, if this is NOT the case you add the finding to the "abnormal_findings_in_ground_truth_missing_in_candidate". And vice-versa.
            Partially matched findings: if the ground truth report and the candidate report mentions the same abnormal finding but in a different location, consider that the finding is partially matched, do NOT include it in the "abnormal_findings_in_ground_truth_missing_in_candidate" and "abnormal_findings_in_candidate_missing_in_ground_truth" list. Use the two new lists "abnormal_findings_in_ground_truth_partially_matched_in_candidate" and "abnormal_findings_in_candidate_partially_matched_in_ground_truth" to store these partially matched findings.
            Example: 
                - Ground truth report: "There is a nodule in the lower right lung". Candidate report: "Presence of nodules". That is both report mention the same abnormality but mention different locations, sizes etc. In this case you should add "nodule in lower right lung" in to "abnormal_findings_in_ground_truth_partially_matched_in_candidate" list AND add "presence of nodules" in the "abnormal_findings_in_candidate_partially_matched_in_ground_truth" list.  
            IMPORTANT: If one report mentions a normal finding, or the absence of a finding, and the other report does not mention anything about that finding, consider that the two reports are matched on that finding. In other words, not mentioning a finding is considered equivalent to reporting a normal finding or the absence of a pathology.

            
            ### Reports to analyse
            
            Ground truth report:
            {report1}

            Candidate report:
            {report2}

            ### Answer format
            IMPORTANT: Your answer should be in the format: 
            
                {{
                    "all_findings_in_ground_truth": [list of findings],
                    "all_findings_in_candidate": [list of findings],
                    "all_abnormal_findings_in_ground_truth": [list of findings],
                    "all_abnormal_findings_in_candidate": [list of findings],
                    "abnormal_findings_in_ground_truth_missing_in_candidate": [list of findings],
                    "abnormal_findings_in_candidate_missing_in_ground_truth": [list of findings],
                    "abnormal_findings_in_ground_truth_partially_matched_in_candidate": [list of findings],
                    "abnormal_findings_in_candidate_partially_matched_in_ground_truth": [list of findings],
                }} 
            
            Do not include any other text in your answer.
            DO NOT EXPLAIN YOUR ANSWER. Do not include tags like ```json or similar, just give the JSON.
            """
        prompt2 = f"""Double check that your previous answer adheres to the previously given instructions and correct any mistakes you may have made.
            
        In particular, check the following:
        1. Any finding listed in "abnormal_findings_in_candidate_missing_in_ground_truth" is truly not mentioned in the ground truth report at all. If it is exactly mentioned in the ground truth report, remove it from the "abnormal_findings_in_candidate_missing_in_ground_truth" list. If it is partially mentioned in the ground truth report, move it to the partially matched list.
        2. Any finding listed in "abnormal_findings_in_ground_truth_missing_in_candidate" is truly not mentioned in the candidate report.  If it is exactly mentioned in the ground truth report, remove it from the "abnormal_findings_in_ground_truth_missing_in_candidate" list. If it is partially mentioned in the ground truth report, move it to the partially matched list.
        3. "all_abnormal_findings_in_ground_truth", "all_abnormal_findings_in_candidate" contain only abnormal findings. If not remove the normal findings from these lists.
        4. "all_findings_in_ground_truth", "all_findings_in_candidate" contain all findings (both normal and abnormal) mentioned in each report. If not, complete these lists accordingly.
        5. Any finding listed in "abnormal_findings_in_ground_truth_partially_matched_in_candidate" is truly partially mentioned in the candidate report (according to the previous definition of partial match). If not move the finding to the missing list.
        6. Any finding listed in "abnormal_findings_in_candidate_partially_matched_in_ground_truth" is truly partially mentioned in the ground truth report (according to the previous definition of partial match). If not move the finding to the missing list.

        MAKE SURE YOU CHECKED ALL THESE POINTS CAREFULLY!
        
        Provide the corrected answer in the same JSON format as before, without any additional text."""

        prompt3 = f"""Double check that your previous answer adheres to the previously given instructions and correct any mistakes you may have made.
            
        Check the following:
        1. The list of "abnormal_findings_in_ground_truth_partially_matched_in_candidate" and "abnormal_findings_in_ground_truth_missing_in_candidate" are disjoint. If any finding appears in both lists, remove it from the "abnormal_findings_in_ground_truth_missing_in_candidate" list.
        2. The list of "abnormal_findings_in_candidate_partially_matched_in_ground_truth" and "abnormal_findings_in_candidate_missing_in_ground_truth" are disjoint. If any finding appears in both lists, remove it from the "abnormal_findings_in_candidate_missing_in_ground_truth" list.
        3. All findings in the list "abnormal_findings_in_ground_truth_missing_in_candidate" appear in "all_abnormal_findings_in_ground_truth". If not, complete the "all_abnormal_findings_in_ground_truth" list accordingly. IMPORTANT: the findings must match exactly.
        4. All findings in the list "abnormal_findings_in_candidate_missing_in_ground_truth" appear in "all_abnormal_findings_in_candidate". If not, complete the "all_abnormal_findings_in_candidate" list accordingly. IMPORTANT: the findings must match exactly.
        5. All findings in the list "abnormal_findings_in_ground_truth_partially_matched_in_candidate" appear in "all_abnormal_findings_in_ground_truth". If not, complete the "all_abnormal_findings_in_ground_truth" list accordingly. IMPORTANT: the findings must match exactly.
        6. All findings in the list "abnormal_findings_in_candidate_partially_matched_in_ground_truth" appear in "all_abnormal_findings_in_candidate". If not, complete the "all_abnormal_findings_in_candidate" list accordingly.  IMPORTANT: the findings must match exactly.

        MAKE SURE YOU CHECKED ALL THESE POINTS CAREFULLY!
        
        Provide the corrected answer in the same JSON format as before, without any additional text.
        """

        all_prompts = [prompt1, prompt2, prompt3]

    return all_prompts


def process_response(resp_text: str, verbose: bool = False) -> Dict[str, Any]:
    try:
        json_response = json.loads(
            resp_text.replace("```json", "").replace("```", "").strip()
        )
    except Exception as e:
        if verbose:
            print(f"Error parsing JSON: {e}", flush=True)
            print("Response text was:")
            print(resp_text, flush=True)
        json_response = {
            "equivalent": "NO",
            "all_findings_in_ground_truth": [],
            "all_findings_in_candidate": [],
            "abnormal_findings_in_ground_truth_missing_in_candidate": [],
            "abnormal_findings_in_candidate_missing_in_ground_truth": [],
            "abnormal_findings_in_ground_truth_partially_matched_in_candidate": [],
            "abnormal_findings_in_candidate_partially_matched_in_ground_truth": [],
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
        return json_response

    if verbose:
        print(json_response["abnormal_findings_in_ground_truth_missing_in_candidate"])
        print(json_response["abnormal_findings_in_candidate_missing_in_ground_truth"])
        print(
            json_response[
                "abnormal_findings_in_candidate_partially_matched_in_ground_truth"
            ]
        )
        print(json_response["all_findings_in_ground_truth"])
        print(json_response["all_findings_in_candidate"])

        inconsistencies_findings = [
            f
            for f in json_response[
                "abnormal_findings_in_ground_truth_missing_in_candidate"
            ]
            if f not in json_response["all_findings_in_ground_truth"]
        ]
        if len(inconsistencies_findings) > 0:
            print(
                "WARN Inconsistencies found in ground truth missing findings. These are missing from ground truth findings:",
                inconsistencies_findings,
                json_response["all_findings_in_ground_truth"],
            )
        inconsistencies_findings = [
            f
            for f in json_response[
                "abnormal_findings_in_candidate_missing_in_ground_truth"
            ]
            if f not in json_response["all_findings_in_candidate"]
        ]
        if len(inconsistencies_findings) > 0:
            print(
                "WARN Inconsistencies found in candidate missing findings. These are missing from candidate findings:",
                inconsistencies_findings,
                json_response["all_findings_in_candidate"],
            )

    total_gt = len(json_response["all_findings_in_ground_truth"])
    total_cand = len(json_response["all_findings_in_candidate"])
    abnormal_gt = len(json_response["all_abnormal_findings_in_ground_truth"])
    abnormal_cand = len(json_response["all_abnormal_findings_in_candidate"])
    # 10 findings, 4 missing, 2 partially matched means we have 4 matching findings and 2 partial matches so we have 5 / 10 score
    #
    match_abnormal_gt = (
        abnormal_gt
        - len(json_response["abnormal_findings_in_ground_truth_missing_in_candidate"])
    ) - 0.5 * len(
        json_response[
            "abnormal_findings_in_ground_truth_partially_matched_in_candidate"
        ]
    )
    match_abnormal_gt = max(0, match_abnormal_gt)
    match_abnormal_cand = (
        abnormal_cand
        - len(json_response["abnormal_findings_in_candidate_missing_in_ground_truth"])
    ) - 0.5 * len(
        json_response[
            "abnormal_findings_in_candidate_partially_matched_in_ground_truth"
        ]
    )
    match_abnormal_cand = max(0, match_abnormal_cand)
    match_gt = (
        total_gt
        - len(json_response["abnormal_findings_in_ground_truth_missing_in_candidate"])
        - 0.5
        * len(
            json_response[
                "abnormal_findings_in_ground_truth_partially_matched_in_candidate"
            ]
        )
    )
    match_gt = max(0, match_gt)

    match_cand = (
        total_cand
        - len(json_response["abnormal_findings_in_candidate_missing_in_ground_truth"])
        - 0.5
        * len(
            json_response[
                "abnormal_findings_in_candidate_partially_matched_in_ground_truth"
            ]
        )
    )
    match_cand = max(0, match_cand)

    recall = match_gt / (total_gt + 1e-12)
    precision = match_cand / (total_cand + 1e-12)

    abnormal_recall = match_abnormal_gt / abnormal_gt if abnormal_gt > 0 else 1.0
    abnormal_precision = (
        match_abnormal_cand / abnormal_cand if abnormal_cand > 0 else 1.0
    )

    json_response["all_findings_recall"] = recall
    json_response["all_findings_precision"] = precision
    json_response["all_findings_f1"] = (
        2 * recall * precision / (recall + precision + 1e-12)
    )
    json_response["abnormal_recall"] = abnormal_recall
    json_response["abnormal_precision"] = abnormal_precision
    json_response["abnormal_f1"] = (
        2
        * abnormal_recall
        * abnormal_precision
        / (abnormal_recall + abnormal_precision + 1e-12)
    )
    json_response["abnormal_mean_rec_prec"] = (abnormal_recall + abnormal_precision) / 2

    if verbose:
        print(f"All findings RECALL / PRECISION {recall:.2f} / {precision:.2f}")
        print(
            f"Abnormal findings RECALL / PRECISION {abnormal_recall:.2f} / {abnormal_precision:.2f}"
        )

    return json_response


def process_tool_sequence_judge_output(output_text: str):
    keys = [
        "tool sequence coherence",
        # "tool diversity",
        "checklist adherence",
    ]  # 'overall trace quality'
    result = {}
    try:
        output = json.loads(output_text)
        for key in keys:
            result[key] = output.get(key, {})
    except:
        pass
    return result


def create_tool_sequence_judge_prompt(data):
    return f"""
    You are given a conversation trace between an AI, tool and a human user, your role is to reflect on the quality of the generated trace. In particular, you should check the following:
    - Is the tool sequence coherent? If the AI request a tool call, it should use the tool output appropriately. The AI should never call the same tool with the exact same arguments more than once. The tool sequence should not be unecessary long. Do not need check whether the AI has double checked its findings here. Please give a score between 1 and 5. 
    - Did the AI check every item on the diagnosis checklist? Please give a score between 1 and 5.

    Please provide your final answer EXACTLY as follows:
    {{
    'tool sequence coherence': {{'reasoning': your short explanation, 'score': your score}},
    'checklist adherence': {{'reasoning': your short explanation, 'score': your score}}
    }}
    <trace>
    {{{data[:-1]}}}
    </trace>
    """

def create_hallucination_judge_prompt(data):
    return f"""
    You are given a conversation trace between an AI, tool and a human user, your role is to reflect on the faithfulness of the final report compared to the generated trace. In particular, you should check the following:
    - Final answer hallucination score: Does the final answer contain any findings that are not supported by the tool sequence, i.e. that do not appear in any of the tool response? Give a score between 1 and 5 as well as number of hallucinated findings and number of total findings.
    - Preliminary findings hallucination score: Does the preliminary_findings list contain any findings that are not supported by the tool sequence? Only assess the preliminary_findings list in the final turn of the conversation.

    Please provide your final answer EXACTLY as follows:
    {{
    'answer hallucination score': {{'reasoning': your short explanation, 'score': your score, 'num_hallucinated_findings': num_hallucinated_findings, 'num_total_findings': num_total_findings}},
    'preliminary findings hallucination score': {{'reasoning': your short explanation, 'score': your score, 'num_hallucinated_findings': num_hallucinated_findings, 'num_total_findings': num_total_findings}},
    }}

    <trace>
    {{{data[:-1]}}}
    </trace>
    """

class ReportJudgeTool:
    def __init__(
        self,
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        sampling_params=SamplingParams(temperature=0.00, max_tokens=12000),
    ):

        os.environ["VLLM_USE_V1"] = "1"
        self.model_name = model_name

        # Use AsyncLLMEngine for parallel processing
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32000,
            gpu_memory_utilization=0.60,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            max_num_seqs=24,
        )
        self.judge = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = None  # Will be loaded lazily
        self.sampling_params = sampling_params

    async def _get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = await self.judge.get_tokenizer()
        return self.tokenizer

    async def _generate_async(self, messages: List[Dict[str, str]]) -> str:
        """Async generation using the async engine"""
        tokenizer = await self._get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        request_id = str(uuid.uuid4())

        results_generator = self.judge.generate(
            prompt, self.sampling_params, request_id
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        output_text = final_output.outputs[0].text
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1].strip()
        return output_text

    async def run_report_judge(
        self, ground_truth_report: str, candidate_report: str
    ) -> dict:
        all_prompts = get_llm_judge_prompts(
            ground_truth_report, candidate_report, model=self.model_name
        )
        messages = []
        for prompt_idx in range(len(all_prompts)):
            prompt = all_prompts[prompt_idx]
            messages.append({"role": "user", "content": prompt})

            output_text = await self._generate_async(messages)

            messages.append({"role": "assistant", "content": output_text})

        try:
            result = process_response(output_text, verbose=True)

        except Exception as e:
            print(f"Error processing response {e}", flush=True)

        return {"meta": None, "outputs": json.dumps(result)}

    async def run_tool_sequence_judge(self, conversation_trajectory: list) -> dict:
        prompt = create_tool_sequence_judge_prompt(conversation_trajectory)
        messages = [
            {"role": "system", "content": "You are an expert AI judge."},
            {"role": "user", "content": prompt},
        ]
        output_text = await self._generate_async(messages)
        result = process_tool_sequence_judge_output(output_text)
        return {"meta": None, "outputs": json.dumps(result)}

    async def run_summary_judge(self, conversation_trajectory: list) -> dict:
        # Placeholder for summary judge implementation
        prompt = f"""You are an AI judge. Analyze the provided conversation to verify that the final report incorporates all findings detected by the tools. If there is contradictory evidence between tools, check that the final report resolves it by reflecting the majority opinion or the strongest evidence. The final report does not need to include which tools were used, nor mentioning explicitely if one tool disagree with the others on a specific finding as long as the majority opinion for each finding is present in the final report.
        
        Summarize your analysis in one short paragraph, focusing only on missing findings or failures to align with the majority evidence.

        <trace>
        {{{conversation_trajectory[:-1]}}}
        </trace>
        """
        messages = [
            {"role": "system", "content": "You are an expert AI judge."},
            {"role": "user", "content": prompt},
        ]
        output_text = await self._generate_async(messages)
        return {"meta": None, "outputs": output_text}

    async def run_ct_chat_judge(self, conversation_trajectory: list) -> dict:
        prompt = f"""You are an AI judge. Analyze the provided CT Chat conversation to verify that the final report incorporates all findings supported across the conversation. If there is contradictory evidence between earlier report drafts or reflection rounds, check that the final report resolves it by reflecting the majority opinion or the strongest evidence. The final report does not need to mention earlier uncertainty or disagreement explicitly, as long as the most supported findings are present.

        Summarize your analysis in one short paragraph, focusing only on missing findings, of failures to align with the majority evidence. 

        <conversation>
        {{{conversation_trajectory}}}
        </conversation>
        """
        messages = [
            {"role": "system", "content": "You are an expert AI judge."},
            {"role": "user", "content": prompt},
        ]
        output_text = await self._generate_async(messages)
        return {"meta": None, "outputs": output_text}

if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()
    mcp = FastMCP("see", stateless_http=False)
    report_judge_tool_instance = ReportJudgeTool(
        model_name="Qwen/Qwen3-30B-A3B-Thinking-2507",
        sampling_params=SamplingParams(temperature=0.1, top_p=0.95, max_tokens=25000),
    )

    @mcp.tool()
    async def report_judge_tool(
        ground_truth_report: str, candidate_report: str
    ) -> dict:
        return await report_judge_tool_instance.run_report_judge(
            ground_truth_report=ground_truth_report,
            candidate_report=candidate_report,
        )

    @mcp.tool()
    async def trajectory_judge_tool(
        conversation_trajectory: List[dict],
    ) -> dict:
        return await report_judge_tool_instance.run_tool_sequence_judge(
            conversation_trajectory
        )

    @mcp.tool()
    async def summary_judge_tool(
        conversation_trajectory: List[dict],
    ) -> dict:
        return await report_judge_tool_instance.run_summary_judge(
            conversation_trajectory
        )

    @mcp.tool()
    async def ct_chat_judge(
        conversation_trajectory: List[dict],
    ) -> dict:
        return await report_judge_tool_instance.run_ct_chat_judge(
            conversation_trajectory
        )

    mcp.run(transport="http", host=args.host, port=args.port)
