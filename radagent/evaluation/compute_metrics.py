"""
Main metrics definition for report generation evaluation.
Contains:
- compute_nlp_metrics: computes standard NLP metrics (BLEU, METEOR, ROUGE, CIDER)
- compute_green_score: computes GREEN score
- compute_multilabel_ct_classification_metrics: computes multi-label classification metrics for CT pathology classification
"""

from typing import Dict, Optional, Tuple
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from evaluation.green_score.green import GREEN
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import pandas as pd

from distributed_utils import tqdm_on_main

from evaluation.text_classifier_CT_pathology import (
    CTPathologyClassifier,
    CTClassifierInferenceDataset,
)
from constants_and_path_utils import PATHOLOGIES_LIST
import time


def compute_nlp_metrics(
    ground_truth_dict: Dict[str, str], output_dict: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Computes standard NLP metrics, following https://github.com/salaniz/pycocoevalcap implementation.

    Args:
        ground_truth_dict: Dictionary {'image_id': 'ground_truth'}
        output_dict: Dictionary {'image_id': 'output'}

    Returns:
        average_scores: Dictionary with evaluation scores for each measure (the mean of the scores of all the instances)
        all_scores: Dictionary with evaluation scores for each measure for each instance
    """

    # Set up scorers
    scorers = [
        (Bleu(1), ["BLEU_1"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "Cider"),
    ]
    if len(ground_truth_dict.keys()) == 1:
        print(
            "WARNING: CIDER metric requires at least 2 sentences to compute score. Skipping CIDER."
        )
        scorers = scorers[:-1]
    # The NLP scorers require a slightly different input format
    ground_truth_dict = {k: [v] for k, v in ground_truth_dict.items()}
    output_dict = {k: [v] for k, v in output_dict.items()}
    eval_res = {}
    detailed_eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        print(f"Computing {method} score...")
        if isinstance(scorer, Bleu):
            score, scores = scorer.compute_score(
                ground_truth_dict, output_dict, verbose=False
            )
        else:
            score, scores = scorer.compute_score(ground_truth_dict, output_dict)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                eval_res[m] = float(sc)
                detailed_eval_res[m] = scs
        else:
            eval_res[method] = float(score)
            detailed_eval_res[method] = scores
    return eval_res, detailed_eval_res


def compute_green_score(
    ground_truth_dict: Dict[str, str], output_dict: Dict[str, str]
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], pd.DataFrame]:
    """
    Computes GREEN score, following https://stanford-aimi.github.io/green.html implementation.

    Args:
        ground_truth_dict: Dictionary {'image_id': 'ground_truth'}
        output_dict: Dictionary {'image_id': 'output'}

    Returns:
        average_scores: Dictionary with {'GREEN': avg score}
        all_scores: Dictionary with evaluation scores for each measure for each instance
    """
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".", cpu=False)
    refs = []
    outs = []
    for id in ground_truth_dict.keys():
        refs.append(ground_truth_dict[id])
        outs.append(output_dict[id])
    mean, _, green_score_list, _, result_df = green_scorer(refs, outs)
    detailed_eval_res = {"GREEN": green_score_list}
    eval_res = {"GREEN": float(mean) if mean is not None else None}
    return eval_res, detailed_eval_res, result_df


def compute_multilabel_ct_classification_metrics(
    ground_truth_dict: Dict[str, str], output_dict: Dict[str, str], batch_size: int
) -> Optional[Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Computes multi-label classification metrics (accuracy, precision, recall, f1-score) for CT pathology classification.

    Args:
        ground_truth_dict: Dictionary {'image_id': [list of binary labels]}
        output_dict: Dictionary {'image_id': [list of binary labels]}
    Returns:
        results: Dictionary with accuracy, precision, recall, f1-score
    """
    t = time.time()
    t = time.time() - t

    torch.cuda.set_device(0)
    model = CTPathologyClassifier("cuda:0")
    model.eval()
    dataset = CTClassifierInferenceDataset(
        ground_truth_dict=ground_truth_dict,
        output_dict=output_dict,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=12
    )

    dist_gt_labels = []
    dist_pred_labels = []
    volume_ids = []
    for batch in tqdm_on_main(
        iterable=dataloader,
        total=len(dataloader),
    ):
        result = model.predict_binary(batch)
        dist_gt_labels.append(result["ground_truth"])
        dist_pred_labels.append(result["predicted"])
        volume_ids.extend(result["volume_id"])

    print("Inference done")
    all_gt_labels = torch.cat(dist_gt_labels, dim=0).cpu()
    all_pred_labels = torch.cat(dist_pred_labels, dim=0).cpu()

    print("Gathering done")

    return {
        "accuracy": accuracy_score(all_gt_labels, all_pred_labels),
        "f1_macro": f1_score(all_gt_labels, all_pred_labels, average="macro"),
        "f1_micro": f1_score(all_gt_labels, all_pred_labels, average="micro"),
        "classification_report": classification_report(
            all_gt_labels, all_pred_labels, digits=4
        ),
    }, {
        "gt": all_gt_labels,
        "pred": all_pred_labels,
        "equal": all_gt_labels == all_pred_labels,
        "volume_id": volume_ids,
    }


def compute_multilabel_ct_predictions(
    output_dict: Dict[str, str], batch_size: int
) -> Optional[Tuple[Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Gets multi-label classification predictions for CT pathology classification.

    Args:
        output_dict: Dictionary {'image_id': [list of binary labels]}
    Returns:
        results: Dataframe with pathology predictions for each instance
    """
    t = time.time()
    t = time.time() - t

    torch.cuda.set_device(0)
    model = CTPathologyClassifier("cuda:0")
    model.eval()
    dataset = CTClassifierInferenceDataset(
        ground_truth_dict=output_dict,
        output_dict=output_dict,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=12
    )

    dist_pred_labels = []
    volume_ids = []
    for batch in tqdm_on_main(
        iterable=dataloader,
        total=len(dataloader),
    ):
        result = model.predict_binary(batch)
        dist_pred_labels.append(result["predicted"])
        volume_ids.extend(result["volume_id"])

    print("Inference done")
    all_pred_labels = torch.cat(dist_pred_labels, dim=0).cpu()
    results_dict = {"image_id": volume_ids}
    for i, p in enumerate(PATHOLOGIES_LIST):
        results_dict[f"pred_{p}"] = all_pred_labels[:, i].float().tolist()
    return pd.DataFrame(results_dict)
