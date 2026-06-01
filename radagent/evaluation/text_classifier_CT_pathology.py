"""
CT Pathology Report Classifier definition to compute
multi-label CT classification metrics (18 patholgies from CT-RATE).
"""

from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from constants_and_path_utils import CT_CHAT_MODELS, PATHOLOGIES_LIST

# Adapted from https://github.com/ibrahimethemhamamci/CT-CLIP/blob/main/text_classifier/classifier.py

class CTPathologyClassifier(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained("zzxslp/RadBERT-RoBERTa-4m")
        self.model = AutoModel.from_pretrained(
            "zzxslp/RadBERT-RoBERTa-4m", config=self.config
        )
        model_path = CT_CHAT_MODELS / "models/RadBertClassifier.pth"
        self.classifier = nn.Linear(self.model.config.hidden_size, 18)
        print(self.load_state_dict(torch.load(model_path), strict=False))
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.to(self.device)

    def forward(self, batch):
        result = {}
        for k in ["predicted", "ground_truth"]:
            input_ids = batch[k]["input_ids"].squeeze(1).to(self.device)
            attn_mask = batch[k]["attention_mask"].squeeze(1).to(self.device)
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            output = torch.sigmoid(self.classifier(output.pooler_output))
            result[k] = output
        return result

    def predict_binary(self, batch):
        result = self.forward(batch)
        for k in ["predicted", "ground_truth"]:
            result[k] = result[k].detach() > 0.5
        if "volume_id" in batch.keys():
            result["volume_id"] = batch["volume_id"]
        return result


class CTClassifierInferenceDataset(Dataset):
    def __init__(self, ground_truth_dict: Dict[str, str], output_dict: Dict[str, str]):
        self.ground_truth_dict = ground_truth_dict
        self.volumes_id = list(output_dict.keys())
        self.output_dict = output_dict
        self.tokenizer = AutoTokenizer.from_pretrained(
            "zzxslp/RadBERT-RoBERTa-4m", do_lower_case=True
        )
        self.max_length = 512

    def __len__(self):
        return len(self.volumes_id)

    def __getitem__(self, idx):
        volume_id = self.volumes_id[idx]
        output = self.output_dict[volume_id]
        ground_truth_report = self.ground_truth_dict[volume_id]
        output_encodings = self.tokenizer(
            output,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        ground_truth_encodings = self.tokenizer(
            ground_truth_report,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "predicted": output_encodings,
            "ground_truth": ground_truth_encodings,
            "volume_id": volume_id,
        }


"""
To get the model weights

hf_hub_download(repo_id=repo_id,
    repo_type='dataset',
    token=hf_token,
    filename='models/RadBertClassifier.pth',
    local_dir=str(CT_CHAT_MODELS),
    local_dir_use_symlinks=False,
    resume_download=True,
    )
"""
