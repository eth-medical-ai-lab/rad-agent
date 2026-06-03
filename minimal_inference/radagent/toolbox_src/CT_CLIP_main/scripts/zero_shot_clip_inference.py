import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformer_maskgit import CTViT
from ct_clip.ct_clip import CTCLIP
from functools import partial
import sys
from constants_and_path_utils import CT_CHAT_MODELS, CT_RATE_ROOT

sys.path.append(Path(__file__).parent.as_posix())


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )
    return resized_array


def apply_softmax(array):
    """
    Applies softmax function to a torch array.

    Args:
        array (torch.Tensor): Input tensor array.

    Returns:
        torch.Tensor: Tensor array after applying softmax.
    """
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array


class SingleFileInference_clip:
    def __init__(self, model_path, default_meta=False):
        """
        Initialize single file inference.

        Args:
            model_path (str): Path to the pre-trained model
            meta_df (pd.DataFrame, optional): Metadata dataframe. If None, default values will be used.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True
        )
        if default_meta:
            self.meta_df = pd.read_csv(CT_RATE_ROOT / "metadata/train_metadata.csv")
        else:
            self.meta_df = None
        # Initialize model architecture
        text_encoder = BertModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized"
        )
        text_encoder.resize_token_embeddings(len(self.tokenizer))

        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
        )

        self.clip = CTCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            extra_latent_projection=False,
            use_mlm=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )

        # Load pre-trained weights
        self.clip.load(model_path)
        self.clip.to(self.device)
        self.clip.eval()

        # Default metadata if not provided
        self.default_metadata = {
            "RescaleSlope": 1.0,
            "RescaleIntercept": -1024.0,
            "XYSpacing": 0.75,
            "ZSpacing": 1.5,
        }

        # Define pathologies
        self.pathologies = [
            "Medical material",
            "Arterial wall calcification",
            "Cardiomegaly",
            "Pericardial effusion",
            "Coronary artery wall calcification",
            "Hiatal hernia",
            "Lymphadenopathy",
            "Emphysema",
            "Atelectasis",
            "Lung nodule",
            "Lung opacity",
            "Pulmonary fibrotic sequela",
            "Pleural effusion",
            "Mosaic attenuation pattern",
            "Peribronchial thickening",
            "Consolidation",
            "Bronchiectasis",
            "Interlobular septal thickening",
        ]

    def nii_to_tensor(self, nii_path):
        """
        Convert NIfTI file to preprocessed tensor.

        Args:
            nii_path (str): Path to the NIfTI file
            accession_name (str, optional): Accession name for metadata lookup

        Returns:
            torch.Tensor: Preprocessed CT tensor
        """
        # print(f"Processing: {nii_path}")
        nii_img = nib.load(str(nii_path))
        img_data = nii_img.get_fdata()

        # Get metadata
        if self.meta_df is not None:
            file_name = nii_path.split("/")[-1]
            row = self.meta_df[self.meta_df["VolumeName"] == file_name]
            slope = 1  # float(row["RescaleSlope"].iloc[0])
            intercept = 0  # float(row["RescaleIntercept"].iloc[0])
            xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
            z_spacing = float(row["ZSpacing"].iloc[0])

        else:
            slope, intercept, xy_spacing, z_spacing = self.default_metadata.values()

        # Define target spacing
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        # Apply rescale and windowing
        img_data = slope * img_data + intercept
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Resize array
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        # Normalize
        img_data = (((img_data) / 1000)).astype(np.float32)

        # Center crop/pad to target shape
        tensor = torch.tensor(img_data)
        target_shape = (480, 480, 240)

        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=-1,
        )

        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor

    def predict_single_file(self, nii_path):
        """
        Perform inference on a single NIfTI file.

        """
        with torch.no_grad():
            # Preprocess the CT scan
            ct_tensor = self.nii_to_tensor(nii_path)
            ct_tensor = ct_tensor.to(self.device)

            results = {}

            for pathology in self.pathologies:
                # Create positive and negative prompts
                text = [f"{pathology} is present.", f"{pathology} is not present."]
                text_tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                # Get model output
                output = self.clip(text_tokens, ct_tensor, device=self.device)
                output = apply_softmax(output)

                # print('out:', output)

                # Get probability of presence
                presence_prob = output[0].item()
                results[pathology] = {
                    "probability": presence_prob,
                }

            return results

    def predict_and_generate_report(self, nii_path):

        results = self.predict_single_file(nii_path)

        # Create results dataframe
        data = []
        for pathology, result in results.items():
            probability = result["probability"]
            data.append(
                f"Pathology: {pathology}, Prediction: {'Positive' if probability > 0.5 else 'Negative'}, Probability: {probability:.4f}"
            )

        return "\n".join(data)

    def predict_and_save(self, nii_path, output_path=None):
        """
        Perform inference and save results to CSV.

        Args:
            nii_path (str): Path to the NIfTI file
            output_path (str, optional): Path to save results CSV
            accession_name (str, optional): Accession name for metadata lookup

        Returns:
            pd.DataFrame: Results dataframe
        """
        results = self.predict_single_file(nii_path)

        # Create results dataframe
        df_data = []
        for pathology, result in results.items():
            df_data.append(
                {
                    "Pathology": pathology,
                    "Probability": result["probability"],
                    "Prediction": result["prediction"],
                }
            )

        df = pd.DataFrame(df_data)

        # Save to file if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            # print(f"Results saved to: {output_path}")

        return df


# Usage example
if __name__ == "__main__":
    # Initialize the inference class
    inference = SingleFileInference_clip(
        model_path=CT_CHAT_MODELS
        / "models/CT-CLIP-Related/CT_VocabFine_v2.pt",  # Update this path
    )

    # Perform inference on a single file
    nii_file_path = "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_4/train_4_a/train_4_a_1.nii.gz"  # Update this path
    results_df = inference.predict_and_generate_report(
        nii_path=nii_file_path,
        output_path="single_file_results.csv",
    )

    # Print results
    # "\nPrediction Results:")
    # print("=" * 50)
    # print(results_df)
