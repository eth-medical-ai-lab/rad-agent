from pathlib import Path
import sys
import nibabel as nib

import torch
import torch.nn as nn

sys.path.append(Path(__file__).parent.as_posix())
from transformer_maskgit import CTViT
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)
import numpy as np
import pandas as pd
import torch.nn.functional as F

from constants_and_path_utils import CT_CHAT_MODELS, CT_RATE_ROOT


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


def nii_img_to_tensor(path):
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()
    spacing_xyz = np.diag(np.abs(nii_img.affine))[:3]
    xy_spacing = spacing_xyz[0]
    z_spacing = spacing_xyz[2]

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    # If not found then don't do anything
    # do_resize = False
    # if (xy_spacing is not None) or (z_spacing is not None):

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)
    do_resize = True

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    if do_resize:
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]

    # print(img_data.shape, flush=True)
    # print(img_data.shape, flush=True)
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = np.clip(img_data, -1000, 1000)
    img_data = (((img_data) / 1000)).astype(np.float32)

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
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

    return tensor.cuda()


class CTChat_full_model(nn.Module):
    def __init__(
        self, device="cuda", merge_lora_weights=True, return_base_model_only=False
    ):
        super().__init__()
        self.image_encoder = (
            CTViT(
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
            .cuda()
            .eval()
        )

        self.device = device
        self.df = pd.read_csv(CT_RATE_ROOT / "metadata/train_metadata.csv")

        # image encoder
        self.image_encoder.load(CT_CHAT_MODELS / "models/CT-CLIP-Related/encoder.pth")
        self.image_encoder = self.image_encoder.to(self.device)

        # llm
        self.conv_mode = "llama3"
        self.temperature = 0.0
        self.max_new_tokens = 512

        self.model_path = CT_CHAT_MODELS / "llava-lora-llama_3.1_8B"
        self.model_base = CT_CHAT_MODELS / "models/meta-llama/Llama-3.1-8B-Instruct"
        self.model_name = get_model_name_from_path(self.model_path)

        self.load_4bit = False
        self.load_8bit = False
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                self.model_path,
                self.model_base,
                self.model_name,
                self.load_8bit,
                self.load_4bit,
                device=self.device,
                merge_lora_weights=merge_lora_weights,
                return_base_model_only=return_base_model_only,
            )
        )
        print("#####" * 5)
        print("loading finish!")

    def run_batch(self, queries, image_paths):
        batch_size = len(queries)
        image_tensors = []
        prompts = []
        image_sizes = []

        failed_indices = []
        for i in range(batch_size):
            query = queries[i]
            image_path = image_paths[i]

            # try:
            #     file_name = image_path.split("/")[-1]
            #     row = self.df[self.df["VolumeName"] == file_name]
            #     xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
            #     z_spacing = float(row["ZSpacing"].iloc[0])
            # except Exception as e:
            #     try:
            #         file_name = (
            #             image_path.split("/")[-2].replace(".nii", "").replace(".gz", "")
            #             + ".nii.gz"
            #         )
            #         row = self.df[self.df["VolumeName"] == file_name]
            #         xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
            #         z_spacing = float(row["ZSpacing"].iloc[0])
            #     except Exception as e:
            #         xy_spacing = None
            #         z_spacing = None
            #         print(f"Error in finding spacing for {image_path}: {e}", flush=True)
            try:
                image = nii_img_to_tensor(path=image_path)
            except Exception as e:
                failed_indices.append(i)
                continue

            image = image.to(self.device)
            image_tensors.append(image)
            # print(image.shape, flush=True)
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
        # print(prompts, flush=True)
        # [B, C, H, W]
        if len(image_tensors) > 0:
            images = torch.cat(image_tensors, dim=0).to(self.device)
            images = self.image_encoder(images, return_encoded_tokens=True)
            image_sizes = [images.shape[1:]] * images.shape[0]
            images = images.to(dtype=torch.float16)

            self.tokenizer.padding_side = "left"

            input_ids = tokenizer_image_token(
                prompts, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )

            input_ids = input_ids.to(self.model.device)
            # print(IMAGE_TOKEN_INDEX, flush=True)
            # print(input_ids[0], flush=True)
            # print(input_ids.shape, flush=True)
            # print(images.shape, flush=True)
            # print(image_sizes, flush=True)
            # print(input_ids.shape, flush=True)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        outputs_f = []
        c = 0
        for i in range(batch_size):
            if i in failed_indices:
                outputs_f.append("ERROR: file not found.")
            else:
                outputs_f.append(outputs[c].strip())
                c += 1
        return outputs_f


if __name__ == "__main__":
    # path = "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_4/train_4_a/train_4_a_1.nii.gz"
    # nii_img = nii_img_to_tensor(path)
    model = CTChat_full_model(device="cuda:1")
    print(
        model.run_batch(
            [
                "<image>\nWhat are the findings in the lung parenchyma of the Chest CT image?<long_answer>"
            ],
            [
                "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_4/train_4_a/train_4_a_1.nii.gz"
            ],
        )
    )

    print(
        model.run_batch(
            [
                "<image>\n<provided>"
                + "Please generate the CT report"
                + "<report_generation>"
            ],
            [
                "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/CT-RATE/dataset/train_fixed/dataset/train_fixed/train_4/train_4_a/train_4_a_1.nii.gz"
            ],
        )
    )
