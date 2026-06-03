import base64
import os
from typing import List

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()


def visible_gpu_count(default=2):
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return default
    return max(1, len([part for part in visible_devices.split(",") if part.strip()]))


def load_slice_as_model_image(image_path: str) -> Image.Image:
    image_path = image_path.replace("'", "")
    if image_path.endswith(".npy"):
        img = np.load(image_path)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
    if image_path.endswith(".png"):
        with Image.open(image_path) as image:
            return image.convert("RGB")
    raise ValueError("all input images must be a PNG or .npy file.")


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def prepare_and_send_to_openai_api(
    png_paths,
    prompt,
    api_url="https://api.openai.com/v1/chat/completions",
    model="gpt-4o",
):
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "Content-Type": "application/json",
    }

    message_content = [{"type": "text", "text": prompt}]
    for p in png_paths:
        b64 = encode_image_base64(p)
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message_content}],
        # "max_tokens": 1000,
        "temperature": 0.0,
    }

    # print(">>> sending msg")
    resp = requests.post(api_url, headers=headers, json=payload)

    if resp.status_code != 200:
        print("API error:", resp.status_code, resp.text)
        return None

    result = resp.json()
    return result["choices"][0]["message"]["content"]


class GPTSliceVQATool:
    def run(self, png_paths, prompt):
        if isinstance(png_paths, str):
            png_paths = [png_paths]
        for image_path in png_paths:
            if image_path.endswith(".npy"):
                img = np.load(image_path)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                im = Image.fromarray((img * 255).astype(np.uint8))
                image_path = image_path.replace(".npy", ".png")
                im.save(image_path)
            elif not image_path.endswith(".png"):
                return {
                    "meta": None,
                    "outputs": "ERROR: all input images must be a PNG or .npy file.",
                }
        png_paths = [
            p if p.endswith(".png") else p.replace(".npy", ".png") for p in png_paths
        ]
        result = prepare_and_send_to_openai_api(png_paths=png_paths, prompt=prompt)
        return {"meta": None, "outputs": result}


class GeminiSliceVQATool:
    def __init__(self, model="gemini-3-pro-preview"):
        self.client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
        self.model = model

    def run(self, png_paths, prompt):
        if isinstance(png_paths, str):
            png_paths = [png_paths]
        png_paths = [p.replace("'", "") for p in png_paths]
        for image_path in png_paths:
            if image_path.endswith(".npy"):
                img = np.load(image_path)
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                im = Image.fromarray((img * 255).astype(np.uint8))
                image_path = image_path.replace(".npy", ".png")
                im.save(image_path)
            elif not image_path.endswith(".png"):
                return {
                    "meta": None,
                    "outputs": f"ERROR: all input images must be a PNG or .npy file. Got <{image_path}>.",
                }
        png_paths = [
            p if p.endswith(".png") else p.replace(".npy", ".png") for p in png_paths
        ]
        result = self.gemini_api_call(png_paths=png_paths, prompt=prompt)
        return {"meta": None, "outputs": result}

    def gemini_api_call(self, png_paths: List[str], prompt: str) -> str:
        inputs = []
        for png_path in png_paths:
            with open(png_path, "rb") as f:
                image_bytes = f.read()
            inputs.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                )
            )
        inputs.append(prompt)
        response = self.client.models.generate_content(
            model=self.model, contents=inputs
        )

        return response.text


class vLLMSliceVQATool:
    def __init__(self):
        from vllm import LLM, SamplingParams

        tensor_parallel_size = visible_gpu_count()
        self.local_model = LLM(
            # model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            # google/medgemma-1.5-4b-it
            model="google/gemma-3-27b-it",  # "Qwen/Qwen3-VL-32B-Instruct-FP8", google/gemma-3-27b-it "OpenGVLab/InternVL2_5-4B"
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=8192,
            gpu_memory_utilization=0.8,
            # enforce_eager=True,
            # disable_custom_all_reduce=True,
            max_num_seqs=4,  # 4 works
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(temperature=0.00, max_tokens=6000)

    def run_alternative(self, png_paths: List[str], prompt: str) -> dict:
        if isinstance(png_paths, str):
            png_paths = [png_paths]
        content = []

        instruction = "You are a clinical expert analyzing a several chest CT slices. Please review the slices provided below carefully."
        content.append({"type": "text", "text": instruction})

        for i, image_path in enumerate(png_paths, 1):
            content.append({"type": "text", "text": f"SLICE {i}"})
            image_path = image_path.replace("'", "")
            if image_path.endswith(".npy"):
                content.append(
                    {
                        "type": "image_pil",
                        "image_pil": load_slice_as_model_image(image_path),
                    }
                )
            elif image_path.endswith(".png"):
                content.append(
                    {
                        "type": "image_pil",
                        "image_pil": load_slice_as_model_image(image_path),
                    }
                )
            elif image_path.endswith(".nii") or image_path.endswith(".nii.gz"):
                return {
                    "meta": None,
                    "outputs": "ERROR: CT volumes in NIfTI format are not supported. Select slices first.",
                }
            else:
                return {
                    "meta": None,
                    "outputs": "ERROR: all input images must be a PNG or .npy file.",
                }

        content.append({"type": "text", "text": prompt})
        content.append(
            {
                "type": "text",
                "text": "Your response should be concise and focus on the main findings, ideally one paragraph only. Do NOT add disclaimer statements.",
            }
        )
        with torch.no_grad():
            outputs = self.local_model.chat(
                [{"role": "user", "content": content}],
                sampling_params=self.sampling_params,
            )
        result = outputs[0].outputs[0].text
        return {"meta": None, "outputs": result}


if __name__ == "__main__":
    from fastmcp import FastMCP
    from tool_configs import args_tools

    args = args_tools()

    mcp = FastMCP("see", stateless_http=False)

    slice_vqa_tool_instance = vLLMSliceVQATool()

    @mcp.tool()
    async def slice_vqa_tool(image_paths: List[str], question: str) -> dict:
        return slice_vqa_tool_instance.run_alternative(
            png_paths=image_paths, prompt=question
        )

    mcp.run(transport="http", host=args.host, port=args.port)
