import asyncio
import os
import random
import sys
from typing import Optional, Unpack

import numpy as np
from peft import PeftModel, get_peft_model_state_dict
import wandb

sys.path.append(
    "/capstor/store/cscs/swissai/a135/wp3-agents/workspace/3dragent-agent-mel/radagent"
)
from ct_chat_full_model import CTChat_full_model
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from tools.tool_configs import JUDGING_TOOLS, SERVERS
from utils import kill_job
from agents.server_manager import MultiServerManager

from agents.art_dataset import generate_report_generation_scenarios
from datetime import datetime
import pandas as pd

from ct_chat_full_model import nii_img_to_tensor
import torch

from torch import nn
import transformers

import json
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin
from transformers.utils.chat_template_utils import render_jinja_template
from transformers.processing_utils import AllKwargsForChatTemplate
from typing import Any, Optional, TypedDict, Union


def preprocess_for_grpo(row, image_encoder, metadata_df, conv_mode, device):
    """
    This replaces the logic inside your 'run' function.
    It processes a single row of your metadata.
    """
    image_path = row["image_path"]  # Or your specific path column

    # 2. Convert NIfTI to Tensor
    image = nii_img_to_tensor(path=image_path)

    # 3. Encode Image (Preprocessing this saves massive VRAM during GRPO)
    image = image.to(device)

    image = image_encoder(image, return_encoded_tokens=True)
    image_size = image.shape
    # print(image_size, flush=True)
    # shape is [B, H, W, C]
    image = torch.tensor(image, dtype=torch.float16)

    # 4. Format the Prompt
    system_prompt = "You are CT-CHAT, an AI assistant specializing in Chest CT imaging, dedicated to providing accurate and relevant information exclusively related to Chest CT scans and associated medical topics. You are equipped to answer questions and offer detailed analyses only when the CT volume/scan/image is provided, indicated by the <provided> token. If this token is not present and users inquire about specific findings, pathologies, or request descriptions related to a Chest CT, respond by requesting the necessary data with the phrase: “Please provide the CT volume.” Once the <provided> token is present in the question, you are authorized to address questions about pathologies, anatomical or clinical findings, diagnostic descriptions, report generation, comparisons, or any other questions regarding the image. If it does not appear in the question, even when special tokens <multiple_choice>, <report_generation>, <long_answer>, and <short_answe> are given, ignore the question and ask for the CT volume. Always look for the <provided> token, even if there are special tokens. If there is a <provided> token in any question (including new and previous ones), never ask for the CT volume again and answer the question. You can ignore the <provided> token check and answer the question directly if and only if the question is about general medical knowledge, not about the provided CT volume (such as typical findings on a Chest CT or management of the patient). For example, “What are the typical imaging findings of acute respiratory distress syndrome (ARDS) on a chest CT?” is a general question not specific. If user asks a CT specific question after non-spesific question, look for the <provided> token as well even if the special tokens are given. It is crucial to avoid discussing topics outside of Chest CT imaging and directly related medical information, ensuring that all responses are clear, concise, and focused on the provided Chest CT data for the highest level of accuracy and relevance. If the user greets you with something like “hello,” respond appropriately."

    # conv = conv_templates[conv_mode].copy()
    # conv.append_message(conv.roles[0], row['task']) # 'query' is your question column
    # conv.append_message(conv.roles[1], None)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "<provided>" + row["original_query"]},
    ]
    return {
        "prompt": conversation,
        "image": image.cpu(),  # Shape: [Tokens, Hidden_Dim]
        "answer": row["gt"],  # Ground truth for reward calculation
        "image_size": image_size,
    }


class GRPO_CTChat_Wrapper(nn.Module):
    def __init__(self, ct_chat_model):
        super().__init__()
        # We only want to train the LLM (and potentially the projector)
        self.model = ct_chat_model.model
        # print(isinstance(self.model, PeftModel))
        self.tokenizer = ct_chat_model.tokenizer
        ct_chat_model.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.actual_image_token_id = ct_chat_model.tokenizer.convert_tokens_to_ids(
            "<image>"
        )

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        image_size = [pixel_values.shape[1:]] * pixel_values.shape[0]
        # print('Pixel values require grad', pixel_values.requires_grad)
        # print('Input id require grad', input_ids.requires_grad)
        # print('Attention require grad', attention_mask.requires_grad)
        self.model.print_trainable_parameters()
        pixel_values.requires_grad_(True)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            image_sizes=image_size,
            **kwargs,
        )

    def generate(self, *args, **kwargs):
        # Delegate generation to the underlying LLM
        # print(kwargs.keys())
        # print('Args passed to generate', str(args))
        # print(str(kwargs['generation_config']))
        kwargs["images"] = kwargs.pop("pixel_values", None)
        kwargs["inputs"] = kwargs.pop("input_ids", None)
        kwargs["image_sizes"] = [kwargs["images"].shape[1:]] * kwargs["images"].shape[0]
        # print('Calling generate')
        output_ids = self.model.generate(*args, **kwargs)
        # print(output)
        # intput_to_decode = kwargs['inputs'][1]
        # intput_to_decode = torch.where(
        #     intput_to_decode == -200,
        #     self.actual_image_token_id,
        #     intput_to_decode
        # )
        # print(self.tokenizer.decode(intput_to_decode, skip_special_tokens=False))
        print(
            "OUTPUT is", self.tokenizer.decode(output_ids[1], skip_special_tokens=False)
        )
        # The GRPO trainer expects the output to be input_ids + completions
        full_sequence_ids = torch.cat([kwargs["inputs"], output_ids], dim=1)
        # print('Full sequence ids shape', full_sequence_ids.shape)
        # prompt_length = kwargs['inputs'].size(1)
        # completion_ids = full_sequence_ids[:, prompt_length:]
        # print(completion_ids == output_ids)
        return full_sequence_ids

    def __getattr__(self, name):
        """
        CRITICAL: This allows the GRPOTrainer to access attributes like
        'config', 'device', 'dtype', and 'peft_config' directly from the
        internal Llama model as if the wrapper were the model itself.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def state_dict(self, *args, **kwargs):
        """
        Force return of only the adapter weights.
        We skip the 'isinstance' check to be safe, assuming self.model is always PEFT.
        """
        # print('kwards state_dict', kwargs)
        return get_peft_model_state_dict(self.model)

    # def save_pretrained(self, output_dir, **kwargs):
    #     """
    #     Explicitly delegate saving to the PEFT model.
    #     This ensures 'adapter_model.bin' is saved instead of 'pytorch_model.bin'.
    #     """
    #     print('kwards save_pretrained', kwargs)
    #     self.model.save_pretrained(output_dir, **kwargs)

    # def state_dict(self, *args, **kwargs):
    #     """Return only trainable (adapter) parameters"""
    #     if isinstance(self.model, PeftModel):
    #         # Use PEFT's method to get only adapter weights
    #         return get_peft_model_state_dict(self.model)
    #     return super().state_dict(*args, **kwargs)


class EmbeddingProcessor(ProcessorMixin):
    def __init__(self, tokenizer):
        # We still need the original tokenizer to handle the text side
        self.tokenizer = tokenizer
        self.chat_template = self.tokenizer.chat_template

        # Identify the special token used for images (e.g., "<image>")
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.feature_extractor = (
            None  # Placeholder for image feature extractor for saving purposes
        )

    def apply_chat_template(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if (
                isinstance(self.chat_template, dict)
                and chat_template in self.chat_template
            ):
                # It's the name of a template, not a full template string
                chat_template = self.chat_template[chat_template]
            else:
                # It's a template string, render it directly
                chat_template = chat_template

        if kwargs.get("continue_final_message", False):
            if kwargs.get("add_generation_prompt", False):
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if kwargs.get("return_assistant_tokens_mask", False):
                raise ValueError(
                    "continue_final_message is not compatible with return_assistant_tokens_mask."
                )

        # Fill sets of kwargs that should be used by different parts of template
        processed_kwargs = {
            "mm_load_kwargs": {},
            "template_kwargs": {},
        }

        for kwarg_type in processed_kwargs:
            for key in AllKwargsForChatTemplate.__annotations__[
                kwarg_type
            ].__annotations__.keys():
                kwarg_type_defaults = AllKwargsForChatTemplate.__annotations__[
                    kwarg_type
                ]
                default_value = getattr(kwarg_type_defaults, key, None)
                value = kwargs.pop(key, default_value)
                if value is not None and not isinstance(value, dict):
                    processed_kwargs[kwarg_type][key] = value

        # Pass unprocessed custom kwargs
        processed_kwargs["template_kwargs"].update(kwargs)

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple))
            or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        tokenize = processed_kwargs["template_kwargs"].pop("tokenize", False)
        return_dict = processed_kwargs["template_kwargs"].pop("return_dict", False)
        mm_load_kwargs = processed_kwargs["mm_load_kwargs"]

        # if tokenize:
        batch_images = []
        text_only_conversations = []
        for conversation in conversations:
            images = []
            text_conversation = []
            for message in conversation:
                images = [
                    content["image"]
                    for content in message["content"]
                    if content["type"] in ["image"]
                ]
                text_content = [
                    content["text"]
                    for content in message["content"]
                    if content["type"] == "text"
                ]
                if len(text_content) > 0:
                    assert (
                        len(text_content) == 1
                    ), "Each message should have only one text content."
                    text_conversation.append(
                        {
                            "role": message["role"],
                            "content": text_content[0],
                        }
                    )
            if images:
                batch_images.append(images)
            text_only_conversations.append(text_conversation)
        # print('Batch images 0', batch_images[0])
        # print(text_only_conversations[0])
        prompt, generation_indices = render_jinja_template(
            conversations=text_only_conversations,
            chat_template=chat_template,
            **processed_kwargs[
                "template_kwargs"
            ],  # different flags such as `return_assistant_mask`
            **self.tokenizer.special_tokens_map,  # tokenizer special tokens are used by some templates
        )
        # print(prompt[0])

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(
                self.tokenizer.bos_token
            ):
                kwargs["add_special_tokens"] = False

            out = self(
                text=prompt,
                images=batch_images if batch_images else None,
                **kwargs,
            )
            if return_dict:
                if processed_kwargs["template_kwargs"].get(
                    "return_assistant_tokens_mask", False
                ):
                    assistant_masks = []
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        for (
                            assistant_start_char,
                            assistant_end_char,
                        ) in generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(
                                start_token,
                                end_token + 1 if end_token else len(input_ids[i]),
                            ):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(
                        tensor_type=kwargs.get("return_tensors", None)
                    )
                return out
            else:
                return out["input_ids"]
        return prompt

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors="pt",
        videos=None,
        audio=None,
        **kwargs,
    ):
        # 1. Standard Tokenization
        # encoding = self.tokenizer.apply_chat_template(text, return_tensors=return_tensors)
        # print(kwargs)
        # print('Input for images',  images[0].shape if isinstance(images, torch.Tensor) else len(images[0]))
        # print(text[0])
        encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        encoding["input_ids"] = torch.where(
            encoding["input_ids"] == self.image_token_id, -200, encoding["input_ids"]
        )
        # 2. Attach the pre-computed embeddings to the output dictionary
        # We rename the key so the model's forward method can identify it
        if images is not None:
            # print(len(images), len(images[0]))
            for i in images:
                assert len(i) == 1, "Each message should contain exactly one image."
            images = torch.cat([torch.tensor(i[0]) for i in images])
            images = images.squeeze(dim=1)
            encoding["pixel_values"] = images.half()
            # print(images.shape)
            # encoding["image_sizes"] = encoding["pixel_values"].shape

        return encoding

    def batch_decode(self, token_ids, skip_special_tokens=True, **kwargs):
        # Need to reconvert the weird token from CT-Chat for decoding
        if isinstance(token_ids, torch.Tensor):
            token_ids = torch.where(token_ids == -200, self.image_token_id, token_ids)
        else:
            token_ids = [
                [self.image_token_id if x == -200 else x for x in sublist]
                for sublist in token_ids
            ]

        return self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def save_pretrained(self, save_directory):
        self.tokenizer.save_pretrained(save_directory)
        # Save feature extractor if needed
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory)


class CustomGRPOTrainer(GRPOTrainer):
    """Custom trainer that saves only PEFT adapters"""

    def _save(self, output_dir=None, state_dict=None):
        """Override save to use PEFT's save method"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Get the wrapped model
        model = self.model
        if hasattr(model, "model"):
            inner_model = model.model
            if isinstance(inner_model, PeftModel):
                print(f"✓ Saving PEFT adapter to {output_dir}")
                inner_model.save_pretrained(output_dir)
                # Also save tokenizer
                if hasattr(model, "tokenizer"):
                    model.tokenizer.save_pretrained(output_dir)
                return

        # Fallback to default behavior
        super()._save(output_dir, state_dict)

    def _load_from_checkpoint(self, checkpoint_path):
        """Override load to use PEFT's load method"""
        model = self.model
        if hasattr(model, "model"):
            inner_model = model.model
            if isinstance(inner_model, PeftModel):
                print(f"✓ Loading PEFT adapter from {checkpoint_path}")
                model.model = PeftModel.from_pretrained(
                    model.base_model.model, checkpoint_path, is_trainable=True
                )
                return

        # Fallback to default behavior
        super()._load_from_checkpoint(checkpoint_path)


class RewardComputer:
    def __init__(self):
        server_dict = {
            t: SERVERS[t] for t in ["report_judge_tool", "f1_text_classifier_tool"]
        }
        server_dict["report_judge_tool"]["device"] = "2"
        self.server_manager = MultiServerManager(servers=server_dict)
        asyncio.run(self.server_manager.startup_all_servers())
        asyncio.run(self.server_manager.connect_all())

    def __name__(self):
        return "RewardComputer"

    async def compute_single_item_reward(self, answer, completion):
        llm_judge_task = self.server_manager.call_tool(
            "report_judge_tool",
            "report_judge_tool",
            {
                "candidate_report": completion[-1]["content"],
                "ground_truth_report": answer,
            },
        )
        # green_task = self.server_manager.call_tool(
        #     "green_tool",
        #     "green_tool",
        #         {"candidate_report": completion[-1]['content'], "ground_truth_report": answer},
        #     )

        f1_task = self.server_manager.call_tool(
            "f1_text_classifier_tool",
            "f1_text_classifier_tool",
            {
                "candidate_report": completion[-1]["content"],
                "ground_truth_report": answer,
            },
        )
        # green_reward, f1_classifier_reward = await asyncio.gather(green_task, f1_task)
        # llm_as_judge_reward, green_reward, f1_classifier_reward = await asyncio.gather(llm_judge_task, green_task, f1_task)
        llm_as_judge_reward, f1_classifier_reward = await asyncio.gather(
            llm_judge_task, f1_task
        )
        try:
            llm_reward_dict = json.loads(llm_as_judge_reward)
        except Exception as e:
            print(f"Error parsing LLM judge reward: {e}", flush=True)
            llm_reward_dict = {}
        f1_llm = llm_reward_dict.get("abnormal_f1", 0.0)
        mean_prec_rec = llm_reward_dict.get("abnormal_mean_rec_prec", 0.0)
        try:
            green_reward = float(green_reward)
        except:
            green_reward = 0.0
        try:
            f1_classifier_reward = float(f1_classifier_reward)
        except:
            f1_classifier_reward = 0.0
        # return f1_classifier_reward + 0.5 * green_reward, 0.0, 0.0, green_reward, f1_classifier_reward
        return (
            f1_llm + f1_classifier_reward,
            f1_llm,
            mean_prec_rec,
            0.0,
            f1_classifier_reward,
        )
        # return f1_llm + mean_prec_rec + green_reward + f1_classifier_reward, f1_llm, mean_prec_rec, green_reward, f1_classifier_reward

    async def compute_reward(self, answers, completions):
        try:
            await self.server_manager.connect_all()
        except:
            kill_job()
        all_reward_tasks = [
            self.compute_single_item_reward(answer, completion)
            for answer, completion in zip(answers, completions)
        ]
        results = await asyncio.gather(*all_reward_tasks)
        rewards, f1, mean_prec_rec, green_reward, f1_classifier_reward = zip(*results)
        if wandb.run is not None:
            wandb.log({"metrics/mean_rewards": np.asarray(rewards).mean()})
            wandb.log({"metrics/mean_f1_llm": np.asarray(f1).mean()})
            wandb.log(
                {"metrics/mean_f1_classifier": np.asarray(f1_classifier_reward).mean()}
            )
            wandb.log({"metrics/mean_prec_rec": np.asarray(mean_prec_rec).mean()})
            wandb.log({"metrics/mean_green_reward": np.asarray(green_reward).mean()})
        print("Rewards mean:", np.asarray(rewards).mean(), flush=True)
        print("F1 mean:", np.asarray(f1).mean(), flush=True)
        print("Mean Prec Rec mean:", np.asarray(mean_prec_rec).mean(), flush=True)
        print("Green Reward mean:", np.asarray(green_reward).mean(), flush=True)
        return rewards

    def __call__(self, completions, answer, **kwargs):
        return asyncio.run(self.compute_reward(answer, completions))


if __name__ == "__main__":
    # 1. Initialize your existing model class
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    chat_instance = CTChat_full_model(merge_lora_weights=False)
    model = chat_instance.model
    tokenizer = chat_instance.tokenizer
    image_encoder = chat_instance.image_encoder

    model.train()
    model.print_trainable_parameters()

    # Prepare dataset
    scenarios = generate_report_generation_scenarios(mode="train", end_idx=5000)
    train_dataset = Dataset.from_pandas(pd.DataFrame(scenarios))

    # 2. Define a wrapper to handle batching (set_transform always receives a batch)
    def lazy_preprocess(batch):
        processed_samples = []
        for i in range(len(batch["image_path"])):
            # Reconstruct single item dict if your func expects it
            single_item = {k: v[i] for k, v in batch.items()}

            # Call your existing logic
            processed = preprocess_for_grpo(
                single_item,
                image_encoder=image_encoder,
                metadata_df=chat_instance.df,
                conv_mode=chat_instance.conv_mode,
                device=model.device,
            )
            processed_samples.append(processed)
        collated = {k: [d[k] for d in processed_samples] for k in processed_samples[0]}
        return collated

    # NO processing happens here. It just registers the function.
    train_dataset.set_transform(lazy_preprocess)

    # Load tools for reward functions
    reward_fn = RewardComputer()

    print("All judging servers connected.")
    os.environ["WANDB_PROJECT"] = "ct-chat-grpo"
    run_name = "1702-f1class-f1llm" + datetime.now().strftime(
        "_%Y%m%d_%H%M%S"
    )  # '1002-grpo-fullrun-bs6r8-f1-classifier-nollm-05green'

    # Define training arguments
    transformers.utils.logging.set_verbosity_info()

    training_args = GRPOConfig(
        output_dir=f"./{run_name}",
        learning_rate=1e-5,
        num_generations=8,  # How many responses to sample per scan
        per_device_train_batch_size=3 * 8,  # Adjust based on your GPU memory
        bf16=True,
        max_completion_length=512,
        # save_only_model=True,
        logging_steps=1,
        save_steps=1,
        temperature=1.0,
        max_steps=1000,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = CustomGRPOTrainer(
        model=GRPO_CTChat_Wrapper(chat_instance),
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=EmbeddingProcessor(chat_instance.tokenizer),
    )
    # print(trainer.eos_token_id)

    # Launch training
    trainer.train(
        resume_from_checkpoint="/capstor/store/cscs/swissai/a135/wp3-agents/workspace/3dragent-agent-mel/radagent/toolbox_src/CT_CHAT_main/1702-f1class-f1llm_20260224_095317/checkpoint-165"
    )
    # resume_from_checkpoint='/capstor/store/cscs/swissai/a135/wp3-agents/workspace/3dragent-agent-mel/radagent/toolbox_src/CT_CHAT_main/grpo-fullrun-bs6r8-n_20260121_092504/checkpoint-180'
