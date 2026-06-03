import json
import asyncio
import os
from pathlib import Path
import time
from typing import List

from openai import AsyncOpenAI
import weave

from agents.server_manager import MultiServerManager
from evaluation.compute_metrics import compute_nlp_metrics
from constants_and_path_utils import OUTPUTS
from utils import clean_and_convert_to_json
from tools.tool_configs import ALL_TOOL_CONFIGS, JUDGING_TOOLS, SERVERS
import art
from art.local import LocalBackend
from agents.art_dataset import MAX_TURNS
from datetime import datetime

from agents.tool_inspection_utils import (
    compute_manual_num_valid_tool_called,
    get_tool_tree,
)


MAX_SEQ_LENGTH = 26624
GPU_MEMORY_UTILIZATION = 0.65
SELF_REFLECT_EVERY_N_TURNS = 12


def tprint(*args, **kwargs):
    """Print with timestamp prefix [DD/MM HH:mm:ss]"""
    timestamp = datetime.now().strftime("[%d/%m %H:%M:%S]")
    print(timestamp, *args, **kwargs)


class BaseOrchestrator:
    def __init__(
        self,
        base_model_name: str,
        subfolder_name: str,
        project_name: str,
        task="report_generation",
        do_tool_augmentation=False,
        for_inference_only=False,
        lambda_tool_success=0.5,
        lambda_llm_judge_f1=1.0,
        lambda_llm_judge_prec_recall=1.0,
        lambda_green=1.0,
        lambda_trajectory_judge=0.0,
        lambda_manual_tool_judge=0.0,
        lambda_f1_text_classifier=0.0,
        self_reflect=False,
        check_summary=False,
        r=16,
    ):
        self.task = task
        self.do_tool_augmentation = do_tool_augmentation
        self.for_inference_only = for_inference_only
        self.base_model_name = base_model_name
        self.subfolder_name = subfolder_name
        self.is_gpt_model = "gpt" in self.base_model_name
        if self.is_gpt_model:
            assert self.for_inference_only, "GPT models can only be used for inference."
        if self.is_gpt_model:
            self.full_model_name = self.base_model_name
        elif self.for_inference_only:
            self.date_time = datetime.now().strftime("_%Y%m%d_%H%M%S")
            self.full_model_name = (
                self.base_model_name.split("/")[-1] + "_inference" + self.date_time
            )
        else:
            self.full_model_name = self.subfolder_name
        self.project_name = project_name
        self.lambda_tool_success = lambda_tool_success
        self.lambda_llm_judge_f1 = lambda_llm_judge_f1
        self.lambda_f1_text_classifier = lambda_f1_text_classifier
        self.lambda_green = lambda_green
        self.lambda_trajectory_judge = lambda_trajectory_judge
        self.lambda_llm_judge_prec_recall = lambda_llm_judge_prec_recall
        self.lambda_manual_tool_judge = lambda_manual_tool_judge
        self.convert_tool_messages_manually = base_model_name in [
            "google/gemma-3-27b-it",
            "gpt-4o-2024-08-06",
        ]
        # the server manager needs to be set on all nodes.
        self.set_server_manager()
        self.verbose = True
        self.self_reflect = self_reflect
        self.check_summary = check_summary
        self.r = r

    @property
    def guidelines(self) -> str:
        raise NotImplementedError("Subclasses must implement guidelines property")

    async def initialize_model(self):
        is_main_node = int(os.environ.get("LOCAL_NODE_RANK", 0)) == 0

        if is_main_node:
            if self.is_gpt_model:
                self.client = AsyncOpenAI()
            else:
                self.backend = LocalBackend(
                    in_process=True,
                    path=OUTPUTS,
                )
                self.model = art.TrainableModel(
                    name=self.full_model_name,
                    project=self.project_name,
                    base_model=self.base_model_name,
                )
                self.model._internal_config = art.dev.InternalModelConfig(
                    init_args=art.dev.InitArgs(
                        max_seq_length=MAX_SEQ_LENGTH,
                    ),
                    peft_args=art.dev.PeftArgs(
                        r=self.r,
                        lora_alpha=self.r * 2,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    ),
                    engine_args=art.dev.EngineArgs(
                        enforce_eager=True,
                        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                        tensor_parallel_size=1,
                        max_lora_rank=16,
                        enable_chunked_prefill=True,
                        max_num_batched_tokens=MAX_SEQ_LENGTH,
                    ),
                )

                asyncio.run(self.model.register(self.backend))
                self.client = self.model.openai_client()

            if self.for_inference_only:
                if self.subfolder_name == 'default':
                    self.output_dir = (
                        OUTPUTS
                        / self.project_name
                        / "models"
                        / self.base_model_name
                        / f"inference_{self.date_time}"
                    )
                else:
                    self.output_dir = (
                        OUTPUTS
                        / self.project_name
                        / "models"
                        / self.base_model_name
                        / self.subfolder_name
                    )
            else:
                self.output_dir = (
                    OUTPUTS
                    / self.project_name
                    / "models"
                    / self.subfolder_name
                    / "train"
                )

            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "trajectory").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "fails").mkdir(parents=True, exist_ok=True)

    def list_tool_descriptions(self) -> List:
        """List of tool descriptions"""
        tools_list = []
        for tool in ALL_TOOL_CONFIGS:
            if tool["name"] in self.tools_to_use:
                tools_list.append(tool)
        return tools_list

    @property
    def tools_to_use(self) -> list:
        "List of tool names. By default use all tools"
        all_tools = list(SERVERS.keys())
        for t in JUDGING_TOOLS:
            if t in all_tools:
                all_tools.remove(t)
        return all_tools

    def format_system_prompt(self, random_tool_gen) -> str:
        raise NotImplementedError(
            "Subclasses must implement format_system_prompt property"
        )

    def _create_tool_selection_prompt(self, random_tool_gen) -> str:
        """Create tool selection prompt"""
        agent_tools = []
        i = 0
        TOOL_SUITABLE_FOR_AUGMENTATION = ["ct_vqa_tool", "disease_classifier_tool"]
        for tool in ALL_TOOL_CONFIGS:
            if (tool["name"] in self.tools_to_use) and (
                not tool["name"] in JUDGING_TOOLS
            ):
                # Unused at the moment
                if (
                    (self.do_tool_augmentation)
                    and (tool["name"] in TOOL_SUITABLE_FOR_AUGMENTATION)
                    and (random_tool_gen[i] < 0.2)
                ):
                    i += 1
                    continue
                agent_tools.append(tool)
                i += 1
        all_tools = json.dumps(agent_tools, ensure_ascii=False, indent=2)

        prompt = f"""
{all_tools}

"""
        return prompt

    def set_server_manager(self):
        self.servers = {}
        for t in self.tools_to_use:
            if t not in SERVERS:
                raise ValueError(f"Tool {t} not found in servers_config")
            self.servers[t] = SERVERS[t]
        for t in JUDGING_TOOLS:
            if t in SERVERS:
                self.servers[t] = SERVERS[t]
        self.server_manager = MultiServerManager(servers=self.servers)

    async def connect_to_servers(self):
        """Connects to all FastMCP servers."""
        await self.server_manager.connect_all()
        tprint("✅ Connection successful!", flush=True)

    async def close_server_connections(self):
        """Closes all FastMCP server connections."""
        await self.server_manager.close_all()
        tprint("👋 All connections closed")

    def _save_conversation_trace(
        self, image_path, traj: art.Trajectory, file_suffix: str = ""
    ) -> None:
        """Save the conversation messages to a JSON file."""
        messages = traj.messages()
        log_message = {"reward": traj.reward}
        for k in traj.metrics.keys():
            log_message[k] = traj.metrics[k]
        messages.append(log_message)
        try:
            filename = str(
                self.output_dir
                / "trajectory"
                / (
                    Path(
                        str(image_path)
                        .replace(".nii", "_trajectory")
                        .replace(".gz", "")
                    ).name
                    + str(file_suffix)
                    + ".json"
                )
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tprint(f"Warning: Could not save conversation trace: {str(e)}")

    def _save_fails(
        self, image_path, traj: art.Trajectory, file_suffix: str = ""
    ) -> None:
        """Save the conversation messages to a JSON file."""
        messages = traj.messages()
        try:
            filename = str(
                self.output_dir
                / "fails"
                / (
                    Path(
                        str(image_path)
                        .replace(".nii", "_trajectory")
                        .replace(".gz", "")
                    ).name
                    + file_suffix
                    + ".json"
                )
            )
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tprint(f"Warning: Could not save conversation trace: {str(e)}")

    async def get_next_agent_answer(self, traj):
        # Get LLM response
        # Not all LLM support "tool" role, convert "tool" messages to "user" messages
        # This is only used when doing inference with GPT models or older models that do not support tool messages.
        converted_trajectory_messages = traj.messages().copy()
        if self.convert_tool_messages_manually:
            for msg in converted_trajectory_messages:
                if msg.get("role", "") == "tool":
                    msg["role"] = "user"

        async with traj.track_duration("llm_completion"):
            response = await self.client.chat.completions.create(
                model=self.full_model_name,
                messages=converted_trajectory_messages,
                max_completion_tokens=4096,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
                temperature=1.0,  # if not self.for_inference_only else 0.1,
            )

        choice = response.choices[0]
        msg = choice.message.content

        traj.messages_and_choices.append(choice)
        decision = clean_and_convert_to_json(msg)

        return traj, decision

    async def tool_call(self, traj, decision):
        # Call the tool, the server name is the same as the tool name
        is_successful_call = False
        try:
            async with traj.track_duration(
                f"{decision['tool_name']}_completion"
            ):
                result = await self.server_manager.call_tool(
                    decision["tool_name"],
                    decision["tool_name"],
                    decision["arguments"],
                )
        except Exception as ve:
            if self.verbose:
                print(f"❌ Tool call error: {ve}", flush=True)
            result = None

        if result is None:
            result = "Tool call failed"
            traj.messages_and_choices.append(
                {"role": "tool", "content": result}
            )
        else:
            traj.messages_and_choices.append(
                {"role": "tool", "content": str(result)}
            )
            if "error" not in str(result).lower():
                is_successful_call = True

        return traj, is_successful_call

    async def process_agent_answer(self, traj, decision, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, num_turns, file_suffix):
        task_failed = False
        task_completed = False
        if decision is None:
            if self.verbose:
                tprint('❌ Failed to parse LLM response as JSON. Please ensure the response is in the correct format.', flush=True)
            traj.messages_and_choices.append(
                {
                    "role": "user",
                    "content": "The previous response could not be parsed as JSON. Ensure that your response follows the specified format and is in a valid JSON format.",
                }
            )
            num_failed_formatting += 1
            # This should count as a failed tool call
            n_tool_calls += 1
            if num_failed_formatting >= 3:
                task_failed = True
                return traj, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, task_completed, task_failed
        else:
            num_failed_formatting = 0
            try:
                if decision.get("action") == "call_tool":
                    if self.verbose:
                        tprint('Tool call in progress...', flush=True)
                    start_tool = time.perf_counter()
                    badly_formatted = True
                    if decision.get("preliminary_findings", None) is not None:
                        if num_turns <= 2:
                            badly_formatted = False
                        elif (
                            isinstance(decision["preliminary_findings"], List)
                            and len(decision["preliminary_findings"]) > 0
                        ):
                            badly_formatted = False
                    n_tool_calls += 1
                    traj, is_successful_call = await self.tool_call(traj, decision)
                    if is_successful_call and not badly_formatted:
                        n_success_tool_calls += 1
                    duration = time.perf_counter() - start_tool
                    # traj.metrics["total_tool_call_duration"] += duration

                elif decision.get("action") == "final_answer":

                    # At the moment we don't really use any of this

                    if self.self_reflect and (
                        n_turn_since_last_self_reflect > MAX_TURNS
                        or never_called_traj_judge
                    ):
                                            
                        # Inject self-reflection or summary check before confirming final answer
                        # User: Before finishing let's self reflect on your analysis. Add tool call to judge. Based on the feedback from the trajectory judge, decide whether
                        # you want to confirm your final answer or continue the analysis by calling more tools.
                        trajectory_judge = await self.server_manager.call_tool(
                            "report_judge_tool",
                            "trajectory_judge_tool",
                            {
                                "conversation_trajectory": traj.messages(),
                            },
                        )
                        user_message = f"""Time to pause and self-reflect on your current strategy.
                        
                        # Judging Feedback:
                        {trajectory_judge}

                        Based on the feedback from the trajectory judge, decide whether you want to confirm your final answer action or continue the analysis by calling more tools.
                        """

                        traj.messages_and_choices.append(
                            {"role": "user", "content": user_message}
                        )
                        n_turn_since_last_self_reflect = 0
                        never_called_traj_judge = False
                    elif self.check_summary and not final_summary_has_passed:
                        summary_judge = await self.server_manager.call_tool(
                            "report_judge_tool",
                            "summary_judge_tool",
                            {
                                "conversation_trajectory": traj.messages(),
                            },
                        )
                        user_message = f"""Before finalizing your analysis, let's double check whether you summarised all the findings well.
                            
                            # Judging Feedback:
                            {summary_judge}

                            Based on this feedback, decide whether you want to confirm your final answer or update your final report.
                            You should NOT be calling more tools at this stage, but you can update your final answer based on the judge's feedback.
                            """
                        traj.messages_and_choices.append(
                            {"role": "user", "content": user_message}
                        )
                        final_summary_has_passed = True
                    
                    # Only path used at the moment!
                    else:
                        task_completed = True
                        traj.metrics["task_completed"] = True

                else:
                    traj.messages_and_choices.append(
                        {
                            "role": "user",
                            "content": "Unrecognised action type. Please ensure the response follows the specified format.",
                        }
                    )

            except Exception as e:
                # log the exact error with traceback in the messages for easier debugging of tool call issues
                import traceback
                error_message = f"Error processing agent answer: {str(e)}\n{traceback.format_exc()}"
                if self.verbose:
                    tprint(error_message, flush=True)
                task_failed = True
                # self._save_fails(
                #     scenario["image_path"], traj, file_suffix=file_suffix
                # )
        return traj, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, task_completed, task_failed

    @weave.op
    async def rollout(
        self,
        scenario,
        file_suffix: str = "",
        compute_reward=False,
        random_tool_gen=None,
    ) -> str:
        """Process complex queries that may require multiple tool calls"""
        start_inf = time.perf_counter()
        if self.verbose:
            tprint(f"🤖 Sending query to agent: {scenario['task']}")

        # Create prompt

        traj = art.Trajectory(
            messages_and_choices=[],
            reward=0,
            metadata={"task": scenario["task"]},
            metrics={
                "task_completed": False,
                "success": False,
                "ran_out_of_turns": False,
                "n_tool_calls": 0,
                "n_success_tool_calls": 0,
                "tool_call_success_rate": 0.0,
                "green": 0,
                "llm_judge_f1": 0,
                "vqa_correct": 0,
                "rollout_duration": 0.0,
            },
            scenario=scenario,
        )
        traj.metrics["total_tool_call_duration"] = 0.0
        traj.messages_and_choices = [
            {"role": "system", "content": self.format_system_prompt(random_tool_gen)},
            {
                "role": "user",
                "content": f"{scenario['task']}",
            },
        ]

        num_turns = 0
        num_failed_formatting = 0
        task_completed = False
        traj.metrics["task_completed"] = False
        n_tool_calls = 0
        n_success_tool_calls = 0
        final_summary_has_passed = False
        n_turn_since_last_self_reflect = 0
        never_called_traj_judge = True
        try:
            while num_turns < MAX_TURNS and not task_completed:
                num_turns += 1
                if self.verbose:
                    tprint(
                        f"🔁 Scenario {Path(scenario['image_path']).stem}, {file_suffix}, step {num_turns} in progress..."
                    )
                try:
                    traj, decision = await self.get_next_agent_answer(traj)
                except Exception as e:
                    self._save_fails(
                        scenario["image_path"], traj, file_suffix=file_suffix
                    )
                    break

                traj, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, task_completed, task_failed = await self.process_agent_answer(traj, decision, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, num_turns, file_suffix)

                if task_failed:
                    if self.verbose:
                        tprint(
                            f"❌ Scenario {Path(scenario['image_path']).stem}, {file_suffix}, Too many formatting failures, ending the task."
                        )
                    self._save_fails(
                        scenario["image_path"], traj, file_suffix=file_suffix
                    )
                    break
                
            if not task_completed and num_turns == MAX_TURNS:
                traj.metrics["ran_out_of_turns"] = True

            if compute_reward and task_completed:
                if self.verbose:
                    tprint(
                        f"🏆 Scenario {Path(scenario['image_path']).stem}, {file_suffix}, computing reward...",
                        flush=True,
                    )
                async with traj.track_duration("reward_computation"):
                    # lambda_f1_text_classifier
                    tasks = []
                    task_names = []
                    reward_dict = {}
                    if self.task == "report_generation":
                        if (
                            self.lambda_llm_judge_f1 > 0.0
                            or self.lambda_llm_judge_prec_recall > 0.0
                        ):
                            llm_as_judge_task = self.server_manager.call_tool(
                                "report_judge_tool",
                                "report_judge_tool",
                                {
                                    "candidate_report": decision.get("answer", ""),
                                    "ground_truth_report": scenario.get("gt", ""),
                                },
                            )
                            tasks.append(llm_as_judge_task)
                            task_names.append("llm_judge")
                        if self.lambda_green > 0.0:
                            green_task = self.server_manager.call_tool(
                                "green_tool",
                                "green_tool",
                                {
                                    "candidate_report": decision.get("answer", ""),
                                    "ground_truth_report": scenario.get("gt", ""),
                                },
                            )
                            tasks.append(green_task)
                            task_names.append("green_score")
                        if self.lambda_trajectory_judge > 0.0:
                            trajectory_judge_task = self.server_manager.call_tool(
                                "report_judge_tool",
                                "trajectory_judge_tool",
                                {
                                    "conversation_trajectory": traj.messages(),
                                },
                            )
                            tasks.append(trajectory_judge_task)
                            task_names.append("trajectory_judge")

                        f1_task = self.server_manager.call_tool(
                            "f1_text_classifier_tool",
                            "f1_text_classifier_tool",
                            {
                                "candidate_report": decision.get("answer", ""),
                                "ground_truth_report": scenario.get("gt", ""),
                            },
                        )
                        tasks.append(f1_task)
                        task_names.append("f1_text_classifier")

                        all_rewards = await asyncio.gather(*tasks)
                        reward_dict = {
                            name: reward
                            for name, reward in zip(task_names, all_rewards)
                        }

                        llm_as_judge_reward = reward_dict.get("llm_judge", None)
                        try:
                            llm_reward_dict = json.loads(llm_as_judge_reward)
                        except Exception as e:
                            tprint(f"Error parsing LLM judge reward: {e}", flush=True)
                            llm_reward_dict = {}
                        f1_llm = llm_reward_dict.get("abnormal_f1", 0.0)
                        mean_prec_rec_llm = llm_reward_dict.get(
                            "abnormal_mean_rec_prec", 0.0
                        )

                        green_reward = reward_dict.get("green_score", None)
                        try:
                            green_reward = float(green_reward)
                        except:
                            green_reward = 0.0

                        f1_classifier = reward_dict.get("f1_text_classifier", None)
                        try:
                            f1_classifier = float(f1_classifier)
                        except:
                            f1_classifier = 0.0

                        traj.reward = (
                            self.lambda_llm_judge_f1 * f1_llm
                            + self.lambda_llm_judge_prec_recall * mean_prec_rec_llm
                            + self.lambda_green * green_reward
                            + self.lambda_f1_text_classifier * f1_classifier
                        )
                        traj.metrics["green"] = green_reward
                        traj.metrics["llm_judge_f1"] = f1_llm
                        traj.metrics["llm_judge_avg_prec_rec"] = mean_prec_rec_llm
                        traj.metrics["f1_text_classifier"] = f1_classifier

                    elif self.task == "vqa":
                        total_reward = 0.0
                        if task_completed:
                            gt_answer = scenario.get("gt", "").strip().lower()
                            final_message = (
                                traj.messages_and_choices[-1]
                                .message.content.strip()
                                .lower()
                            )
                            decision = clean_and_convert_to_json(final_message)
                            predicted_answer = (
                                decision.get("answer", "").strip().lower()
                            )
                            nlp_metrics = compute_nlp_metrics(
                                {scenario["task_id"]: gt_answer},
                                {scenario["task_id"]: predicted_answer},
                            )[1]
                            if predicted_answer == gt_answer:
                                total_reward += 1
                            total_reward += nlp_metrics["BLEU_1"][0]
                            total_reward += nlp_metrics["ROUGE_L"][0]
                            traj.metrics["vqa_correct"] = int(
                                predicted_answer == gt_answer
                            )
                        traj.reward = total_reward

                    if self.lambda_trajectory_judge > 0.0:
                        trajectory_judge_reward = reward_dict.get(
                            "trajectory_judge", None
                        )
                        trajectory_judge_reward_dict = json.loads(
                            trajectory_judge_reward
                        )
                        checklist_adherence_score = trajectory_judge_reward_dict.get(
                            "checklist adherence", {}
                        ).get("score", 2.5)
                        tool_seq_coherence_score = trajectory_judge_reward_dict.get(
                            "tool sequence coherence", {}
                        ).get("score", 2.5)
                        traj.metrics["checklist_adherence_reward"] = (
                            checklist_adherence_score
                        )
                        traj.metrics["tool_seq_coherence_reward"] = (
                            tool_seq_coherence_score
                        )

                        total_traj_score = (
                            checklist_adherence_score / 5.0
                            + tool_seq_coherence_score / 5.0
                        )

                        traj.metrics["trajectory_judge_normalized_reward"] = (
                            total_traj_score
                        )
                        traj.reward += self.lambda_trajectory_judge * total_traj_score

                    tool_tree = get_tool_tree(traj.messages())
                    n_unused_files = len(tool_tree["unused_files"])
                    n_total_files = len(tool_tree["critical_path"]) + n_unused_files
                    if n_total_files > 0:
                        manual_tool_judge_score = 1.0 - (n_unused_files / n_total_files)
                    else:
                        manual_tool_judge_score = 0.0
                    traj.metrics["manual_tool_coherence_score"] = (
                        manual_tool_judge_score
                    )

                    num_tool_called = compute_manual_num_valid_tool_called(
                        traj.messages(), self.tools_to_use
                    )
                    traj.metrics["manual_num_tool_called"] = num_tool_called
                    diversity_score_manual = num_tool_called / len(self.tools_to_use)
                    traj.metrics["manual_tool_diversity"] = diversity_score_manual

                    if self.lambda_manual_tool_judge > 0.0:
                        traj.reward += self.lambda_manual_tool_judge * (
                            manual_tool_judge_score + diversity_score_manual
                        )

                    traj.reward += (
                        self.lambda_tool_success * (n_success_tool_calls / n_tool_calls)
                        if n_tool_calls > 0
                        else 0.0
                    )

            traj.metrics["num_turns"] = num_turns
            traj.metrics["n_tool_calls"] = n_tool_calls
            traj.metrics["n_success_tool_calls"] = n_success_tool_calls
            traj.metrics["tool_call_success_rate"] = (
                (n_success_tool_calls / n_tool_calls) if n_tool_calls > 0 else 0.0
            )
            duration_inf = time.perf_counter() - start_inf
            traj.metrics["rollout_duration"] = duration_inf
            if task_completed:
                self._save_conversation_trace(
                    scenario["image_path"], traj, file_suffix=file_suffix
                )
            else:
                self._save_fails(scenario["image_path"], traj, file_suffix=file_suffix)
            tprint(
                f"✅ DONE - Scenario {scenario['image_path']}, {file_suffix}",
                flush=True,
            )
            return traj
        except Exception as e:
            self._save_fails(scenario["image_path"], traj, file_suffix=file_suffix)
            return traj
        

    async def partial_inference_rollout(
        self,
        volume_name,
        traj,
        query="",
        file_suffix: str = "",
        stop_after_n_turns = 5,
    ) -> str:
        '''
        To continue a trajectory from a given point, for example to test the agent's response to a specific query or trace perturbation.
        '''
        # Create prompt
        if traj is None:
            traj = art.Trajectory(
                messages_and_choices=[],
                reward=0,
            )
            traj.messages_and_choices = [
                {"role": "system", "content": self.format_system_prompt(None)},
                {
                    "role": "user",
                    "content": query,
                },
            ]

        num_turns = 0
        num_failed_formatting = 0
        task_completed = False
        # These below are not needed for inference
        n_tool_calls = 0
        n_success_tool_calls = 0
        final_summary_has_passed = False
        n_turn_since_last_self_reflect = 0
        never_called_traj_judge = True
        try:
            while num_turns < stop_after_n_turns and not task_completed:
                num_turns += 1
                traj, decision = await self.get_next_agent_answer(traj)
                traj, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, task_completed, task_failed = await self.process_agent_answer(traj, decision, num_failed_formatting, n_turn_since_last_self_reflect, never_called_traj_judge, n_tool_calls, n_success_tool_calls, final_summary_has_passed, num_turns, file_suffix)
                print(volume_name, f'{num_turns / stop_after_n_turns}', task_failed, task_completed, flush=True)
                if task_failed:
                    break
            traj.metrics['completed'] = task_completed
            if task_completed:
                self._save_conversation_trace(
                    volume_name, traj, file_suffix=file_suffix
                )
            else:
                self._save_fails(volume_name, traj, file_suffix=file_suffix)
            return traj
        except Exception as e:
            traj.metrics['completed'] = False
            self._save_fails(volume_name, traj, file_suffix=file_suffix)
            return traj
