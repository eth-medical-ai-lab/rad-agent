from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

from radagent.tools.tool_configs import build_tool_selection_prompt


ART_DEFAULT_PROMPT = (
    "<image>\n"
    "Would you mind generating the radiology report for the specified chest CT "
    "scan?<report_generation>"
)
ART_MAX_STEPS = 60
ART_TOOL_NAMES = [
    "ct_vqa_tool",
    "report_generation_tool",
    "disease_classifier_tool",
    "effusion_segmentation_tool",
    "anatomy_segmentation_tool",
    "biggest_slice_selection_tool",
    "windowing_tool",
    "get_several_slices_from_segmentation",
    "slice_vqa_tool",
    "extract_slices_from_ct",
]
JSON_RETRY_MESSAGE = (
    "The previous response could not be parsed as JSON. Ensure that your response "
    "follows the specified format and is in a valid JSON format."
)
UNKNOWN_ACTION_MESSAGE = (
    "Unrecognised action type. Please ensure the response follows the specified format."
)


CHECKLIST = """1. Check airways: in particular trachea (position, caliber, wall thickness), carina, main bronchi, bronchial thickening, bronchiectasis, bronchiolitis, mucoid impaction etc.
2. Lung parenchyma assessment: check for nodules and masses, focal abnormalities, assess presence of diffuse patterns (ground-glass opacities, consolidation, reticular, nodular, etc)
3. Pleural assessment: check for effusion (location, severity, associated findings), pneumothorax (approximate size, tension signs), pleural thickening (smooth vs. nodular, calcification, enhancement pattern)
4. Heart: check pericardium (effusion, thickening, calcification), coronary arteries, cardiac chambers
5. Cardiovascular & mediastinum: check aorta, atherosclerosis, pulmonary arteries (diameter of pulmonary trunk, patency if contrast-enhanced), and mediastinum (e.g. lymph nodes, thymus, esophagus, thyroid)
6. Diaphragm & upper abdominal organs: diaphgram (position, defects, hernias), liver, adrenals, spleen, kidneys, pancreas, stomach. Note any abnormalities, focal lesions, masses, thickening etc.
7. Spine, ribs, sternum, sternum & clavicles: check fractures, lesions, facet arthropathy, canal stenosis etc.
8. Check chest wall, breasts, axillae, look for muscle asymmetry or masses, subcutaneous emphysema, nodules, edema etc.
9. Check for presence of devices like catheters, tubes, lines, pacemakers, surgical clips etc. and note their position and any complications.
"""


def clean_and_convert_to_json(input_string, verbose=False):
    """
    Convert a string to valid JSON format
    """

    # Attempt direct parsing
    if input_string.startswith("```json") and input_string.endswith("```"):
        input_string = re.sub(r"```json\s*|\s*```", "", input_string.strip())
    try:
        json_obj = json.loads(input_string)
        return json_obj
    except:  # noqa
        # If direct parsing fails, attempt to clean the string
        if verbose:
            print(
                "Unable to convert directly to JSON, starting string cleanup..., str is:",
                input_string,
            )
        try:
            import ast

            json_obj = ast.literal_eval(input_string)
            assert isinstance(json_obj, dict) or isinstance(json_obj, list)
            return json_obj
        except:  # noqa
            if verbose:
                print("ast.literal_eval failed, trying alternative methods...")
            pass
        # Method 1: Attempt to extract content within outermost braces
        brace_match = re.search(r"\{.*\}", input_string, re.DOTALL)
        if brace_match:
            cleaned_string = brace_match.group(0)
            try:
                json_obj = json.loads(cleaned_string)
                return json_obj
            except:  # noqa
                if verbose:
                    print("Method 1 failed, trying Method 2...")
                pass
        # Method 2: Remove potential problematic characters and extra content
        # Remove leading ```json and trailing ```
        cleaned_string = re.sub(r"^```json\s*|\s*```$", "", input_string.strip())

        try:
            json_obj = json.loads(cleaned_string)
            return json_obj
        except:  # noqa
            if verbose:
                print("Attempting Method 3: Manual JSON construction...")

            # Method 3: Apply more comprehensive cleaning
            # Remove all comments and extra whitespace/newlines
            cleaned_string = re.sub(r"//.*", "", cleaned_string)
            cleaned_string = re.sub(r"/\*.*?\*/", "", cleaned_string, flags=re.DOTALL)
            cleaned_string = re.sub(r",\s*}", "}", cleaned_string)
            cleaned_string = re.sub(r",\s*]", "]", cleaned_string)

            try:
                json_obj = json.loads(cleaned_string)
                return json_obj
            except:  # noqa
                if verbose:
                    print("All automated methods failed, returning None")
                return None


def serialize_tool(tool: Any) -> dict[str, Any]:
    payload = tool.model_dump() if hasattr(tool, "model_dump") else {}
    payload.setdefault("name", getattr(tool, "name", "unknown"))
    payload.setdefault("description", getattr(tool, "description", "") or "")
    payload.setdefault("inputSchema", getattr(tool, "inputSchema", {}))
    return {
        "name": payload["name"],
        "description": payload["description"],
        "input_schema": payload["inputSchema"],
    }


def build_v8c_system_prompt(tool_prompt: str) -> str:
    return f"""
# GENERAL INSTRUCTIONS
You are an AI radiologist that can use different tools for answering questions about the provided CT image, diagnosing diseases or generating a complete CT report.

## Available tools:
{tool_prompt}

WARNING: individual tool may make mistakes, so when possible, double check your findings using multiple tools.

ALWAYS start by outlining your analysis plan, specifying which tools you intend to use for which purpose, in which order, before proceeding with the analysis. You may revisit and revise your plan as needed based on the information you gather during your analysis.

IMPORTANT: At each turn of the conversation, you will decide which action to take next. You can:
1. Call a tool to get more information. To use a tool, respond with a JSON object in this exact format:
{{
   "reasoning": Thought process,
   "preliminary_findings": "list of medical findings based on all the information you have gathered so far, if any",
    "action": "call_tool",
    "tool_name": "tool_name",
    "arguments": {{"param_name": "param_value"}}
}}
NOTE: "preliminary_findings" is a list of all the medical findings you have gathered so far based on the information you have collected so far. If it contains contradictory findings, make sure to resolve those contradictions using additional tools, the preliminary findings list should NOT contain contradictory findings but reflect the current consensus based on the majority agreement between the different tools you have used so far.

2. If you already have enough information, summarise the LAST "preliminary_findings" list to provide the final answer to the user query, in one paragraph (not a list). IMPORTANT: only summarise the LAST "preliminary_findings" list for your final answer, ignore any previous message. Make sure to provide your final answer in this EXACT format:
{{
    "reasoning": "your final reasoning",
    "preliminary_findings": "list of medical findings based on all the information you have gathered so far, if any",
    "action": "final_answer",
    "answer": "your final answer to the user"
}}

IMPORTANT: At each step, carefully consider which tool or combination of tools will provide the most accurate and comprehensive information for the specific item you are assessing. Feel free to pause, plan your next steps carefully and reflect on your strategy to ensure optimal use of the available tools and ensure the best analysis quality.


# REPORT GENERATION INSTRUCTIONS
If you are asked to generate a CT report, start by using the report_generation_tool to generate a preliminary report based on the CT image. Then, use the diagnosis checklist provided below to check your report. It provides the organs / abnormalities and the specific issues you need to check for the final report. 

## Checklist:
{CHECKLIST}

IMPORTANT: 
 - Make sure each item in the checklist is mentioned in the final report. If not, use the proper tools to check for the presence of any abnormalities related to that item and provide their location if known, and update the report accordingly.
 - For any identified abnormalities identified in the preliminary report, make sure to double check their presence and location using the other tools, and update the report accordingly.
 - For every item, you can use multiple tools sequentially.
 - Be mindful that individual tools may make mistakes. For increased accuracy, use a combination of different tools to DOUBLE CHECK your findings, for example using both a slice-based VQA and a whole CT VQA tool. 
 - If you find any contradictions in the information provided by different tools, make sure to resolve those contradictions using additional tools, and provide the most accurate answer in the final report. ALWAYS find a consensus, do not provide contradictory information in the final report.
 - In the final report, you do NOT need to mention which tools you used to derive which finding, just provide a succint summary of all the relevant medical findings, based on the consensus of all the tools you used.

IMPORTANT: you should VARY the tools you use for different items on the checklist, as some tools may be better suited for detecting abnormalities than others. Each tool will have different strengths and weaknesses, so using a diverse set of tools will help ensure a more comprehensive analysis.

# VISUAL QUESTION ANSWERING INSTRUCTIONS
Consider the precise question asked by the doctor to choose which tools you should use to gather the necessary information to answer the question accurately.
IMPORTANT: if you are provided with multiple answers options, your answer MUST match EXACTELY one of the provided options. Do not add any additional explanation. Do not attempt to generate an answer that is not among the provided options. One of the provided answer options MUST be correct. Do not answer just with the letter or the number of the option, always include the full text of the answer.
Example: with the question "What is the biggest object in this image? (a) A potatoe (b) Tomato (c) Car", you should answer with "(c) Car" and NOT just "c" or just "Car" or "c Car", your answer need to be exactly matching the options.


# FORMATTING INSTRUCTIONS
YOU SHOULD ALWAYS RESPOND IN THE ABOVE JSON FORMAT. Do not include anything outside the JSON object in your response. For example do not include ```json around your answer.
You are already provided with the CT image, you should not ask the doctor to provide you the CT image again. If the tool asks you to provide the image, rephrase your prompt and try again, try another tool or move on to the next item on the checklist. Do not ask the doctor to provide more information, always use the tools to get information you need.
"""


def result_payload(result: Any) -> Any:
    if hasattr(result, "structured_content") and result.structured_content is not None:
        payload = result.structured_content
        if isinstance(payload, dict) and "outputs" in payload:
            return payload["outputs"]
        return payload
    return str(result)


def ensure_required_tools_present(tools: list[Any]) -> None:
    tool_names = {serialize_tool(tool)["name"] for tool in tools}
    missing_tools = [name for name in ART_TOOL_NAMES if name not in tool_names]
    if missing_tools:
        raise RuntimeError(
            "Missing ART tool(s) from the minimal toolbox proxy: "
            + ", ".join(missing_tools)
        )


def build_art_user_prompt(prompt: str, image_path: str | None) -> str:
    user_prompt = prompt if prompt else ART_DEFAULT_PROMPT
    if image_path and "<image>" not in user_prompt:
        user_prompt = f"<image>\n{user_prompt}"
    if image_path and "The image file path is" not in user_prompt:
        suffix = "" if user_prompt.endswith((".", "?", "!")) else "."
        user_prompt = f"{user_prompt}{suffix} The image file path is {image_path}. "
    return user_prompt


def append_message(messages: list[dict[str, str]], role: str, content: str) -> dict[str, str]:
    message = {"role": role, "content": content}
    messages.append(message)
    return message


def serialize_message_content(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return str(payload)


def parse_assistant_decision(message: dict[str, str]) -> dict[str, Any] | None:
    if message.get("role") != "assistant":
        return None
    payload = clean_and_convert_to_json(message.get("content", ""))
    return payload if isinstance(payload, dict) else None


@dataclass(slots=True)
class CollectedRollout:
    answer: str
    messages: list[dict[str, str]]


class AgentRolloutError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.messages = list(messages or [])


async def stream_agent_rollout(
    prompt: str,
    image_path: str | None,
    model_url: str = "http://127.0.0.1:8000/v1",
    model_name: str = "ct-agent",
    api_key: str = "ct-agent",
    toolbox_url: str = "http://127.0.0.1:8080/mcp",
    max_steps: int = ART_MAX_STEPS,
) -> AsyncIterator[list[dict[str, str]]]:
    from fastmcp import Client as MCPClient
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(base_url=model_url, api_key=api_key)

    async with MCPClient(toolbox_url, timeout=3600) as mcp_client:
        all_tools = await mcp_client.list_tools()
        ensure_required_tools_present(all_tools)
        tool_names = set(ART_TOOL_NAMES)
        tool_prompt = build_tool_selection_prompt(ART_TOOL_NAMES)

        user_prompt = build_art_user_prompt(prompt, image_path)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": build_v8c_system_prompt(tool_prompt)},
            {"role": "user", "content": user_prompt},
        ]
        num_failed_formatting = 0
        yield list(messages)

        for _ in range(1, max_steps + 1):
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1.0,
                max_completion_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            content = response.choices[0].message.content or ""
            yield [append_message(messages, "assistant", content)]

            decision = clean_and_convert_to_json(content)
            if decision is None:
                num_failed_formatting += 1
                yield [append_message(messages, "user", JSON_RETRY_MESSAGE)]
                if num_failed_formatting >= 3:
                    raise ValueError("Too many JSON formatting failures")
                continue

            num_failed_formatting = 0
            action = decision.get("action")
            if action == "final_answer":
                return

            if action != "call_tool":
                yield [append_message(messages, "user", UNKNOWN_ACTION_MESSAGE)]
                continue

            tool_name = decision["tool_name"]
            arguments = decision.get("arguments", {})

            payload: Any = "Tool call failed"
            if tool_name in tool_names:
                try:
                    result = await mcp_client.call_tool(tool_name, arguments)
                except Exception:
                    result = None
                if result is not None:
                    payload = result_payload(result)

            yield [append_message(messages, "tool", serialize_message_content(payload))]

    raise RuntimeError("Max steps reached without final answer")


async def collect_agent_rollout(**kwargs: Any) -> CollectedRollout:
    tracked_messages: list[dict[str, str]] = []

    try:
        async for new_messages in stream_agent_rollout(**kwargs):
            tracked_messages.extend(new_messages)
            for message in new_messages:
                decision = parse_assistant_decision(message)
                if decision is None or decision.get("action") != "final_answer":
                    continue
                return CollectedRollout(
                    answer=(decision.get("answer") or "").strip(),
                    messages=list(tracked_messages),
                )
    except Exception as exc:
        raise AgentRolloutError(
            str(exc),
            messages=tracked_messages,
        ) from exc

    raise AgentRolloutError(
        "Rollout completed without a final report.",
        messages=tracked_messages,
    )


def iter_agent_rollout_sync(**kwargs: Any):
    loop = asyncio.new_event_loop()
    generator: Any | None = None
    try:
        asyncio.set_event_loop(loop)
        generator = stream_agent_rollout(**kwargs)
        while True:
            try:
                yield loop.run_until_complete(generator.__anext__())
            except StopAsyncIteration:
                break
    finally:
        if generator is not None:
            try:
                loop.run_until_complete(generator.aclose())
            except (RuntimeError, StopAsyncIteration):
                pass
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            asyncio.set_event_loop(None)
            loop.close()
