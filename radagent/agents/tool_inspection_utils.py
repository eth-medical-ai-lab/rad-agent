import json
from typing import Dict, List
from collections import defaultdict, deque
import re

from constants_and_path_utils import CAPSTOR_ROOT


def extract_storage_root_paths(text: str) -> List[str]:
    """
    Extracts file paths starting with the configured storage root from a string.

    Handles:
    - Python lists (removes quotes, brackets, commas).
    - Natural language (removes trailing periods, question marks).

    Args:
        text (str): The input string containing paths.

    Returns:
        List[str]: A list of clean file paths found in the text.
    """
    # Regex explanation:
    # 1. CAPSTOR_ROOT       -> Matches the configured storage root
    # 2. [^\s,\]'"?!;:\)]+     -> Matches path chars, stopping at whitespace, quotes, or punctuation
    # 3. (?<!\.)               -> Negative lookbehind: ensures the match doesn't end in a dot
    storage_root = re.escape(str(CAPSTOR_ROOT).rstrip("/"))
    pattern = rf"{storage_root}/[^\s,\]'\"?!;:\)]+(?<!\.)"

    matches = re.findall(pattern, text)
    if isinstance(matches, str):
        matches = [matches]
    return matches


def compute_manual_num_valid_tool_called(messages: List[dict], valid_tools) -> float:
    tool_called = set()
    for message in messages:
        if isinstance(message, Dict):
            try:
                if message["role"] == "assistant":
                    response = json.loads(message["content"])
                    if response.get("action", None) == "call_tool":
                        last_tool_called = response["tool_name"]
                        if last_tool_called in valid_tools:
                            tool_called.add(last_tool_called)
            except:  # noqa
                continue
    return len(tool_called)


def get_tool_tree(data: List[dict]):
    last_tool_called = None
    input_output_dict = []
    for message in data:
        if isinstance(message, Dict):
            try:
                if message["role"] == "assistant":
                    response = json.loads(message["content"])
                    if response.get("action", None) == "call_tool":
                        last_tool_called = response["tool_name"]
                        if response.get("arguments", None) is not None:
                            input_file_paths = extract_storage_root_paths(
                                json.dumps(response["arguments"])
                            )
                            last_tool_input = input_file_paths
                if message["role"] == "tool":
                    if (
                        message["content"] == "Tool call failed"
                        or ("ERROR" in message["content"].upper())
                        or last_tool_called is None
                    ):
                        continue
                    else:
                        last_tool_called = None
                        response = message["content"]
                        # case where the tool outputs a file path
                        if f"{CAPSTOR_ROOT}/" in response:
                            file_paths = extract_storage_root_paths(response)
                            input_output_dict.append(
                                {i: file_paths for i in last_tool_input}
                            )
                        else:
                            input_output_dict.append(
                                {i: "LEAF" for i in last_tool_input}
                            )
            except:  # noqa
                continue
    return analyze_dependencies(input_output_dict)


def analyze_dependencies(trace):
    # 1. Build Graph (Handling lists and single strings)
    graph_backward = defaultdict(set)
    all_nodes = set()
    leaf_nodes = set()

    for step in trace:
        for inp, output_data in step.items():
            all_nodes.add(inp)

            # valid_outputs will be a list of filenames
            valid_outputs = []

            # Handle "LEAF" string directly
            if output_data == "LEAF":
                leaf_nodes.add(inp)
                continue

            # Handle list of files vs single file string
            if isinstance(output_data, list):
                valid_outputs = output_data
            else:
                valid_outputs = [output_data]

            for out in valid_outputs:
                if out == "LEAF":
                    leaf_nodes.add(inp)
                else:
                    all_nodes.add(out)
                    # Record dependency: Child depends on Parent (Parent -> Child)
                    # For backtracking, we store: Child -> {Parents}
                    graph_backward[out].add(inp)

    # 2. Backtrack from LEAF nodes to find Ancestors (Critical Path)
    critical_path = set()
    queue = deque(leaf_nodes)

    # Initialize critical path with the leaves themselves
    for node in leaf_nodes:
        critical_path.add(node)

    while queue:
        current_node = queue.popleft()

        # specific check: if this node has parents, add them
        if current_node in graph_backward:
            parents = graph_backward[current_node]
            for parent in parents:
                if parent not in critical_path:
                    critical_path.add(parent)
                    queue.append(parent)

    # 3. Identify Unused Files
    unused_files = all_nodes - critical_path

    return {"critical_path": list(critical_path), "unused_files": list(unused_files)}
