import re
import json

import os
import sys


def kill_job():
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        print(f"Aborting Slurm job {job_id}...")
        # Flush stdout to ensure the message is logged before the kill
        sys.stdout.flush()
        os.system(f"scancel {job_id}")
    else:
        print("Not running under Slurm.")


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
            cleaned_string = re.sub(
                r"//.*", "", cleaned_string
            )  # Remove single-line comments
            cleaned_string = re.sub(
                r"/\*.*?\*/", "", cleaned_string, flags=re.DOTALL
            )  # Remove multi-line comments

            # Normalize string format
            cleaned_string = re.sub(
                r",\s*}", "}", cleaned_string
            )  # Fix trailing commas in objects
            cleaned_string = re.sub(
                r",\s*]", "]", cleaned_string
            )  # Fix trailing commas in arrays

            try:
                json_obj = json.loads(cleaned_string)
                return json_obj
            except:  # noqa
                if verbose:
                    print("All automated methods failed, returning None")
                return None
