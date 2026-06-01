import pandas as pd
import json
import numpy as np
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import argparse
from agents.server_manager import MultiServerManager
from agents.tool_inspection_utils import compute_manual_num_valid_tool_called
from tools.tool_configs import SERVERS

load_dotenv()

local_model = None


async def add_local_score_to_results_df(
    output_file,
    max_idx=None,
):
    server = {
        "report_judge_tool": {
            "script": str(
                Path(__file__).parent.parent / "tools" / "llm_judge_tool_thinking.py"
            ),
            "port": 8010,
            "env": "RadAgentMel",
            "device": "0,1,2,3",
            "target_node": 0,
        }
    }

    server_manager = MultiServerManager(servers=server)
    await server_manager.startup_all_servers()
    await server_manager.connect_all()

    df = pd.read_csv(output_file)

    start_idx = 0
    end_idx = min(max_idx, len(df)) if max_idx is not None else len(df)

    column_prefix = "Qwen/Qwen3-30B-A3B-Thinking-2507".replace("/", "_")

    # Find rows that need processing for report judge (only rows where abnormal_f1 is NaN)
    all_rows = list(range(start_idx, end_idx))
    abnormal_f1_col = f"{column_prefix}_abnormal_f1"
    if abnormal_f1_col in df.columns:
        rows_to_process = [
            idx for idx in all_rows if pd.isna(df.loc[idx, abnormal_f1_col])
        ]
    else:
        rows_to_process = all_rows

    print(f"Processing {len(rows_to_process)} rows", flush=True)

    trajectory_folder = Path(output_file).parent / "trajectory"
    all_json_trajectory_files = list(trajectory_folder.glob("*.json"))
    num_valid_tools_called_list = []
    valid_tools = SERVERS.keys()

    # Pre-compute trajectory data for all rows
    trajectory_data = []
    for idx in rows_to_process:
        image_id = df.loc[idx, "image_id"]
        trajectory_file = [
            f for f in all_json_trajectory_files if f.stem.startswith(image_id)
        ]
        if len(trajectory_file) == 0 or len(trajectory_file) > 1:
            trajectory_file = ""
        else:
            trajectory_file = trajectory_file[0]
        trajectory = json.load(open(trajectory_file)) if trajectory_file != "" else []
        # remove last message which is reward placeholder
        trajectory = trajectory[:-1] if trajectory_file != "" else []
        num_valid_tools_called_list.append(
            compute_manual_num_valid_tool_called(trajectory, valid_tools)
        )
        trajectory_data.append(trajectory)

    # Process in batches of 16
    BATCH_SIZE = 16
    for batch_start in range(0, len(rows_to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(rows_to_process))
        batch_indices = rows_to_process[batch_start:batch_end]
        print(
            f"Processing batch {batch_start // BATCH_SIZE + 1}: rows {batch_start} to {batch_end}", flush=True
        )

        tasks = []
        for idx in batch_indices:
            tasks.append(
                server_manager.call_tool(
                    "report_judge_tool",
                    "report_judge_tool",
                    {
                        "candidate_report": df["generated_report"][idx],
                        "ground_truth_report": df["gt_report"][idx],
                    },
                )
            )

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        number_of_failed = 0
        for idx, abn_judge_result in zip(
            batch_indices,
            batch_results,
        ):
            try:
                llm_reward_dict = json.loads(abn_judge_result)
            except Exception as e:
                print(f"Error parsing LLM judge reward: {e}", flush=True)
                llm_reward_dict = None
                number_of_failed += 1
            if llm_reward_dict is not None:
                for c in llm_reward_dict.keys():
                    df.loc[idx, f"{column_prefix}_{c}"] = (
                        str(llm_reward_dict[c])
                        if isinstance(llm_reward_dict[c], list)
                        else llm_reward_dict[c]
                    )
        # Save after each batch
        df.to_csv(output_file, index=False)

    # Process trajectory judge in batches of 24
    # Only process the rows for which checklist_adherence is NaN
    checklist_col = f"{column_prefix}_checklist_adherence"
    if checklist_col in df.columns:
        traj_rows_to_process = [
            i for i, idx in enumerate(rows_to_process)
            if pd.isna(df.loc[idx, checklist_col])
        ]
    else:
        traj_rows_to_process = list(range(len(rows_to_process)))

    traj_indices = [rows_to_process[i] for i in traj_rows_to_process]
    traj_trajectory_data = [trajectory_data[i] for i in traj_rows_to_process]
    traj_num_valid_tools = [num_valid_tools_called_list[i] for i in traj_rows_to_process]

    for batch_start in range(0, len(traj_indices), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(traj_indices))
        batch_trajectories = traj_trajectory_data[batch_start:batch_end]
        print(
            f"Processing trajectory batch {batch_start // BATCH_SIZE + 1}: rows {batch_start} to {batch_end}", flush=True
        )

        trajectory_judge_tasks = []
        for trajectory in batch_trajectories:
            trajectory_judge_tasks.append(
                server_manager.call_tool(
                    "report_judge_tool",
                    "trajectory_judge_tool",
                    {
                        "conversation_trajectory": trajectory,
                    },
                )
            )

        batch_traj_results = await asyncio.gather(*trajectory_judge_tasks, return_exceptions=True)
        number_of_failed = 0
        for idx, traj_result, num_tool in zip(
            traj_indices[batch_start:batch_end],
            batch_traj_results,
            traj_num_valid_tools[batch_start:batch_end],
        ):
            try:
                trajectory_judge_reward_dict = json.loads(traj_result)
            except Exception as e:
                trajectory_judge_reward_dict = None
                if number_of_failed > 5:
                    print(
                        f"More than 5 failures in batch {batch_start // BATCH_SIZE + 1}, restart server",
                        flush=True,
                    )

            if trajectory_judge_reward_dict is not None:
                checklist_adherence_score = trajectory_judge_reward_dict.get(
                    "checklist adherence", {}
                ).get("score", None)
                tool_seq_coherence_score = trajectory_judge_reward_dict.get(
                    "tool sequence coherence", {}
                ).get("score", None)
                if checklist_adherence_score is not None:
                    df.loc[idx, f"{column_prefix}_checklist_adherence"] = (
                        checklist_adherence_score
                    )
                if tool_seq_coherence_score is not None:
                    df.loc[idx, f"{column_prefix}_tool_seq_coherence"] = (
                        tool_seq_coherence_score
                    )
            df.loc[idx, "num_tools"] = num_tool
        # Save after each batch
        df.to_csv(output_file, index=False)

    # Compute average scores
    average_scores = {
        f"{column_prefix}_all_findings_f1": df[f"{column_prefix}_all_findings_f1"]
        .dropna()
        .mean(),
        f"{column_prefix}_abnormal_findings_f1": df[f"{column_prefix}_abnormal_f1"]
        .dropna()
        .mean(),
        f"{column_prefix}_abnormal_mean_rec_prec": df[
            f"{column_prefix}_abnormal_mean_rec_prec"
        ]
        .dropna()
        .mean(),
        "n_samples_processed": len(df[f"{column_prefix}_abnormal_f1"].dropna()),
        "checklist_adherence": (
            df[f"{column_prefix}_checklist_adherence"].dropna().mean()
            if f"{column_prefix}_checklist_adherence" in df.columns
            else None
        ),
        "tool_seq_coherence": (
            df[f"{column_prefix}_tool_seq_coherence"].dropna().mean()
            if f"{column_prefix}_tool_seq_coherence" in df.columns
            else None
        ),
        "num_tools": (
            df["num_tools"].dropna().mean() if "num_tools" in df.columns else None
        ),
    }

    print(average_scores)

    # Save final results
    df.to_csv(output_file, index=False)

    # Save summary
    output_path = Path(output_file).parent
    results_file = output_path / f"{column_prefix}_results.json"
    with open(results_file, "w") as f:
        json.dump(average_scores, f, indent=4)

    print(f"Results saved to {results_file}")
    print(f'Total processed: {len(df[f"{column_prefix}_abnormal_f1"].dropna())} rows')


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser(
        description="Add GPT or HuggingFace scores to results dataframe"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the detailed_results.csv file",
    )
    parser.add_argument(
        "--max-idx",
        type=int,
        default=None,
        help="Maximum index to process (default: process all rows)",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.output_file)

    if os.environ.get("RANK", "0") == "0":
        # Initialize columns for HF model
        column_prefix = "Qwen/Qwen3-30B-A3B-Thinking-2507".replace("/", "_")
        columns = [
            "equivalent",
            "all_findings_in_ground_truth",
            "all_findings_in_candidate",
            "abnormal_findings_in_ground_truth_missing_in_candidate",
            "abnormal_findings_in_candidate_missing_in_ground_truth",
            "abnormal_findings_in_ground_truth_partially_matched_in_candidate",
            "abnormal_findings_in_candidate_partially_matched_in_ground_truth",
            "recall",
            "precision",
            "f1",
            "all_abnormal_findings_in_ground_truth",
            "all_abnormal_findings_in_candidate",
            "abnormal_recall",
            "abnormal_precision",
            "abnormal_f1",
            "all_findings_recall",
            "all_findings_precision",
            "all_findings_f1",
            "abnormal_mean_rec_prec",
            "checklist_adherence",
            "tool_seq_coherence",
        ]
        for c in columns:
            col_name = f"{column_prefix}_{c}"
            if col_name not in df.columns:
                df[col_name] = np.nan
        df["num_tools"] = np.nan
        df.to_csv(args.output_file, index=False)

        asyncio.run(add_local_score_to_results_df(args.output_file, args.max_idx))
