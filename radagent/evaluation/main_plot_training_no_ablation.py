"""
Code to plot the main results.
"""

import json
from pathlib import Path

import pandas as pd

from constants_and_path_utils import (
    FIGURES_DIR,
    RADAGENT_REPO_ROOT,
    RADAGENT_RESULTS_DIR,
)
from evaluation.plotting_utils import (
    BASELINE_NAME,
    get_bootstrap_results,
    plot_bar_metrics_with_significance,
    assess_signficance,
    plot_diff_to_baseline,
    plot_mirrored_sens_spec,
    plot_pathology_grouped_bars,
)

FIGURE_FOLDER = FIGURES_DIR
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)

RADAGENT_TRAININGFREE_NAME = "RadAgent Training Free"
RADAGENT_NAME = "RadAgent"
CUSTOM_COLORS = [
    "#5296a5ff",
    "#aef6c7ff",
]

# These maps are provided with the Github repo for convenience.
TASK_ID_IMAGE_ID_MAPS = {
    "radchest": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_test_radchestct_start0_endNone.json"
    ),
    "ct_rate_test": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_test_ctrate_start0_endNone.json"
    ),
    "ct_rate_val": str(
        RADAGENT_REPO_ROOT
        / "radagent/outputs/task_to_id_maps/task_id_to_image_id_base_val_ctrate_start0_end1000.json"
    ),
}

#Note: These are placeholders. You have to replace the paths to the detailed_results.csv
# This file gets created by executing the metrics script, which inturn requires running inference on the evaluation datasets first.
DATASET_PATHS = {
    "radchest": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<radchest_inference_results/radagent_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<radchest_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "RadChestCT",
        "file_prefix": "radchest_baseline_vs_radagent",
    },
    "ct_rate_test": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_test_inference_results/radagent_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_test_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Test Set",
        "file_prefix": "ct_rate_test_baseline_vs_radagent",
    },
    "ct_rate_val": {
        RADAGENT_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_val_inference_results/radagent_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_val_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Validation Set",
        "file_prefix": "ct_rate_val_baseline_vs_radagent",
    },
}

# Note: As above, these placeholders have to be set by you
TRAININGFREE_DATASET_PATHS = {
    "radchest": {
        RADAGENT_TRAININGFREE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<radchest_training_free_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<radchest_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "RadChestCT",
        "file_prefix": "radchest_baseline_vs_training_free_radagent",
    },
    "ct_rate_test": {
        RADAGENT_TRAININGFREE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_test_training_free_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_test_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Test Set",
        "file_prefix": "ct_rate_test_baseline_vs_training_free_radagent",
    },
    "ct_rate_val": {
        RADAGENT_TRAININGFREE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_val_training_free_results/detailed_results.csv>"
        ),
        BASELINE_NAME: str(
            RADAGENT_RESULTS_DIR
            / "<ct_rate_val_inference_results/ct_chat_results/detailed_results.csv>"
        ),
        "title": "CT-RATE Validation Set",
        "file_prefix": "ct_rate_val_baseline_vs_training_free_radagent",
    },
}


OPTIONAL_COLUMNS = [
    "num_tools",
    "Qwen_Qwen3-30B-A3B-Thinking-2507_abnormal_f1",
]


def normalize_path(path_str: str) -> Path:
    if path_str.startswith("/"):
        return Path(path_str)
    return Path("/" + path_str)


def format_int(value: int) -> str:
    return f"{value:,}"


def print_separator(char: str = "=", width: int = 80) -> None:
    print(char * width)

def merge_image_id(df: pd.DataFrame, dataset_key: str):
    # CT Chat dfs contain task id as image id. Needs changing
    with open(TASK_ID_IMAGE_ID_MAPS[dataset_key], "r") as f:
        task_to_image = json.load(f)
    
    df = df.rename(columns={"image_id": "task_id"})
    mapping_df = pd.DataFrame(
            task_to_image.items(),
            columns=["task_id", "image_id"]
            )
    df = df.merge(mapping_df, on="task_id", how="left")
    missing = df[df["image_id"].isna()]
    if len(missing) > 0:
        raise ValueError("ID Merge left NAs")
    return df

def load_result_maps(dataset_key: str, dataset_config: dict, radagent_name: str) -> dict[str, pd.DataFrame]:
    print(f"Loading CSV files for dataset: {dataset_key}")
    df_maps: dict[str, pd.DataFrame] = {}
    for model_name in (BASELINE_NAME, radagent_name):
        csv_path = normalize_path(dataset_config[model_name])
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {model_name}: {csv_path}")
        print(f"Loading {model_name} from {csv_path}")
        df = pd.read_csv(csv_path)
        if model_name == BASELINE_NAME:
            df = merge_image_id(df, dataset_key=dataset_key)
        print(f"Loaded {model_name}: len = {format_int(len(df))}, shape = {df.shape}")
        df_maps[model_name] = df
    return df_maps


def print_dataset_summary(dataset_key: str, dataset_title: str, df_maps: dict[str, pd.DataFrame], radagent_name: str) -> None:
    print_separator("-")
    print(f"Dataset summary for {dataset_key} ({dataset_title})")

    baseline_df = df_maps[BASELINE_NAME]
    radagent_df = df_maps[radagent_name]

    for model_name, df in ((BASELINE_NAME, baseline_df), (radagent_name, radagent_df)):
        print(f"{model_name}: len = {format_int(len(df))}, shape = {df.shape}")
        if "id" in df.columns:
            unique_ids = df["id"].nunique(dropna=True)
            print(f"{model_name}: unique ids = {format_int(unique_ids)}")
        else:
            print(f"{model_name}: no 'id' column present")

        missing_optional = [col for col in OPTIONAL_COLUMNS if col not in df.columns]
        if missing_optional:
            print(f"{model_name}: missing optional columns = {missing_optional}")
        else:
            print(f"{model_name}: all optional columns present")

    if "id" in baseline_df.columns and "id" in radagent_df.columns:
        baseline_ids = set(baseline_df["id"].dropna().tolist())
        radagent_ids = set(radagent_df["id"].dropna().tolist())
        shared_ids = baseline_ids & radagent_ids
        baseline_only_ids = baseline_ids - radagent_ids
        radagent_only_ids = radagent_ids - baseline_ids
        print(f"Shared ids: {format_int(len(shared_ids))}")
        print(f"Baseline only ids: {format_int(len(baseline_only_ids))}")
        print(f"{radagent_name} only ids: {format_int(len(radagent_only_ids))}")

    print_separator("-")


def plot_all_for_dataset(dataset_key: str, dataset_config: dict, radagent_name: str) -> None:
    dataset_title = dataset_config["title"]
    file_prefix = dataset_config["file_prefix"]
    names_to_plot = [BASELINE_NAME, radagent_name]
    colors = CUSTOM_COLORS[: len(names_to_plot)]

    df_maps = load_result_maps(dataset_key, dataset_config, radagent_name)
    for name in names_to_plot:
        print(df_maps[name])
        print(df_maps[name].columns)
    print_dataset_summary(dataset_key, dataset_title, df_maps, radagent_name)

    print(f"Running bootstrap evaluation for: {names_to_plot}")
    big_df, big_df_diff = get_bootstrap_results(df_maps, names_to_plot)

    variables_to_compute = big_df_diff.index.tolist()
    significance_df = assess_signficance(
            df=df_maps[radagent_name], 
            baseline_df=df_maps[BASELINE_NAME], 
            variables_to_compute=variables_to_compute, 
            names=names_to_plot
            )
    big_df_diff_with_sig = big_df_diff.join(significance_df, how="left")
    
    print(f"Finished bootstrap evaluation for {dataset_key}")
    print(f"Bootstrap metrics shape: {big_df.shape}")
    print(f"Bootstrap diff metrics shape: {big_df_diff.shape}")

    summary_path = FIGURE_FOLDER / f"{file_prefix}_summary.pdf"
    print(f"Creating summary plot: {summary_path}")
    plot_bar_metrics_with_significance(
        big_df,
        names_to_plot,
        target_metrics=["Macro-F1", "Micro-F1"],
        colors=colors,
        df_diff=big_df_diff_with_sig,
        baseline_name=BASELINE_NAME,
        title=f"Baseline vs {radagent_name}\n{dataset_title}",
        savepath=summary_path,
    )

    f1_path = FIGURE_FOLDER / f"{file_prefix}_pathologies_f1.pdf"
    print(f"Creating pathology F1 plot: {f1_path}")
    pathology_order = plot_pathology_grouped_bars(
        my_df=big_df,
        colors=colors,
        metric="f1",
        title=f"Pathology Recognition F1 Scores\n{dataset_title}",
        savepath=f1_path,
        df_diff=big_df_diff,
        baseline_name=BASELINE_NAME,
    )
    print(f"Computed pathology plotting order with {len(pathology_order)} pathologies")

    sensitivity_path = FIGURE_FOLDER / f"{file_prefix}_pathologies_sensitivity.pdf"
    print(f"Creating pathology sensitivity plot: {sensitivity_path}")
    plot_pathology_grouped_bars(
        my_df=big_df,
        colors=colors,
        metric="sensitivity",
        title=f"Pathology Recognition Sensitivity Scores\n{dataset_title}",
        savepath=sensitivity_path,
        df_diff=big_df_diff,
        baseline_name=BASELINE_NAME,
        pathology_order=pathology_order,
    )

    specificity_path = FIGURE_FOLDER / f"{file_prefix}_pathologies_specificity.pdf"
    print(f"Creating pathology specificity plot: {specificity_path}")
    plot_pathology_grouped_bars(
        my_df=big_df,
        colors=colors,
        metric="specificity",
        title=f"Pathology Recognition Specificity Scores\n{dataset_title}",
        savepath=specificity_path,
        df_diff=big_df_diff,
        baseline_name=BASELINE_NAME,
        pathology_order=pathology_order,
    )

    diff_path = FIGURE_FOLDER / f"{file_prefix}_diff_to_baseline.pdf"
    print(f"Creating difference to baseline plot: {diff_path}")
    plot_diff_to_baseline(
        my_df=big_df,
        baseline_name=BASELINE_NAME,
        df_diff=big_df_diff,
        colors=[CUSTOM_COLORS[1]],
        title=f"Difference to Baseline: Sensitivity vs Specificity\n{dataset_title}",
        pathology_order=pathology_order,
        savepath=diff_path,
    )

    mirrored_path = FIGURE_FOLDER / f"{file_prefix}_mirrored_sens_spec.pdf"
    print(f"Creating mirrored sensitivity vs specificity plot: {mirrored_path}")
    plot_mirrored_sens_spec(
        my_df=big_df,
        colors=colors,
        title=f"Pathology Recognition: Sensitivity vs Specificity\n{dataset_title}",
        savepath=mirrored_path,
        df_diff=big_df_diff,
        baseline_name=BASELINE_NAME,
    )

    print(f"Finished dataset: {dataset_key}")
    print(f"Saved figures to: {FIGURE_FOLDER}")



def main() -> None:
    # Training Free Evaluation
    tfree_dataset_paths = TRAININGFREE_DATASET_PATHS
    print("Starting evaluation")
    print(f"Figure output folder: {FIGURE_FOLDER}")
    print(f"Datasets to process: {list(tfree_dataset_paths.keys())}")
    print_separator()

    for dataset_key, dataset_config in tfree_dataset_paths.items():
        print_separator()
        print(f"Processing dataset: {dataset_key}")
        plot_all_for_dataset(dataset_key, dataset_config, radagent_name = RADAGENT_TRAININGFREE_NAME)

    print_separator()
    print("Finished baseline vs training-free RadAgent evaluation")
    print(f"All figures saved in: {FIGURE_FOLDER}")
    del tfree_dataset_paths
    
    # Evaluation of RL Trained RadAgent
    dataset_paths = DATASET_PATHS
    print("Starting evaluation")
    print(f"Figure output folder: {FIGURE_FOLDER}")
    print(f"Datasets to process: {list(dataset_paths.keys())}")
    print_separator()

    for dataset_key, dataset_config in dataset_paths.items():
        print_separator()
        print(f"Processing dataset: {dataset_key}")
        plot_all_for_dataset(dataset_key, dataset_config, radagent_name = RADAGENT_NAME)

    print_separator()
    print("Finished baseline vs training-free RadAgent evaluation")
    print(f"All figures saved in: {FIGURE_FOLDER}")


if __name__ == "__main__":
    main()
