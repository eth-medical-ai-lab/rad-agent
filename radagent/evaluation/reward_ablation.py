import importlib
from pathlib import Path
import pandas as pd  
  
from constants_and_path_utils import FIGURES_DIR, RADAGENT_RESULTS_DIR
#import plotting_utils
from evaluation.plotting_utils import (
      BASELINE_NAME,
      get_bootstrap_results,
      plot_bar_metrics_with_errorbars,
  )

#importlib.reload(plotting_utils)
FIGURE_FOLDER = FIGURES_DIR
FIGURE_FOLDER.mkdir(parents=True, exist_ok=True)




def main():
    df_maps_val = {
    
    # Training free - without summary check during inference
    "Before RL (Qwen3-14B-base, WITH CT-Chat report, WITH findings scratchpad)": pd.read_csv(
        RADAGENT_RESULTS_DIR / "training_free_ct_rate_val_results/detailed_results.csv"
    ),
    
    # Main run - without summary check during inference
    "After 150 RL steps (Qwen3-14B-base, WITH CT-Chat report, WITH findings scratchpad)": pd.read_csv(
        RADAGENT_RESULTS_DIR / "main_rl_ct_rate_val_results/detailed_results.csv"
    ),

    # Run without using the tool sequence judge, for reward ablation
    "After 150 RL steps DEGENERATE": pd.read_csv(
        RADAGENT_RESULTS_DIR / "no_sequence_judge_reward_results/detailed_results.csv"
    ),

    # Run without with the tool sequence judge from beginning, for reward ablation
    "v8c_base_scratch_52": pd.read_csv(
        RADAGENT_RESULTS_DIR / "sequence_judge_from_start_results/detailed_results.csv"
    ),
    
    BASELINE_NAME: pd.read_csv(
        RADAGENT_RESULTS_DIR / "ct_chat_ct_rate_val_results/detailed_results.csv"
    ),
}

    df_maps_val["No sequence judge reward"] = df_maps_val["After 150 RL steps DEGENERATE"]
    df_maps_val["Sequence judge from the start"] = df_maps_val["v8c_base_scratch_52"]
    df_maps_val["Mixed reward"] = df_maps_val[
        "After 150 RL steps (Qwen3-14B-base, WITH CT-Chat report, WITH findings scratchpad)"
    ]
    names_to_plot = [
        "Mixed reward",
        "No sequence judge reward",
        "Sequence judge from the start",
    ]
    colors = ["lightskyblue", "powderblue", "cadetblue"]
    big_df, _ = get_bootstrap_results(df_maps_val, names_to_plot)
    big_df.replace("nan [nan,nan]", "0.00 [0.00,0.00]", inplace=True)
    plot_bar_metrics_with_errorbars(
        big_df,
        names_to_plot,
        target_metrics=["Macro-F1", "Micro-F1"],
        colors=colors,
        title="Reward ablation: report quality\nCT-RATE Validation Set",
        savepath=FIGURE_FOLDER / "ctrate_val_reward_abalation_quali.pdf",
        x_width=4
    )
    plot_bar_metrics_with_errorbars(
        big_df,
        names_to_plot,
        target_metrics=[
            "ChecklistAdherenceJudge",
            "ToolSequenceCoherenceJudge",
        ],
        colors=colors,
        title="Reward ablation: tool sequence scores\nCT-RATE Validation Set",
        savepath=FIGURE_FOLDER / "ctrate_val_reward_abalation_tool.pdf",
        x_width=4
    )

if __name__ == "__main__":
    main()
