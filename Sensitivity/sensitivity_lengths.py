# sensitivity_lengths.py
#
# Small sensitivity study for the candidate shapelet length (bag_size),
# interpreted as the "L" hyperparameter in the paper.
#
# For each dataset in DATASETS:
#   - Load best bag_size, K (n_prototypes), lambda_fused_lasso, lambda_prototypes from Optuna CSV
#   - Define three length variants: short (0.5*bag_size), base (1.0*bag_size), long (1.5*bag_size),
#     clamped to the time-series length
#   - For each variant, run N_RUNS times and compute mean accuracy + 95% CI
#   - Save results to Results/sensitivity_lengths/sensitivity_lengths_<dataset>.csv

import os
import numpy as np
import pandas as pd

from optimize_shapelets import (
    get_filename_output_for_search,
    evaluate_model_performance,
)
from set_seed import set_seed

# ----------------- CONFIG -----------------

DATASETS = ["Herring", "TwoPatterns"]

LENGTH_MULTIPLIERS = {
    "short": 0.5,
    "base": 1.0,
    "long": 1.5,
}

N_RUNS = 3
MAX_EPOCH = 50  # shorter than full training; this is a robustness check

SEARCH_TYPE = "OptunaSearch"
SEARCH_N_TRIALS = 5          # matches your get_train_results.py
SEARCH_MAX_EPOCH = 50

OUTPUT_DIR = os.path.join("Results", "sensitivity_lengths")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_SEED = 79

# same folder as in process_datasets.py
DATASET_MAIN_FOLDER = "Datasets"

# ------------------------------------------


def get_time_series_size(dataset_name: str) -> int:
    """Return the length T (number of time points) of the series for a dataset."""
    train_file_txt = os.path.join(DATASET_MAIN_FOLDER, f"{dataset_name}_TRAIN.txt")
    if not os.path.exists(train_file_txt):
        # fallback: no extension version if needed
        train_file_txt = os.path.join(DATASET_MAIN_FOLDER, f"{dataset_name}_TRAIN")
    data = np.loadtxt(train_file_txt)
    # last dimension includes label+features → subtract 1 for label
    return data.shape[1] - 1


def load_best_from_search(dataset_name):
    """Load best bag_size, K, lambda_f, lambda_s from Optuna summary CSV."""
    search_filename = get_filename_output_for_search(
        search_type=SEARCH_TYPE,
        dataset_name=dataset_name,
        n_trials=SEARCH_N_TRIALS,
        max_epoch=SEARCH_MAX_EPOCH,
    )

    if not os.path.exists(search_filename):
        raise FileNotFoundError(
            f"Search results file not found: {search_filename}\n"
            f"Run: python get_train_results.py {dataset_name}"
        )

    df = pd.read_csv(search_filename)
    best = df.iloc[0]

    bag_size = int(best["bag_size"])
    K = int(best["n_prototypes"])
    lambda_f = float(best["lambda_fused_lasso"])
    lambda_s = float(best["lambda_prototypes"])

    return bag_size, K, lambda_f, lambda_s


def run_length_sensitivity(dataset_name):
    print(f"\n=== Length (bag_size) sensitivity for {dataset_name} ===")

    T = get_time_series_size(dataset_name)
    print(f"Time series length T = {T}")

    base_bag_size, K, lambda_f, lambda_s = load_best_from_search(dataset_name)
    print(
        f"Using Optuna best: bag_size={base_bag_size}, K={K}, "
        f"lambda_f={lambda_f}, lambda_s={lambda_s}"
    )

    records = []

    for policy_name, mult in LENGTH_MULTIPLIERS.items():
        raw_size = int(round(mult * base_bag_size))
        # clamp to [2, T]
        bag_size = max(2, min(raw_size, T))

        print(
            f"[{dataset_name}] policy={policy_name}, multiplier={mult}, "
            f"bag_size={bag_size}"
        )

        accs = []

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            set_seed(seed)

            mean_acc, _ = evaluate_model_performance(
                dataset_name=dataset_name,
                bag_size=bag_size,
                n_prototypes=K,
                max_epoch=MAX_EPOCH,
                phase="training",
                lambda_fused_lasso=lambda_f,
                lambda_prototypes=lambda_s,
                augment=True,
            )
            accs.append(mean_acc)

        accs = np.array(accs)
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        half_width = 0.0
        if len(accs) > 1:
            half_width = 1.96 * std_acc / np.sqrt(len(accs))

        ci_lower = mean_acc - half_width
        ci_upper = mean_acc + half_width

        records.append({
            "dataset": dataset_name,
            "policy": policy_name,
            "multiplier": mult,
            "bag_size_base": base_bag_size,
            "bag_size_effective": bag_size,
            "n_prototypes": K,
            "lambda_fused_lasso": lambda_f,
            "lambda_prototypes": lambda_s,
            "n_runs": len(accs),
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        })

    df_out = pd.DataFrame(records)
    out_file = os.path.join(OUTPUT_DIR, f"sensitivity_lengths_{dataset_name}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Saved length sensitivity for {dataset_name} to: {out_file}")


if __name__ == "__main__":
    print("Starting length (bag_size) sensitivity sweeps...\n")
    for ds in DATASETS:
        run_length_sensitivity(ds)
    print("\nAll length sensitivity sweeps completed.")
