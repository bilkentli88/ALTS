# sensitivity_sweep.py
#
# Runs a small sensitivity study for ALTS over:
#   - K (n_prototypes) in {30, 50, 70}
#   - lambda_fused_lasso in {1e-8, 1e-4, 1e-2}
# for three datasets: ECG200, Herring, TwoPatterns.
#
# For each (dataset, K, lambda_f) combination, it:
#   - reuses bag_size and lambda_prototypes from the existing Optuna search results
#   - runs evaluate_model_performance() N_RUNS times with different seeds
#   - computes mean accuracy and a 95% CI (using normal approximation)
#   - saves all results to CSV files in Results/sensitivity/

import os
import numpy as np
import pandas as pd

from optimize_shapelets import (
    get_filename_output_for_search,
    evaluate_model_performance
)
from set_seed import set_seed

# ---------------- CONFIG ----------------

DATASETS = ["Herring", "TwoPatterns"]

K_VALUES = [30, 50, 70]                 # n_prototypes
LAMBDA_F_VALUES = [1e-8, 1e-4, 1e-2]    # fused-lasso strength

N_RUNS = 3          # repeated runs per (dataset, K, lambda_f)
MAX_EPOCH = 30      # lower than full benchmark for faster sweeps

SEARCH_TYPE = "OptunaSearch"
SEARCH_N_TRIALS = 5
SEARCH_MAX_EPOCH = 50

OUTPUT_DIR = os.path.join("Results", "sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_SEED = 79  # base seed for reproducibility

# ----------------------------------------


def load_best_hyperparams_from_search(dataset_name):
    """
    Load best bag_size and lambda_prototypes from existing Optuna search CSV.
    Assumes you have already run get_train_results.py for this dataset
    with the same SEARCH_N_TRIALS and SEARCH_MAX_EPOCH.
    """
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
    # assumes the first row contains the best trial (this is how optimize_shapelets writes it)
    best = df.iloc[0]

    bag_size = int(best["bag_size"])
    lambda_prototypes = float(best["lambda_prototypes"])

    return bag_size, lambda_prototypes


def run_sensitivity_for_dataset(dataset_name):
    print(f"\n=== Sensitivity sweep for {dataset_name} ===")

    bag_size, lambda_prototypes = load_best_hyperparams_from_search(dataset_name)
    print(f"Using bag_size={bag_size}, lambda_prototypes={lambda_prototypes} from Optuna search.")

    records = []

    for K in K_VALUES:
        for lambda_f in LAMBDA_F_VALUES:
            accs = []

            for run_idx in range(N_RUNS):
                seed = BASE_SEED + run_idx
                set_seed(seed)

                print(
                    f"[{dataset_name}] K={K}, lambda_f={lambda_f}, "
                    f"run {run_idx+1}/{N_RUNS}, seed={seed}"
                )

                avg_acc, _ = evaluate_model_performance(
                    dataset_name=dataset_name,
                    bag_size=bag_size,
                    n_prototypes=K,
                    max_epoch=MAX_EPOCH,
                    phase="training",               # use CV as in training phase
                    lambda_fused_lasso=lambda_f,
                    lambda_prototypes=lambda_prototypes,
                    augment=True                    # consistent with your training setup
                )

                accs.append(avg_acc)

            accs = np.array(accs)
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0

            # 95% CI (normal approximation): mean ± 1.96 * std / sqrt(n)
            if len(accs) > 1:
                half_width = 1.96 * std_acc / np.sqrt(len(accs))
            else:
                half_width = 0.0

            ci_lower = mean_acc - half_width
            ci_upper = mean_acc + half_width

            records.append({
                "dataset": dataset_name,
                "bag_size": bag_size,
                "n_prototypes": K,
                "lambda_fused_lasso": lambda_f,
                "lambda_prototypes": lambda_prototypes,
                "n_runs": len(accs),
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
            })

    df_out = pd.DataFrame(records)
    out_file = os.path.join(OUTPUT_DIR, f"sensitivity_{dataset_name}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Saved sensitivity results for {dataset_name} to: {out_file}")


if __name__ == "__main__":
    print("Starting ALTS sensitivity sweep over K and lambda_fused_lasso...\n")
    for ds in DATASETS:
        run_sensitivity_for_dataset(ds)
    print("\nAll sensitivity analysis completed.")
