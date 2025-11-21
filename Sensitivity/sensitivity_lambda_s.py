# sensitivity_lambda_s.py
#
# Small sensitivity study for the shapelet L2 regularization coefficient λ_s
# (lambda_prototypes in the code).
#
# For each dataset in DATASETS:
#   - Load best bag_size, K (n_prototypes), lambda_fused_lasso from Optuna CSV
#   - Fix them
#   - Vary lambda_prototypes ∈ {1e-8, 1e-4, 1e-2}
#   - For each value, run N_RUNS times and compute mean accuracy + 95% CI
#   - Save results to Results/sensitivity_lambda_s/sensitivity_lambda_s_<dataset>.csv

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

LAMBDA_S_VALUES = [1e-8, 1e-4, 1e-2]
N_RUNS = 3
MAX_EPOCH = 30  # keep short, this is sensitivity, not full benchmark

SEARCH_TYPE = "OptunaSearch"
SEARCH_N_TRIALS = 5         # matches your get_train_results.py
SEARCH_MAX_EPOCH = 50

OUTPUT_DIR = os.path.join("Results", "sensitivity_lambda_s")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_SEED = 24680

# ------------------------------------------


def load_best_from_search(dataset_name):
    """Load best bag_size, K, lambda_f from Optuna summary CSV."""
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
    # we will override lambda_prototypes in the sweep
    return bag_size, K, lambda_f


def run_lambda_s_sensitivity(dataset_name):
    print(f"\n=== λ_s sensitivity for {dataset_name} ===")

    bag_size, K, lambda_f = load_best_from_search(dataset_name)
    print(f"Using bag_size={bag_size}, K={K}, lambda_f={lambda_f} from Optuna search.")

    records = []

    for lambda_s in LAMBDA_S_VALUES:
        accs = []

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            set_seed(seed)
            print(
                f"[{dataset_name}] λ_s={lambda_s:.1e}, "
                f"run {run_idx+1}/{N_RUNS}, seed={seed}"
            )

            mean_acc, _ = evaluate_model_performance(
                dataset_name=dataset_name,
                bag_size=bag_size,
                n_prototypes=K,
                max_epoch=MAX_EPOCH,
                phase="training",   # as in the previous sensitivity script
                lambda_fused_lasso=lambda_f,
                lambda_prototypes=lambda_s,
                augment=True,
            )
            accs.append(mean_acc)

        accs = np.array(accs)
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
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
            "lambda_prototypes": lambda_s,
            "n_runs": len(accs),
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
        })

    df_out = pd.DataFrame(records)
    out_file = os.path.join(OUTPUT_DIR, f"sensitivity_lambda_s_{dataset_name}.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Saved λ_s sensitivity results for {dataset_name} to: {out_file}")


if __name__ == "__main__":
    print("Starting λ_s sensitivity sweeps...\n")
    for ds in DATASETS:
        run_lambda_s_sensitivity(ds)
    print("\nAll λ_s sensitivity sweeps completed.")
