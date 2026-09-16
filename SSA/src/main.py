"""
Thesis pipeline for State Sequence Analysis (SSA) and Predictive Survival Modeling.

========================================================================================
HOW TO RUN THIS PIPELINE (CLI USAGE WITH ARGPARSE):
========================================================================================
You can execute the entire pipeline from start to finish or run individual modules
independently using the `--step` flag:

1. Run the entire pipeline (Steps 1 -> 2 -> 3):
   $ python main.py
   or
   $ python main.py --step all

2. Run ONLY the dataset engineering / trajectory extraction step:
   $ python main.py --step dataset

3. Run ONLY sequence analysis, dissimilarity computation, and clustering:
   $ python main.py --step analysis

4. Run ONLY clinical target definition and predictive/survival models:
   $ python main.py --step predict

5. Display CLI help and options:
   $ python main.py --help
========================================================================================
"""

import argparse
from config import SEQUENCE_NAME, CHANNEL, DISTANCE_METRIC, NUM_CLUSTERS

# Core step imports from modular scripts
from dataset_sequences import build_dataset
from sequence_analysis import run_sequence_analysis
from prediction import run_prediction


def main():
    parser = argparse.ArgumentParser(
        description="State Sequence Analysis & Predictive Survival Pipeline Runner"
    )
    parser.add_argument(
        "--step",
        choices=["all", "dataset", "analysis", "predict"],
        default="all",
        help="Select pipeline step to execute (default: 'all')",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("           THESIS BREATH: PIPELINE EXECUTION")
    print("=" * 65)
    print(f"Target Sequence     : {SEQUENCE_NAME}")
    print(f"Active Channel      : {CHANNEL}")
    print(f"Distance Metric     : {DISTANCE_METRIC}")
    print(f"Clusters (k)        : {NUM_CLUSTERS}")
    #print(f"Prediction Model    : {MODEL}")
    print(f"Selected Step       : {args.step}")
    print("=" * 65 + "\n")

    # STEP 1: Sequence engineering and cohort preprocessing
    if args.step in ["all", "dataset"]:
        print(">>> [Step 1/3] Generating patient sequence trajectories...")
        dataset_sequences = build_dataset()
        print(">>> [Step 1/3] Sequence dataset successfully created.\n")

    # STEP 2: Sequence metrics, distance matrices, and trajectory clustering
    if args.step in ["all", "analysis"]:
        print(">>> [Step 2/3] Running Sequence Analysis, Dissimilarity & Clustering...")
        run_sequence_analysis()
        print(">>> [Step 2/3] Clustering and sequence profiling completed.\n")

    # STEP 3: Survival modeling (Cox) and risk factor evaluation
    if args.step in ["predict"]:
        print(f">>> [Step 3/3] Training and evaluating predictive model...")
        run_prediction()
        print(">>> [Step 3/3] Predictive modeling pipeline completed.\n")

    print("=" * 65)
    print("       PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()