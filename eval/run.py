"""
run_eval.py

    python run_eval.py [--output eval_report.csv]

Everything you'd want to change lives in config.py (which datasets, which
targets, which models). This script just wires it together.
"""
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import DATASETS, MODELS
from eval_embeddings import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_report.csv")
    args = parser.parse_args()

    report = run_evaluation(DATASETS, MODELS, output_csv=args.output)
    print("\n", report)