import argparse
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

from objective.c50_metric import C50Metric
from objective.estoi_metric import ESTOIMetric
from objective.sd_metric import SignalDistortionMetric
from objective.snr_metric import SnrMetric
from objective.stoi_metric import STOIMetric
from subjective.mos_metric import MOSMetric
from subjective.pesq_metric import PESQMetric


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull
        sys.stderr = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._devnull.close()


def run_specified_metrics(audio_dirty, metrics, audio_clean=None, explain=True):
    metric_instances = {
        "STOI": (STOIMetric(), "referenced"),
        "MOS": (MOSMetric(), "not_referenced"),
        "C50": (C50Metric(), "not_referenced"),
        "ESTOI": (ESTOIMetric(), "referenced"),
        "PESQ": (PESQMetric(), "referenced"),
        "SD": (SignalDistortionMetric(), "referenced"),
        "SNR": (SnrMetric(), "not_referenced"),
    }

    results = {}
    for metric_name in tqdm(metrics):
        if metric_name in metric_instances:
            metric_instance = metric_instances[metric_name][0]
            if explain:
                metric_instance.explain_score()
            metric_type = metric_instances[metric_name][1]
            with SuppressOutput():
                if audio_clean and metric_type == "referenced":
                    score = metric_instance.score_pair_audio(audio_clean, audio_dirty)
                elif audio_clean is None or metric_type == "not_referenced":
                    score = metric_instance.score_audio(audio_dirty)
            results[metric_name] = score
        else:
            print(f"Metric {metric_name} is not recognized.")
    return results


def save_results_to_csv(reference_audio_path, noisy_audio_path, results, output_file):
    results["reference_audio_path"] = reference_audio_path
    results["noisy_audio_path"] = noisy_audio_path

    df = pd.DataFrame(results, index=[0])
    df.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specified metrics on audio files")
    parser.add_argument("--audio_file_1", type=str, help="Path to the first audio file")
    parser.add_argument(
        "--audio_file_2", type=str, help="Path to the second audio file", default=None
    )
    parser.add_argument(
        "--metrics",
        help="List of metrics to run",
        required=True,
        type=lambda s: re.split(" |, ", s),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output CSV file to save results",
        default="results.csv",
    )
    parser.add_argument("--explain", type=bool, default=True)

    args = parser.parse_args()

    print(f"Parse metrics {args.metrics}")
    results = run_specified_metrics(
        args.audio_file_1, args.metrics, args.audio_file_2, explain=args.explain
    )
    save_results_to_csv(args.audio_file_1, args.audio_file_2, results, args.output_file)