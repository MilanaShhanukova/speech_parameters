import argparse
import os
import re
import sys
import json
import pandas as pd
from tqdm import tqdm

from objective.c50_metric import C50Metric
from objective.estoi_metric import ESTOIMetric
from objective.sd_metric import SignalDistortionMetric
from objective.snr_metric import SnrMetric
from objective.stoi_metric import STOIMetric
from subjective.mos_metric import MOSMetric
from subjective.pesq_metric import PESQMetric
from objective.srmr_metric import SRMRMetric


from typing import List

metric_instances = {
    "STOI": (STOIMetric, "referenced"),
    "ESTOI": (ESTOIMetric, "referenced"),
    "PESQ": (PESQMetric, "referenced"),
    "SD": (SignalDistortionMetric, "referenced"),
    "SNR": (SnrMetric, "not_referenced"),
    "SRMR": (SRMRMetric, "not_referenced"),
    "MOS": (MOSMetric, "not_referenced"),
    "C50": (C50Metric, "not_referenced")
}

def categorize_score(score, metric_name):
    """Categorize a score as 'high' or 'low' quality based on metric-specific thresholds"""
    if pd.isna(score):
        return "unknown"
        
    thresholds = {
        "PESQ": 2.0,    # Below 2 is poor quality
        "STOI": 0.5,    # Using middle point as threshold
        "ESTOI": 0.5,   # Using middle point as threshold
        "MOS": 3.0,     # Below 3 indicates poor quality
        "SNR": 15.0,    # Common threshold for acceptable SNR
        "SRMR": 5.0,    # Middle of typical range
        "C50": 0.0,     # Positive values indicate better clarity
        "SD": 0.0       # Higher values indicate better quality
    }
    
    threshold = thresholds.get(metric_name, 0.0)
    return "high" if score > threshold else "low"

def run_pipe(audio_files_target: List[str], metrics: List[str], output_csv: str, audio_files_source=None):
    """
    Runs the specified metrics on the target (dirty) audio files and optionally source (clean) audio files.
    
    Parameters:
    - audio_files_target: List of paths to dirty audio files.
    - audio_files_source: List of paths to clean audio files (can be empty if not referenced).
    - metrics: List of metric names to run.
    - output_csv: Path to the CSV file to save the results.
    
    Returns:
    - Saves the results to a CSV file.
    """
    
    info = {file_name: {} for file_name in audio_files_target}  

    source_given = audio_files_source is not None

    initialized_metrics = {}
    for metric_name in metrics:
        if metric_name in metric_instances:
            metric_class, metric_type = metric_instances[metric_name]
            initialized_metrics[metric_name] = (metric_class(), metric_type)
        else:
            print(f"Metric {metric_name} is not recognized.")
    
    for metric_name, (metric, metric_type) in tqdm(initialized_metrics.items()):
        for idx, audio_dirty in tqdm(enumerate(audio_files_target), total=len(audio_files_target), desc=metric_name):
            if source_given and metric_type == "referenced":
                audio_clean = audio_files_source[idx]
                score = metric.score_pair_audio(audio_clean, audio_dirty)
            elif not source_given or metric_type == "not_referenced":
                score = metric.score_audio(audio_dirty)
            else:
                score = None  # Handle case where clean audio is not provided for referenced metric
            
            # Store both score and quality rating
            info[audio_dirty][f"{metric_name}_score"] = score
            info[audio_dirty][f"{metric_name}_quality"] = categorize_score(score, metric_name)

    df = pd.DataFrame.from_dict(info, orient='index')
    df.index.name = 'file_name'
    
    # Add overall quality based on majority vote
    quality_columns = [col for col in df.columns if col.endswith('_quality')]
    df['overall_quality'] = df[quality_columns].apply(
        lambda x: 'high' if (x == 'high').sum() > (x == 'low').sum() else 'low', 
        axis=1
    )
    
    df.to_csv(output_csv)
    return df

if __name__ == "__main__":
    # python metrics_frame_pipe.py --csv_path speech_parameters/dataframe_example_noisy.csv --dirty_column file_path --metrics MOS SNR SRMR C50 --output_file noisy_examples_wavs.csv
    parser = argparse.ArgumentParser(description="Run specified metrics on audio files")
    parser.add_argument("--csv_path", type=str, help="model that was used")
    parser.add_argument("--dirty_column", type=str, help="column that includes dirty files")
    parser.add_argument("--clean_column", default=None, type=str, help="column that includes clean files", required=False)
    parser.add_argument(
        "--metrics",
        help="List of metrics to run",
        required=True,
        nargs='+',
        choices=list(metric_instances.keys())
    )
    parser.add_argument(
        "--output_file"
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.csv_path)
    audio_files_target = dataset[args.dirty_column].to_list() # generated audio

    if args.clean_column:
        clean_audio_files = dataset[args.clean_column].to_list() # source audio
    else:
        clean_audio_files = [None] * len(audio_files_target)
    

    run_pipe(audio_files_target=audio_files_target,
            audio_files_source=clean_audio_files,
            metrics=args.metrics,
            output_csv=args.output_file)
