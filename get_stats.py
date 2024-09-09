import pandas as pd
from metrics_pipeline import run_specified_metrics
import re
import argparse
from tqdm import tqdm

def process_audio_files(clean_audio_paths, dirty_audio_paths, metrics, output_file):
    all_results = []
    
    for clean_path, dirty_path in tqdm(zip(clean_audio_paths, dirty_audio_paths)):
        results = run_specified_metrics(dirty_path, metrics, clean_path)
        results['clean_audio'] = clean_path
        results['dirty_audio'] = dirty_path
        all_results.append(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specified metrics on audio files")
    parser.add_argument("--csv_path", type=str, help="model that was used")
    parser.add_argument("--dirty_column", type=str, help="column that includes dirty files")
    parser.add_argument("--clean_column", type=str, help="column that includes clean files")
    parser.add_argument(
        "--metrics",
        help="List of metrics to run",
        required=True,
        type=lambda s: re.split(" |, ", s),
    )
    parser.add_argument(
        "--output_file"
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.csv_path)
    clean_audio_files = dataset[args.clean_column].to_list()
    target_audio_files = dataset[args.dirty_column].to_list()

    process_audio_files(clean_audio_files, target_audio_files,
                    metrics=args.metrics, output_file=args.output_file)