import pandas as pd
import json
import numpy as np
import argparse


def load_metric_descriptions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def categorize_score(score, metric_name):
    """
    Categorize a score as 'high' or 'low' quality based on metric-specific thresholds
    """
    if pd.isna(score):
        return "unknown"
        
    # Metric-specific thresholds
    thresholds = {
        "PESQ": 2.0,  # Below 2 is poor quality
        "STOI": 0.75,  # Using middle point as threshold
        "ESTOI": 0.75, # Using middle point as threshold
        "MOS": 3.0,   # Below 3 indicates poor quality
        "SNR": 15.0,  # Common threshold for acceptable SNR
        "SRMR": 5.0,  # Middle of typical range
        "C50": 0.0,   # Positive values indicate better clarity
        "SD": 0.0     # Higher values indicate better quality
    }
    
    threshold = thresholds.get(metric_name, 0.0)
    return "high" if score > threshold else "low"

def analyze_audio_quality(results_csv, descriptions_json):
    df = pd.read_csv(results_csv)
    descriptions = load_metric_descriptions(descriptions_json)
    
    # Create quality columns for each metric
    metrics = [col for col in df.columns if col != 'file_name']
    df["quality"] = [0] * df.shape[0]
    
    results = []
    
    for _, row in df.iterrows():
        file_quality = {
            'file_name': row['file_name'],
            'quality_ratings': {},
            'overall_quality': 'unknown'
        }
        
        quality_counts = {'high': 0, 'low': 0}
        
        for metric in metrics:
            score = row[metric]
            quality = categorize_score(score, metric)
            file_quality['quality_ratings'][metric] = {
                'score': score,
                'quality': quality,
                'interpretation': descriptions[metric]['high'] if quality == 'high' 
                                else descriptions[metric]['low']
            }            
            if quality in ['high', 'low']:
                quality_counts[quality] += 1
        
        # Determine overall quality (if more than 50% metrics agree)
        total_valid_ratings = sum(quality_counts.values())
        if total_valid_ratings > 0:
            if quality_counts['high'] / total_valid_ratings > 0.5:
                file_quality['overall_quality'] = 'high'
            elif quality_counts['low'] / total_valid_ratings > 0.5:
                file_quality['overall_quality'] = 'low'
        
        results.append(file_quality)

    df['overall_quality'] = [results[i]["overall_quality"] for i in range(len(df))]
    df.to_csv(results_csv, index=False)
    return results

def print_analysis(results):
    for file_result in results:
        print(f"\nFile: {file_result['file_name']}")
        print(f"Overall Quality: {file_result['overall_quality'].upper()}")
        print("\nDetailed Metrics:")
        
        for metric, details in file_result['quality_ratings'].items():
            print(f"\n{metric}:")
            print(f"  Score: {details['score']:.3f}")
            print(f"  Quality: {details['quality'].upper()}")
            print(f"  Interpretation: {details['interpretation']}")
        print("-" * 80)

if __name__ == "__main__":
    # python speech_parameters/metrics_analytics.py --results_csv /speech_parameters/noisy_examples_wavs.csv
    parser = argparse.ArgumentParser(description="Analyze audio quality metrics")
    parser.add_argument("--results_csv", required=True, help="Path to the CSV file with metric results")
    parser.add_argument("--descriptions_json", 
                       default="./speech_parameters/metrics_descriptions.json",
                       help="Path to the JSON file with metric descriptions")
    
    args = parser.parse_args()
    
    results = analyze_audio_quality(args.results_csv, args.descriptions_json)
    print_analysis(results)
    
    output_json = args.results_csv.replace('.csv', '_analysis.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed analysis saved to: {output_json}")