from abc import ABC

from utils import read_json


class Metric(ABC):
    name: str = None

    def __init__(self): ...

    def score_audio(self, audio_path: str): ...

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str): ...

    def explain_score(self, metric_name: str):
        data_metrics = read_json(
            "metrics_descriptions.json"
        )
        assert metric_name in data_metrics, "Metric doesn't have a description"

        metric_info = data_metrics[metric_name]
        return metric_info
