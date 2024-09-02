import os

import numpy as np
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from pyannote.audio import Model

from base import Metric


class SnrMetric(Metric):
    name = "SNR"

    # "Bredin, Hervé and Antoine Laurent. “End-to-end speaker segmentation for overlap-aware resegmentation.” Interspeech (2021)."
    def __init__(self):
        super().__init__()
        self._load_model()

    def _load_model(self):
        if not os.environ.get("HF_TOKEN"):
            raise Exception(
                "HF_TOKEN is not set. Please set HF_TOKEN to use this model."
            )

        self.model = Model.from_pretrained(
            "pyannote/brouhaha", strict=False, device="cpu"
        )
        self.pipeline = RegressiveActivityDetectionPipeline(segmentation=self.model)

    def score_audio(self, audio_path: str):
        try:
            results = self.pipeline(audio_path)
            snr = np.mean(results["snr"])
            return snr
        except Exception as e:
            return -420.69

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        score_clean = self.score_audio(target_audio_path)
        score_dirty = self.score_audio(source_audio_path)
        # positive difference -> the dirty sample is worse quality
        return score_dirty - score_clean

    def explain_score(self):
        return super().explain_score(self.name)
