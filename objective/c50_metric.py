import os

import numpy as np
from brouhaha.pipeline import RegressiveActivityDetectionPipeline
from pyannote.audio import Model

from base import Metric


class C50Metric(Metric):
    name = "C50"
    # "Bredin, Hervé and Antoine Laurent. End-to-end speaker segmentation for overlap-aware resegmentation."

    def __init__(self):
        super().__init__()
        self._load_model()

    def _load_model(self):
        # check if HF_TOKEN is set
        if not os.environ.get("HF_TOKEN"):
            raise Exception(
                "HF_TOKEN is not set. Please set HF_TOKEN to use this model."
            )

        self.model = Model.from_pretrained(
            "pyannote/brouhaha", strict=False, device="cpu"
        )
        self.pipeline = RegressiveActivityDetectionPipeline(self.model)

    def score_audio(self, audio_path: str):
        try:
            results = self.pipeline(audio_path)
            c50 = np.mean(results["c50"])
            return c50
        except Exception:
            return -420.69

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        score_clean = self.score_audio(target_audio_path)
        score_dirty = self.score_audio(source_audio_path)
        # positive difference -> the dirty sample is worse quality
        return score_clean - score_dirty

    def explain_score(self):
        return super().explain_score(self.name)
