from pystoi import stoi

from base import Metric
from utils import resample_audio


class STOIMetric(Metric):
    name = "STOI"

    def __init__(self, sr=16000):
        super().__init__()

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        audio_ref = resample_audio(source_audio_path)
        audio_deg = resample_audio(target_audio_path)

        audio_ref = audio_ref.squeeze().numpy()
        audio_deg = audio_deg.squeeze().numpy()[: audio_ref.shape[0]]

        if audio_ref.shape[0] > audio_deg.shape[0]:
            audio_ref = audio_ref[: audio_deg.shape[0]]

        print(f"shapes: {audio_ref.shape}, {audio_deg.shape}")

        stoi_score = stoi(audio_ref, audio_deg, 16000)
        return stoi_score

    def score_audio(self, audio_path: str):
        raise Exception("The metric does not support scoring without reference audio")

    def explain_score(self):
        return super().explain_score(self.name)
