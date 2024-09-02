from pesq import pesq

from base import Metric
from utils import resample_audio


class PESQMetric(Metric):
    name = "PESQ"
    _requires_dependencies = ("jiwer")

    def __init__(self):
        super().__init__()

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        audio_ref = resample_audio(source_audio_path)
        audio_deg = resample_audio(target_audio_path)

        audio_ref = audio_ref.squeeze().numpy()
        audio_deg = audio_deg.squeeze().numpy()[: audio_ref.shape[0]]

        if audio_ref.shape[0] > audio_deg.shape[0]:
            audio_ref = audio_ref[: audio_deg.shape[0]]

        pesq_score = pesq(fs=16000, ref=audio_ref, deg=audio_deg)
        return pesq_score

    def score_audio(self, audio_path: str):
        raise Exception("The metric does not support scoring without reference audio")

    def explain_score(self):
        return super().explain_score(self.name)
