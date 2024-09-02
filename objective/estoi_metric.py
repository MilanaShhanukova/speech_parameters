from pystoi import stoi

from base import Metric
from utils import resample_audio


class ESTOIMetric(Metric):
    name = "ESTOI"

    def __init__(self, sr=16000):
        """
        doi:10.1109/TASLP.2016.2585878
        """
        super().__init__()

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        audio_ref = resample_audio(source_audio_path)
        audio_deg = resample_audio(target_audio_path)

        audio_ref = audio_ref.squeeze().numpy()
        audio_deg = audio_deg.squeeze().numpy()[: audio_ref.shape[0]]

        if audio_ref.shape[0] > audio_deg.shape[0]:
            audio_ref = audio_ref[: audio_deg.shape[0]]

        stoi_score = stoi(audio_ref, audio_deg, 16000, extended=True)
        return stoi_score

    def score_audio(self, audio_path: str):
        raise "The metric does not support scoring without reference audio"

    def explain_score(self):
        return super().explain_score(self.name)
