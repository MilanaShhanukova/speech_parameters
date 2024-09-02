from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

from base import Metric
from utils import resample_audio


class SignalDistortionMetric(Metric):
    name = "Scale-Invariant-Signal-Distortion-Ratio"

    def __init__(self, sr=16000):
        """
        paper
        """
        super().__init__()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        audio_ref = resample_audio(source_audio_path)
        audio_deg = resample_audio(target_audio_path)

        audio_ref = audio_ref.squeeze()
        audio_deg = audio_deg.squeeze()[: audio_ref.shape[0]]

        if audio_ref.shape[0] > audio_deg.shape[0]:
            audio_ref = audio_ref[: audio_deg.shape[0]]
        sdr_score = self.si_sdr(audio_deg, audio_ref)
        return sdr_score

    def score_audio(self, audio_path: str):
        raise "The metric does not support scoring without reference audio"

    def explain_score(self):
        return super().explain_score(self.name)
