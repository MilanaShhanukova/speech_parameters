from pesq import pesq

from base import Metric
from utils import resample_audio


class PESQMetric(Metric):
    name = "PESQ"
    _requires_dependencies = ("pesq",)

    def __init__(self):
        super().__init__()

    def score_pair_audio(self, audio_path: str, source_audio_path: str):
        """
        Calculate PESQ score between two audio files.
        
        :param audio_path: generated, clean, preprocessed audio
        :param source_audio_path: the original audio, dirty or not generated one.
        """
        audio_ref = resample_audio(source_audio_path).squeeze().numpy()
        audio_deg = resample_audio(audio_path).squeeze().numpy()

        # Match lengths of both arrays
        min_length = min(audio_ref.shape[0], audio_deg.shape[0])
        audio_ref = audio_ref[:min_length]
        audio_deg = audio_deg[:min_length]

        # PESQ calculation (wide-band mode with 16kHz sampling rate)
        pesq_score = pesq(fs=16000, ref=audio_ref, deg=audio_deg)
        return pesq_score

    def score_audio(self, audio_path: str):
        raise Exception("The metric does not support scoring without reference audio")

    def explain_score(self):
        return super().explain_score(self.name)
