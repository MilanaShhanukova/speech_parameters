from srmrpy.srmr import srmr
import numpy as np
import torchaudio

from base import Metric


class SRMRMetric(Metric):
    name = "SRMR"
    _requires_dependencies = ("srmrpy",)  # Add missing dependency declaration

    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr  # Actually use the sr parameter

    def score_audio(self, audio_path: str):
        """
        Calculate SRMR (Speech to Reverberation Modulation energy Ratio) score for audio.
        
        :param audio_path: Path to the audio file to evaluate
        :return: SRMR score, or -420.69 if calculation fails
        """
        audio, sr = torchaudio.load(audio_path)

        try:
            if sr != self.sr:
                audio = torchaudio.functional.resample(audio, sr, self.sr)
            
            audio = audio.squeeze().numpy()

            srmr_score, _ = srmr(
                audio,
                self.sr,
                n_cochlear_filters=23,
                low_freq=125,
                min_cf=4,
                max_cf=128,
                fast=True,
                norm=False
            )
            return float(srmr_score)
        except Exception as e:
            print(f"SRMR calculation failed: {str(e)}")
            return -420.69

    def score_pair_audio(self, audio_path: str, source_audio_path: str):
        """
        Compare SRMR scores between two audio files.
        
        :param audio_path: Path to the processed/generated audio
        :param source_audio_path: Path to the original/reference audio
        :return: Difference between source and processed SRMR scores.
                Positive value means the source audio has better quality.
        """
        score_source = self.score_audio(source_audio_path)
        score_processed = self.score_audio(audio_path)
        return score_source - score_processed

    def explain_score(self):
        return super().explain_score(self.name)
