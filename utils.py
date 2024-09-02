import json

import torchaudio
import torchaudio.transforms as T


def read_json(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    return data


def load_audio(audio_path: str):
    audio, sr = torchaudio.load(audio_path)

    assert audio.shape[1] == 0, f"Audio is empty in the file {audio_path}"
    return audio


def resample_audio(audio_path, resample_rate=16000):
    audio, sr = torchaudio.load(audio_path)

    assert (
        audio.shape[1] != 0
    ), f"Audio is empty in the file {audio_path}, shape is {audio.shape}"

    resampler = T.Resample(sr, resample_rate)
    resamapled_audio = resampler(audio)
    return resamapled_audio
