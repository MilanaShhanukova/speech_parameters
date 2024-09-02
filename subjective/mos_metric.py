from base import Metric
from models.nisqa.utils import _loadDatasetsFile, _loadModel, predict


class MOSMetric(Metric):
    name = "MOS"
    _requires_dependencies = []
    model_path = "/metrics_base/models/nisqa/nisqa.tar"

    def __init__(self):
        super().__init__()
        self._load_model()

    def _load_model(self):
        args = {
            "mode": "predict_file",
            "pretrained_model": self.model_path,
            "output_dir": None,
            "tr_bs_val": 1,
            "tr_num_workers": 0,
            "ms_channel": None,
        }
        self.model, self.args = _loadModel(args, dev="cpu")

    def score_audio(self, audio_path: str):
        try:
            self.args["deg"] = audio_path
            self.ds = _loadDatasetsFile(self.args)
            pred, _, _ = predict(self.args, self.model, self.ds, "cpu")
            mos_score = pred["mos_pred"].values[0]
            return mos_score
        except:
            return -420.69

    def score_pair_audio(self, source_audio_path: str, target_audio_path: str):
        score_clean = self.score_audio(target_audio_path)
        score_dirty = self.score_audio(source_audio_path)
        # positive difference -> the dirty sample is of worse quality
        return score_clean - score_dirty

    def explain_score(self):
        return super().explain_score(self.name)


if __name__ == "__main__":
    metric = MOSMetric()
