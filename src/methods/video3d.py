"""Family C: a video model watches the whole clip, and all of it trains.

Same clips, same flip and same training as B01 (Adam 1e-4, plateau, batch
4); only the model changes. B01 looks at 15 photos one at a time and a GRU
strings them together; these see space and time at once.

The clips are video_gru's, read the same way, so a C run and a B run on
the same split see exactly the same clips. The model converts them to the
frame count, size and colours its weights were trained on
(`VideoClassifier.prepare`).
"""
from src.methods import video_gru
from src.models.video_classifier import (PYTORCHVIDEO, VideoClassifier,
                                         known_video)

NAME = "video3d"

READS = ("variant", "processed", "flip", "backbone", "pretrained",
         "same_clips_as")


def _pytorchvideo_problem():
    """Why pytorchvideo cannot be used, or None. Its last release is from
    2022, so it may fail to import with torch 2.13 (23.2)."""
    try:
        import pytorchvideo.models.hub  # noqa: F401
    except ModuleNotFoundError:
        return "needs pytorchvideo: pip install pytorchvideo"
    except Exception as err:
        return f"pytorchvideo does not import: {type(err).__name__}: {err}"
    return None


def check(cfg):
    """Why this config cannot start yet, or None."""
    from src.models.encoder import _hf_cached
    from src.models.hf import WEIGHTS
    from src.models.video_encoder import HF_VIDEO

    name = cfg["backbone"]
    if not known_video(name):
        return f"{name} is not a video model video3d knows"
    if name in PYTORCHVIDEO:
        problem = _pytorchvideo_problem()
        if problem:
            return problem
    if name in HF_VIDEO and cfg["pretrained"]:
        repo = HF_VIDEO[name]
        if not _hf_cached(repo, WEIGHTS[repo]):
            return (f"weights not downloaded: HF_HUB_OFFLINE=0 hf download "
                    f"{repo} --include \"*.json\" --include "
                    f"\"{WEIGHTS[repo]}\"")
    return video_gru.data_problem(cfg)


def datasets(cfg):
    return video_gru.datasets(cfg)


def eval_set(cfg, split_csv, classes=None):
    return video_gru.eval_set(cfg, split_csv, classes)


def model(cfg, train_set, num_classes):
    return VideoClassifier(cfg["backbone"], num_classes,
                           pretrained=cfg["pretrained"])
