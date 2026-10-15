"""Hugging Face plumbing shared by the VideoMAE and CLIP encoders.

HF_HUB_OFFLINE=1 is set in this machine's .bashrc for another project, so a
model that is not in the cache cannot download itself. Each one is fetched
once with the flag lifted for that one command; after that everything runs
offline. `hf_file` turns a missing file into that command.
"""
import json
from contextlib import contextmanager

__all__ = ["hf_file", "hf_input_spec", "quiet_loading", "WEIGHTS"]

# the weights file each repo actually has. CLIP's official repo ships no
# safetensors; transformers loads its .bin with weights_only, which is safe
WEIGHTS = {
    "MCG-NJU/videomae-base-finetuned-kinetics": "model.safetensors",
    "openai/clip-vit-base-patch16": "pytorch_model.bin",
}


def hf_file(repo, filename):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename)
    except Exception as err:
        weights = WEIGHTS.get(repo, "model.safetensors")
        raise SystemExit(
            f"{repo}: {filename} is not in the local HF cache "
            f"({type(err).__name__}). Download it once with:\n"
            f"  HF_HUB_OFFLINE=0 hf download {repo} --include \"*.json\" "
            f"--include \"*.txt\" --include \"{weights}\"") from err


def hf_input_spec(repo):
    """(size, mean, std) from the repo's preprocessor_config.json."""
    with open(hf_file(repo, "preprocessor_config.json"),
              encoding="utf-8") as fh:
        cfg = json.load(fh)
    size = cfg.get("crop_size", cfg["size"])
    if isinstance(size, dict):
        size = size.get("height", size.get("shortest_edge"))
    return int(size), tuple(cfg["image_mean"]), tuple(cfg["image_std"])


@contextmanager
def quiet_loading():
    """No load report and no progress bar while weights load.

    CLIP's checkpoint holds the text tower too, so loading only the image
    half prints a 25-line table of "UNEXPECTED" keys. Expected; not useful.
    """
    from transformers.utils import logging

    was = logging.get_verbosity()
    bar = logging.is_progress_bar_enabled()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    try:
        yield
    finally:
        logging.set_verbosity(was)
        if bar:
            logging.enable_progress_bar()
