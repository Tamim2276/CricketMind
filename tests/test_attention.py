"""tools/attention.py: per-class attention from an attention-pooling head."""
import json
import os

import pytest
import torch

from src.models.head import AttnPoolHead, GRUHead
from src.train import DEFAULTS
from tools import attention as A

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def ckpt(tmp_path):
    """A small attention head saved like train.py saves best.pt."""
    feats = tmp_path / "features"
    lines = "clip,class,label,match\n"
    for label, cls in enumerate(CLASSES):
        for k in range(4):
            clip = f"{cls}/v{label}{k}_0.avi"
            lines += f"{clip},{cls},{label},v{label}\n"
            path = feats / "avg" / (clip[:-4] + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.randn(5, 8).half(), str(path))
    split = tmp_path / "split"
    split.mkdir()
    (split / "val.csv").write_text(lines, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")

    cfg = dict(DEFAULTS)
    cfg.update({"method": "cached_head", "head": "attnpool",
                "features": str(feats), "view": "avg", "batch_size": 3,
                "dense": 16})
    torch.manual_seed(0)
    head = AttnPoolHead(8, 3, dense=16)
    path = tmp_path / "best.pt"
    torch.save({"model_state": head.state_dict(), "cfg": cfg,
                "classes": CLASSES}, str(path))
    return str(path), str(split / "val.csv"), tmp_path


def test_each_class_gets_one_row_of_weights_summing_to_one(ckpt):
    path, val_csv, _ = ckpt
    result = A.attention_by_class(path, val_csv, device=torch.device("cpu"))
    assert result["classes"] == CLASSES
    assert result["frames"] == 5
    assert result["clips"] == [4, 4, 4]
    for row in result["weights"]:
        assert len(row) == 5
        assert abs(sum(row) - 1.0) < 1e-5


def test_a_head_without_attention_is_refused(ckpt, tmp_path):
    path, val_csv, _ = ckpt
    saved = torch.load(path, weights_only=False)
    saved["cfg"]["head"] = "gru"
    saved["cfg"]["hidden"] = 16
    saved["model_state"] = GRUHead(8, 3, hidden=16, dense=16).state_dict()
    other = str(tmp_path / "gru.pt")
    torch.save(saved, other)
    with pytest.raises(ValueError):
        A.attention_by_class(other, val_csv, device=torch.device("cpu"))


def test_the_figure_is_drawn_in_both_themes(ckpt):
    path, val_csv, tmp = ckpt
    result = A.attention_by_class(path, val_csv, device=torch.device("cpu"))
    panels = [("author", result), ("grouped", result)]
    for theme in ("light", "dark"):
        out = str(tmp / f"att_{theme}.png")
        A.draw(panels, out, theme=theme)
        assert os.path.getsize(out) > 10_000
