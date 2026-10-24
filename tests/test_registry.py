"""The registry: 54 experiments as data, checked before anything runs."""
import io
import os

import pytest

from src import methods
from src.experiments import registry as R
from src.experiments import run as runner
from src.train import DEFAULTS, LOOP_READS, load_config

COUNTS = {"P": 5, "A": 14, "D": 6, "V": 10, "B": 8, "C": 8, "R": 3}


def test_every_delta_key_is_a_config_key():
    # a typo like "droupout" must fail here, not become a row
    for entry in R.ENTRIES:
        for key in entry["cfg"]:
            assert key in DEFAULTS, (entry["id"], key)


def test_every_base_key_is_a_config_key():
    for name, base in R.BASES.items():
        load_config(base["config"])          # raises on unknown keys
        for key in base["set"]:
            assert key in DEFAULTS, (name, key)


def test_ids_are_unique_and_well_formed():
    seen = set()
    for entry in R.ENTRIES:
        assert entry["id"] not in seen, entry["id"]
        seen.add(entry["id"])
        assert R.family_of(entry["id"]) in R.FAMILIES


def test_every_entry_says_why():
    for entry in R.ENTRIES:
        assert entry["why"].strip(), entry["id"]


def test_every_family_has_its_count():
    for fam, n in COUNTS.items():
        assert len(R.entries(fam)) == n, fam
    assert len(R.ENTRIES) == 54


def test_108_run_names_none_repeated():
    names = set()
    for entry in R.ENTRIES:
        for split in R.SPLITS:
            names.add(R.run_name(entry["id"], split))
    assert len(names) == 108


def test_the_harness_owns_name_and_split():
    for entry in R.ENTRIES:
        assert "name" not in entry["cfg"], entry["id"]
        assert "split" not in entry["cfg"], entry["id"]
    for base in R.BASES.values():
        assert "name" not in base["set"]
        assert "split" not in base["set"]


def test_deltas_stay_small():
    for entry in R.ENTRIES:
        assert len(entry["cfg"]) <= R.MAX_DELTA, entry["id"]


def test_every_base_is_used_and_known():
    used = set()
    for entry in R.ENTRIES:
        assert entry["base"] in R.BASES, entry["id"]
        used.add(entry["base"])
    assert used == set(R.BASES)


def test_resolve_sets_name_and_split():
    cfg = R.resolve(R.get("A07_bilstm64"), "grouped")
    assert cfg["name"] == "A07_bilstm64__grouped"
    assert cfg["split"] == "data/splits_grouped"
    assert cfg["head"] == "lstm"
    assert cfg["hidden"] == 64
    assert cfg["bidirectional"] is True
    # untouched keys come from the base yaml
    assert cfg["features"] == "data/features/effnetv2s"


def test_b01_is_mimic_author_exactly():
    # so its author row can be the 78.14% reproduction, not a re-run
    cfg = R.resolve(R.get("B01_effnetv2s"), "author")
    want = dict(DEFAULTS)
    want.update(load_config(os.path.join("configs", "mimic_author.yaml")))
    for key in want:
        if key == "name":
            continue
        assert cfg[key] == want[key], key


def test_a03_is_cached_gru_but_for_the_method():
    cfg = R.resolve(R.get("A03_gru128"), "author")
    want = dict(DEFAULTS)
    want.update(load_config(os.path.join("configs", "cached_gru.yaml")))
    for key in want:
        if key in ("name", "method"):
            continue
        assert cfg[key] == want[key], key


def test_unknown_split_and_id_fail_loudly():
    with pytest.raises(ValueError):
        R.run_name("A01_mean", "random")
    with pytest.raises(KeyError):
        R.get("A99_nothing")
    with pytest.raises(ValueError):
        R.family_of("a01_mean")


def test_methods_declare_what_they_read():
    for name in methods.KNOWN:
        module = methods.load(name)
        assert module.READS, name
        for key in module.READS:
            assert key in DEFAULTS, (name, key)
            assert key not in LOOP_READS, (name, key)


def test_no_built_method_ignores_a_delta_key():
    # cached_gru reads no "backbone": a delta setting one would be a
    # silent no-op that still writes resnet50 into the row
    for entry in R.ENTRIES:
        unread = R.unread_keys(entry)
        if unread is not None:
            assert unread == [], (entry["id"], unread)


def test_unread_key_is_caught(monkeypatch):
    fake = {"id": "B99_fake", "base": "mimic_author",
            "cfg": {"head": "lstm"}, "why": "x"}
    assert R.unread_keys(fake) == ["head"]
    assert R.status(fake) == "video_gru ignores head"


def test_status_names_what_is_missing():
    assert R.status(R.get("C01_r2plus1d_18")) == "needs method video3d"
    assert R.status(R.get("B01_effnetv2s")) == "ready"
    fake = {"id": "B99_fake", "base": "mimic_author",
            "cfg": {"backbone": "no_such_net"}, "why": "x"}
    assert R.status(fake) == "neither torchvision nor timm has no_such_net"


def test_status_spots_a_missing_cache(monkeypatch, tmp_path):
    monkeypatch.setitem(R.BASES, "fake", {
        "config": os.path.join("configs", "mimic_author.yaml"),
        "set": {"method": "cached_gru"}})
    fake = {"id": "P99_fake", "base": "fake",
            "cfg": {"features": str(tmp_path / "nothing")}, "why": "x"}
    assert R.status(fake) == "needs cache nothing"


def test_list_prints_every_id():
    out = io.StringIO()
    shown = runner.print_list(out=out)
    text = out.getvalue()
    assert shown == 54
    for entry in R.ENTRIES:
        assert entry["id"] in text
    assert "54 IDs, 108 runs" in text


def test_list_one_family():
    out = io.StringIO()
    assert runner.print_list("D", out=out) == 6
    assert "A01_mean" not in out.getvalue()


def test_every_b_run_trains_four_clips_at_a_time():
    """The head's BatchNorm needs the four: at batch 2 it turns every feature
    into +-1, and B07 fell from 32% val to 16% (2026-10-22). B07 fits batch 4
    by recomputing its backbone in backward instead."""
    for entry in R.entries("B"):
        cfg = R.resolve(entry, "author")
        assert (cfg["batch_size"], cfg["accum"]) == (4, 1), entry["id"]
    assert R.resolve(R.get("B07_regnety_16gf"), "author")["grad_checkpoint"]
    assert R.status(R.get("B07_regnety_16gf")) == "ready"


def test_every_training_run_uses_deterministic_kernels():
    """17.3: 1.5-1.7x the time end to end, kept anyway (2026-10-20), so a
    run repeats exactly and a resumed run equals an unbroken one."""
    for family in ("P", "A", "V", "B", "C", "R"):
        for entry in R.entries(family):
            assert R.resolve(entry, "author")["deterministic"] is True, \
                entry["id"]


def test_deterministic_does_not_change_what_a_row_means():
    old = {"lr": 1e-4}
    new = {"lr": 1e-4, "deterministic": False, "name": "x"}
    assert R.config_differences(old, new) == []
    new["lr"] = 3e-4
    assert R.config_differences(old, new) == ["lr"]


def test_the_port_is_their_notebook_not_mimic_author():
    """7.7: read from their training notebook, not inferred from the .keras."""
    cfg = R.resolve(R.get("R02_port_box"), "author")
    assert cfg["backbone"] == "tf_efficientnetv2_s.in1k"   # Google's weights
    assert cfg["monitor"] == "loss"
    assert cfg["lr_patience"] == 3                          # = Keras' 4
    assert cfg["adam_eps"] == 1e-7
    assert cfg["keras_dense"] is True
    assert cfg["drop_path"] == 0.2                          # Keras' default
    assert cfg["variant"] == "box"
    assert R.resolve(R.get("R01_port_replica"), "author")["variant"] == "replica"
    # R03 is B01 but for the frames
    r03 = R.resolve(R.get("R03_mimic_replica"), "author")
    b01 = R.resolve(R.get("B01_effnetv2s"), "author")
    for key in b01:
        if key in ("name", "variant"):
            continue
        assert r03[key] == b01[key], key


def test_a_replica_run_waits_for_the_replica_clips(tmp_path):
    told = R.status(R.get("R03_mimic_replica"),
                    overrides={"processed": str(tmp_path)})
    assert told == "needs the replica clips: python -m tools.replica --build --all"


def test_a_few_replica_clips_are_not_a_dataset(tmp_path):
    """Found 2026-10-21: the 300-clip replica sample made R03 read 'ready',
    and the dataset would have trained on just those clips."""
    import shutil

    # box has two train clips, replica one of them
    for variant, clips in (("box", ("vid100_41", "vid100_60")),
                           ("replica", ("vid100_41",))):
        os.makedirs(tmp_path / variant / "Cover Drive")
        for clip in clips:
            shutil.copy(os.path.join("data", "processed", "box", "Cover Drive",
                                     clip + ".npy"),
                        tmp_path / variant / "Cover Drive")
    told = R.status(R.get("R03_mimic_replica"),
                    overrides={"processed": str(tmp_path)})
    assert told.startswith("needs the replica clips (1 of 2 train clips)")
    assert "--build --all" in told
