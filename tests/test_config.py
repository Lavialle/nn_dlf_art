"""Every committed config loads; a typo in a key is an error, not a silent default."""
from pathlib import Path

import pytest

from artguide.config import load_config, parse_overrides

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT / "configs")))
def test_every_config_loads(config_file):
    cfg = load_config(config_file)
    assert (ROOT / cfg.data_dir).name in cfg.experiment
    assert isinstance(cfg.lr, float)  # YAML reads 1e-4 as the string "1e-4": write 1.0e-4


def test_typo_in_a_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ndata_dir: data/wikiart_style5\nlearning_rate: 0.1\n", encoding="utf-8")  # not "lr"
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(bad)


def test_hidden_list_becomes_tuple(tmp_path):
    cfg_file = tmp_path / "ok.yaml"
    cfg_file.write_text("name: x\ndata_dir: data/wikiart_style5\nhidden: [512, 256]\n", encoding="utf-8")
    assert load_config(cfg_file).hidden == (512, 256)


def test_overrides_replace_values_and_rename_the_run(tmp_path):
    cfg_file = tmp_path / "ok.yaml"
    cfg_file.write_text("name: mlp_v3\ndata_dir: data/wikiart_style5\ndropout: 0.5\n", encoding="utf-8")
    cfg = load_config(cfg_file, parse_overrides(["dropout=0.2", "hidden=[256]"]))
    assert cfg.dropout == 0.2 and cfg.hidden == (256,)
    assert cfg.name == "mlp_v3-dropout0.2-hidden[256]"
    with pytest.raises(ValueError, match="dropuot"):
        load_config(cfg_file, {"dropuot": 0.2})  # typos in overrides are caught too
