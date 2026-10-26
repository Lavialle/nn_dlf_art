"""Every committed config loads; a typo in a key is an error, not a silent default."""
from pathlib import Path

import pytest

from artguide.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = sorted((ROOT / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("config_file", CONFIG_FILES, ids=lambda p: str(p.relative_to(ROOT / "configs")))
def test_every_config_loads(config_file):
    cfg = load_config(config_file)
    assert (ROOT / cfg.data_dir).name in cfg.experiment


def test_typo_in_a_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ndata_dir: data/wikiart_style5\nlearning_rate: 0.1\n", encoding="utf-8")  # not "lr"
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(bad)


def test_hidden_list_becomes_tuple(tmp_path):
    cfg_file = tmp_path / "ok.yaml"
    cfg_file.write_text("name: x\ndata_dir: data/wikiart_style5\nhidden: [512, 256]\n", encoding="utf-8")
    assert load_config(cfg_file).hidden == (512, 256)
