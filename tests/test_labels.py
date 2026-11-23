"""Testes do LabelProvider para o dataset EGFxSet."""

from pathlib import Path

from guitar_string_identifier.sources.dataset_source import EGFxSetLabelProvider


def test_egfxset_label_parsing() -> None:
    provider = EGFxSetLabelProvider()

    lbl = provider.get_label(Path("data/egfxset/Clean/Bridge/3-5.wav"))
    assert lbl is not None
    assert lbl.string == 3
    assert lbl.fret == 5
    assert lbl.note == "C4"
    assert lbl.pickup == "Bridge"
    assert lbl.effect == "Clean"

    lbl2 = provider.get_label(Path("data/egfxset/Clean/Bridge-Middle/6-20.wav"))
    assert lbl2 is not None
    assert (lbl2.string, lbl2.fret, lbl2.note) == (6, 20, "C4")
    assert lbl2.pickup == "Bridge-Middle"


def test_non_egfxset_filename_returns_none() -> None:
    provider = EGFxSetLabelProvider()
    assert provider.get_label(Path("qualquer_coisa.wav")) is None
    assert provider.get_label(Path("7-0.wav")) is None
