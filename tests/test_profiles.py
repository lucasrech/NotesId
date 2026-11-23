"""Testes de leitura, validação e persistência de perfis YAML."""

from pathlib import Path

import pytest

from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.profiles.store import ProfileStore


def test_egfxset_default_profile_loads() -> None:
    store = ProfileStore("configs/profiles")
    assert "egfxset" in store.list_profiles()
    profile = store.load("egfxset")
    assert profile.instrument.num_frets == 22
    assert profile.instrument.scale_length_mm == 648.0
    assert profile.capture.sample_rate == 48000
    assert profile.instrument.tuning[1] == "E4"
    assert profile.instrument.tuning[6] == "E2"


def test_profile_crud(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    p = GuitarProfile(name="minha_strato", description="Teste CRUD")
    p.instrument.scale_length_mm = 647.7
    store.save(p)

    assert store.list_profiles() == ["minha_strato"]
    loaded = store.load("minha_strato")
    assert loaded.instrument.scale_length_mm == 647.7

    dup = store.duplicate("minha_strato", "copia_strato")
    assert dup.name == "copia_strato"
    assert sorted(store.list_profiles()) == ["copia_strato", "minha_strato"]

    renamed = store.rename("copia_strato", "les_paul")
    assert renamed.name == "les_paul"
    assert not store.exists("copia_strato")
    assert store.exists("les_paul")

    store.delete("les_paul")
    assert store.list_profiles() == ["minha_strato"]


def test_invalid_tuning_rejected() -> None:
    with pytest.raises(ValueError):
        GuitarProfile(name="bad", instrument={"tuning": {1: "H9"}})
