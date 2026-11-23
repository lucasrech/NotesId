"""Testes de integração da API FastAPI (REST e WebSocket)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from guitar_string_identifier.synth.stiff_string import SynthParams, synthesize_stiff_string
from guitar_string_identifier.web.app import create_app


def _make_wav_bytes(audio: np.ndarray, sr: int = 48000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_web_api_profiles_and_upload(tmp_path: Path) -> None:
    app = create_app(profiles_dir=tmp_path / "profiles", dataset_dir=tmp_path / "data")
    client = TestClient(app)

    # Página principal
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "Identificador de Corda e Casa" in r_idx.text

    # Perfis CRUD
    r_list = client.get("/api/profiles")
    assert r_list.status_code == 200
    assert r_list.json()["active"] == "egfxset"

    r_dup = client.post("/api/profiles/egfxset/duplicate", json={"new_name": "telecaster"})
    assert r_dup.status_code == 200
    assert r_dup.json()["name"] == "telecaster"

    r_act = client.post("/api/profiles/telecaster/activate")
    assert r_act.status_code == 200
    assert r_act.json()["active"] == "telecaster"

    # Upload de áudio sintético (C4)
    audio = synthesize_stiff_string(SynthParams(f0=261.63, b_coeff=2.0e-4, sr=48000, duration=1.0))
    wav_bytes = _make_wav_bytes(audio, 48000)
    r_up = client.post(
        "/api/analyze/upload",
        files={"file": ("minha_nota.wav", wav_bytes, "audio/wav")},
    )
    assert r_up.status_code == 200
    body = r_up.json()
    assert body["note"] == "C4"
    assert len(body["candidates"]) == 5
    assert body["comparison"]["has_label"] is False
    assert body["features"]["B"] is not None
    assert len(body["spectrum"]["freqs"]) > 20

    # Calibração via PCM
    r_plan = client.get("/api/calibration/plan")
    assert r_plan.status_code == 200
    assert len(r_plan.json()["steps"]) == 12

    r_cal = client.post(
        "/api/calibration/measure-pcm",
        json={
            "string": 3,
            "fret": 0,
            "sample_rate": 48000,
            "samples": audio[:36000].tolist(),
        },
    )
    assert r_cal.status_code == 200
    assert r_cal.json()["accepted"] is True

    # Limpeza: renomeia e exclui perfil extra
    r_ren = client.post("/api/profiles/telecaster/rename", json={"new_name": "lespaul"})
    assert r_ren.status_code == 200
    assert client.delete("/api/profiles/lespaul").status_code == 200


def test_live_websocket_level_and_trigger(tmp_path: Path) -> None:
    app = create_app(profiles_dir=tmp_path / "profiles", dataset_dir=tmp_path / "data")
    client = TestClient(app)

    audio = synthesize_stiff_string(SynthParams(f0=196.0, b_coeff=1.2e-4, sr=48000, duration=1.0))
    block = audio[:4096].astype(np.float32).tobytes()

    with client.websocket_connect("/ws/live") as ws:
        ws.send_json({"type": "config", "sample_rate": 48000, "profile": "egfxset"})
        ready = ws.receive_json()
        assert ready["type"] == "ready"

        ws.send_bytes(block)
        lvl = ws.receive_json()
        assert lvl["type"] == "level"
        assert "rms_db" in lvl
        assert "clipping" in lvl
