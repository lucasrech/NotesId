"""Testes unitários e de integração do módulo Auto Tuning com sinais sintéticos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.autotune.adaptive_params import AdaptiveParams
from guitar_string_identifier.autotune.device_probe import DeviceProbe
from guitar_string_identifier.autotune.hum_detector import HumDetector
from guitar_string_identifier.autotune.level_analyzer import LevelAnalyzer
from guitar_string_identifier.autotune.noise_floor import NoiseFloorEstimator
from guitar_string_identifier.autotune.threshold_calibrator import ThresholdCalibrator
from guitar_string_identifier.autotune.tuning_detector import TuningDetector
from guitar_string_identifier.core.music import generate_candidates, midi_to_hz
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.synth.stiff_string import SynthParams, synthesize_stiff_string
from guitar_string_identifier.web.app import create_app


def test_device_probe_ranking_and_channel() -> None:
    probe = DeviceProbe()
    devices = [
        {"deviceId": "d1", "label": "Microfone integrado (Realtek High Definition Audio)"},
        {"deviceId": "d2", "label": "Focusrite Scarlett 2i2 USB"},
        {"deviceId": "d3", "label": "Webcam HD Audio"},
    ]
    ranked = probe.rank_devices(devices)
    assert ranked[0].device_id == "d2"
    assert ranked[0].score > 0
    assert ranked[-1].score < 0

    # Canal ativo em sinal estéreo: canal 0 silêncio, canal 1 com sinal
    stereo = np.zeros((4800, 2), dtype=np.float64)
    stereo[:, 1] = 0.25 * np.sin(2 * np.pi * 220 * np.arange(4800) / 48000)
    ch_res = probe.select_active_channel(stereo)
    assert ch_res.active_channel == 1
    assert ch_res.has_signal is True

    buf = probe.suggest_buffer_size(48000, base_latency_sec=0.01)
    assert buf in (1024, 2048, 4096)


def test_noise_floor_and_60hz_hum_detection_and_exclusion() -> None:
    sr = 48000
    rng = np.random.default_rng(11)
    t = np.arange(int(2.0 * sr)) / sr

    # Ruído branco em ~-50 dBFS + ronco de 60 Hz e harmônicos (120, 180, 240 Hz)
    noise_amp = 10.0 ** (-50.0 / 20.0)
    noise = rng.normal(0.0, noise_amp, size=t.size)
    hum = (
        0.015 * np.sin(2 * np.pi * 60.0 * t)
        + 0.010 * np.sin(2 * np.pi * 120.0 * t)
        + 0.008 * np.sin(2 * np.pi * 180.0 * t)
        + 0.005 * np.sin(2 * np.pi * 240.0 * t)
    )

    nf_only = NoiseFloorEstimator().estimate(noise, sr)
    assert -53.0 <= nf_only.rms_db <= -47.0
    assert nf_only.suggested_onset_threshold_db > nf_only.rms_db + 10.0

    hum_res = HumDetector().detect(noise + hum, sr)
    assert hum_res.detected is True
    assert hum_res.fundamental_hz == 60.0
    assert 60.0 in hum_res.harmonics_hz
    assert 120.0 in hum_res.harmonics_hz
    assert 180.0 in hum_res.harmonics_hz

    # Verifica que InharmonicityExtractor exclui faixas de ronco configuradas
    # Nota em 120 Hz: com notch em [120.0], o parcial n=1 (120 Hz) deve ser excluído
    sig = synthesize_stiff_string(SynthParams(f0=120.0, b_coeff=1.0e-4, sr=sr, duration=1.2))
    inh = InharmonicityExtractor(hum_notch_freqs_hz=[60.0, 120.0, 240.0])
    feats = inh.extract(Note(audio=sig, sr=sr, f0=120.0))
    detected_ns = {p.n for p in feats["partials"]}
    assert 1 not in detected_ns
    assert 2 not in detected_ns
    assert 3 in detected_ns


def test_level_analyzer_clipping_silence_and_chord_rejection() -> None:
    sr = 48000
    analyzer = LevelAnalyzer()

    # 1. Silêncio
    silence = np.zeros(sr, dtype=np.float64)
    q_sil = analyzer.check_input_quality(silence, sr)
    assert q_sil.ok is False and q_sil.is_silence is True

    # 2. Sinal bom (nota isolada)
    clean_note = 0.5 * synthesize_stiff_string(
        SynthParams(f0=261.63, b_coeff=1.5e-4, sr=sr, duration=1.2)
    )
    lvl_good = analyzer.analyze_level(clean_note)
    assert lvl_good.status == "good"
    q_good = analyzer.check_input_quality(clean_note, sr, f0=261.63)
    assert q_good.ok is True

    # 3. Sinal baixo demais
    low_note = 0.02 * clean_note
    assert analyzer.analyze_level(low_note).status == "too_low"

    # 4. Clipping
    clipped = np.clip(clean_note * 6.0, -1.0, 1.0)
    assert analyzer.analyze_level(clipped).status == "clipping"
    q_clip = analyzer.check_input_quality(clipped, sr, f0=261.63)
    assert q_clip.ok is False and q_clip.is_clipping is True

    # 5. Acorde (tríade de Dó maior: C4 + E4 + G4 simultâneas)
    c4 = synthesize_stiff_string(
        SynthParams(f0=261.63, b_coeff=1.0e-4, sr=sr, duration=1.2, seed=1)
    )
    e4 = synthesize_stiff_string(
        SynthParams(f0=329.63, b_coeff=1.0e-4, sr=sr, duration=1.2, seed=2)
    )
    g4 = synthesize_stiff_string(
        SynthParams(f0=392.00, b_coeff=1.0e-4, sr=sr, duration=1.2, seed=3)
    )
    chord = 0.25 * (c4 + e4 + g4)
    q_chord = analyzer.check_input_quality(chord, sr, f0=261.63)
    assert q_chord.ok is False and q_chord.is_polyphonic is True


def test_tuning_detector_drop_d_and_detuned_strings() -> None:
    sr = 48000
    det = TuningDetector()

    # Caso A: Drop D (corda 6 em D2 em vez de E2) com desvio global +20 cents
    drop_d_notes = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 38}  # D2 = MIDI 38
    meas_list = []
    for s, midi in drop_d_notes.items():
        # +20 cents acima de A4=440 Hz
        f0_target = midi_to_hz(midi, a4_hz=440.0) * (2.0 ** (20.0 / 1200.0))
        sig = synthesize_stiff_string(
            SynthParams(f0=f0_target, b_coeff=8e-5, sr=sr, duration=1.2, seed=s)
        )
        m = det.measure_open_string(sig, sr, string=s)
        assert m is not None and m.accepted is True
        meas_list.append(m)

    summary = det.summarize(meas_list)
    assert summary.tuning[6] == "D2"
    assert summary.preset_name == "Drop D (D-A-D-G-B-E)"
    assert summary.is_alternate_tuning is True
    # A4 estimado próximo de 440 * 2^(20/1200) ≈ 445.11 Hz
    assert abs(summary.a4_hz - 445.1) < 2.0

    # Verifica que generate_candidates usa a4_hz e tuning_cents_offset
    cands = generate_candidates(
        midi_to_hz(60, a4_hz=summary.a4_hz),
        tuning=summary.tuning,
        a4_hz=summary.a4_hz,
        tuning_cents_offset=summary.cents_offset_by_string,
    )
    assert any(c.string == 6 and c.fret == 22 for c in cands)


def test_adaptive_params_and_threshold_calibrator() -> None:
    sr = 48000
    ap = AdaptiveParams()
    low_e = synthesize_stiff_string(SynthParams(f0=82.41, b_coeff=1.5e-4, sr=sr, duration=1.5))
    high_e = synthesize_stiff_string(SynthParams(f0=329.63, b_coeff=3e-5, sr=sr, duration=1.5))

    w_low = ap.resolve_window_sec(82.41, sustain_sec=1.2)
    w_high = ap.resolve_window_sec(329.63, sustain_sec=1.2)
    assert w_low > w_high

    t_ms = ap.estimate_transient_ms(low_e, sr)
    assert 15.0 <= t_ms <= 120.0
    assert ap.resolve_max_partials(329.63, sr) >= 12
    assert ap.estimate_sustain_sec(high_e, sr, 30.0) > 0.2

    # Calibrador de limiares para erro alvo <= 10%
    records = [
        {"confidence": 0.95, "margin": 0.80, "correct": True, "n_candidates": 3},
        {"confidence": 0.92, "margin": 0.70, "correct": True, "n_candidates": 3},
        {"confidence": 0.88, "margin": 0.60, "correct": True, "n_candidates": 2},
        {"confidence": 0.85, "margin": 0.55, "correct": True, "n_candidates": 2},
        {"confidence": 0.80, "margin": 0.45, "correct": True, "n_candidates": 3},
        {"confidence": 0.75, "margin": 0.35, "correct": True, "n_candidates": 2},
        {"confidence": 0.72, "margin": 0.30, "correct": True, "n_candidates": 2},
        {"confidence": 0.55, "margin": 0.10, "correct": False, "n_candidates": 2},
        {"confidence": 0.52, "margin": 0.05, "correct": False, "n_candidates": 2},
        {"confidence": 0.50, "margin": 0.02, "correct": False, "n_candidates": 3},
    ]
    th = ThresholdCalibrator().calibrate(records, target_error_rate=0.10)
    assert th.achieved_error_rate <= 0.10
    assert th.min_confidence >= 0.55 or th.min_margin >= 0.15


def test_autotune_api_simulated_flow_never_overwrites_until_confirm(tmp_path: Path) -> None:
    app = create_app(profiles_dir=tmp_path / "profiles", dataset_dir=tmp_path / "data")
    client = TestClient(app)
    sr = 48000

    # 1. Cria sessão
    r_sess = client.post("/api/autotune/session", json={"mode": "live", "profile": "egfxset"})
    assert r_sess.status_code == 200
    sid = r_sess.json()["session_id"]

    # 2. Etapa placa
    r_dev = client.post(
        f"/api/autotune/session/{sid}/device",
        json={"device_name": "Focusrite Scarlett", "sample_rate": sr, "channel": 0},
    )
    assert r_dev.status_code == 200

    # 3. Etapa silêncio com ronco 60 Hz
    t = np.arange(int(1.0 * sr)) / sr
    silence_hum = 0.002 * np.random.default_rng(5).normal(size=t.size) + 0.02 * np.sin(
        2 * np.pi * 60.0 * t
    )
    r_sil = client.post(
        f"/api/autotune/session/{sid}/silence",
        json={"sample_rate": sr, "samples": silence_hum.tolist()},
    )
    assert r_sil.status_code == 200
    assert r_sil.json()["hum"]["detected"] is True

    # 4. Etapa nível
    note_sig = 0.5 * synthesize_stiff_string(
        SynthParams(f0=196.0, b_coeff=1.1e-4, sr=sr, duration=1.0)
    )
    r_lvl = client.post(
        f"/api/autotune/session/{sid}/level",
        json={"samples": note_sig[:24000].tolist()},
    )
    assert r_lvl.status_code == 200
    assert r_lvl.json()["status"] == "good"

    # 5. Proposta pronta — perfil original 'egfxset' ainda NÃO deve ter sido alterado!
    r_prop = client.get(f"/api/autotune/session/{sid}/proposal")
    assert r_prop.status_code == 200
    assert len(r_prop.json()["changes"]) >= 5

    orig = client.get("/api/profiles/egfxset").json()
    assert orig["capture"]["device_name"] is None
    assert orig["param_modes"]["window_sec"] == "manual"

    # 6. Salvar como novo perfil
    r_apply = client.post(
        f"/api/autotune/session/{sid}/apply",
        json={"action": "save_as_new", "new_name": "minha_strato_auto"},
    )
    assert r_apply.status_code == 200
    assert r_apply.json()["active"] == "minha_strato_auto"
    saved = client.get("/api/profiles/minha_strato_auto").json()
    assert saved["capture"]["device_name"] == "Focusrite Scarlett"
    assert saved["param_modes"]["window_sec"] == "auto"
    assert 60.0 in saved["capture"]["hum_notch_freqs_hz"]
