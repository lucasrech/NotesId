"""Testes do gerador sintético de corda rígida e recuperação de f0 e B."""

import pytest

from guitar_string_identifier.analysis.complementary import ComplementaryFeatureExtractor
from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.analysis.pitch import YinPitchEstimator
from guitar_string_identifier.core.music import freq_to_cents
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.synth.stiff_string import SynthParams, synthesize_stiff_string


@pytest.mark.parametrize("f0_target", [82.41, 110.0, 196.0, 261.63, 329.63, 440.0])
def test_yin_subcent_precision_harmonic(f0_target: float) -> None:
    """Valida precisão sub-cent do YIN em corda ideal (B = 0)."""
    sr = 48000
    audio = synthesize_stiff_string(
        SynthParams(f0=f0_target, b_coeff=0.0, sr=sr, duration=1.2, seed=7)
    )
    note = Note(audio=audio, sr=sr, t0=0.0)
    f0_meas = YinPitchEstimator().estimate(note)
    assert f0_meas is not None
    assert abs(freq_to_cents(f0_meas, f0_target)) < 1.5


@pytest.mark.parametrize(
    ("f0_target", "b_target"),
    [
        (110.0, 8.0e-5),   # A2 (corda 5 solta típica)
        (196.0, 1.5e-4),   # G3
        (261.63, 6.0e-5),  # C4 (corda fina, casa baixa)
        (261.63, 4.5e-4),  # C4 (corda intermediária, casa média)
        (261.63, 1.2e-3),  # C4 (corda grossa, casa alta)
    ],
)
def test_synth_f0_and_b_recovery(f0_target: float, b_target: float) -> None:
    sr = 48000
    audio = synthesize_stiff_string(
        SynthParams(f0=f0_target, b_coeff=b_target, sr=sr, duration=1.8, seed=123)
    )
    note = Note(audio=audio, sr=sr, t0=0.0)

    pitch_est = YinPitchEstimator()
    f0_meas = pitch_est.estimate(note)
    assert f0_meas is not None
    # Em cordas rígidas, YIN pode puxar levemente para cima pelos parciais altos,
    # mas permanece dentro do semitom (~40 cents) para iniciar o rastreamento
    assert abs(freq_to_cents(f0_meas, f0_target)) < 40.0

    note.f0 = f0_meas
    inharm = InharmonicityExtractor(window_sec=0.8, max_partials=10)
    feats = inharm.extract(note)

    # f0 refinada pelo ajuste de parciais recupera a fundamental com precisão de poucos cents
    assert abs(freq_to_cents(feats["f0_refined"], f0_target)) < 3.0

    b_meas = feats["B"]
    assert b_meas is not None
    assert feats["n_partials_used"] >= 5
    # B recuperado dentro de 15% de tolerância relativa em relação ao B imposto
    rel_err = abs(b_meas - b_target) / b_target
    assert rel_err < 0.15, f"B esperado={b_target:.2e}, obtido={b_meas:.2e} (erro={rel_err:.1%})"
    assert feats["B_residual_cents"] < 3.0


def test_complementary_features_on_synth() -> None:
    sr = 48000
    audio = synthesize_stiff_string(
        SynthParams(f0=261.63, b_coeff=2.0e-4, sr=sr, duration=1.5, pluck_position=0.25)
    )
    note = Note(audio=audio, sr=sr, t0=0.0, f0=261.63)
    inharm = InharmonicityExtractor().extract(note)
    comp = ComplementaryFeatureExtractor().extract(note, partials=inharm["partials"])

    assert comp["decay_mean_db_s"] is not None
    assert comp["decay_mean_db_s"] > 0.0
    assert comp["spectral_centroid_hz"] > 261.63
    assert comp["spectral_rolloff_hz"] > comp["spectral_centroid_hz"]
