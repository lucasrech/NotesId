"""Testes de calibração e classificação em dados sintéticos com separação por casa."""

from guitar_string_identifier.calibration.calibrator import GuitarCalibrator
from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.synth.stiff_string import synthesize_from_guitar_position

SYNTH_B0 = {
    1: 2.5e-5,
    2: 4.5e-5,
    3: 8.5e-5,
    4: 1.2e-4,
    5: 1.7e-4,
    6: 2.2e-4,
}


def test_calibrate_on_open_and_12_evaluate_on_c4_positions() -> None:
    sr = 48000
    profile = GuitarProfile(name="synth_test")
    calibrator = GuitarCalibrator(analysis_config=profile.analysis)

    # Calibração APENAS nas casas 0 e 12
    cal_notes: list[tuple[Note, int, int]] = []
    for string_num in range(1, 7):
        for fret in (0, 12):
            audio, _, _ = synthesize_from_guitar_position(
                string_num, fret, SYNTH_B0, sr=sr, duration=1.5, seed=string_num * 100 + fret
            )
            cal_notes.append((Note(audio=audio, sr=sr), string_num, fret))

    report = calibrator.calibrate(cal_notes)
    assert report.accepted_count == 12
    for s in range(1, 7):
        b0_est = report.calibration.b0_by_string[s]
        assert abs(b0_est - SYNTH_B0[s]) / SYNTH_B0[s] < 0.22

    profile.calibration = report.calibration
    pipeline = GuitarIdentifierPipeline(profile)

    # Avaliação nas 5 posições do Dó4 (casas NÃO usadas na calibração!)
    c4_positions = [(2, 1), (3, 5), (4, 10), (5, 15), (6, 20)]
    hits = 0
    for s, f in c4_positions:
        audio, _, _ = synthesize_from_guitar_position(
            s, f, SYNTH_B0, sr=sr, duration=1.5, seed=900 + s * 10 + f
        )
        res = pipeline.analyze_note(Note(audio=audio, sr=sr))
        assert res.note == "C4"
        assert res.candidates[0].string == s and res.candidates[0].fret == f
        hits += 1

    assert hits == 5
