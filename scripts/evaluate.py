#!/usr/bin/env python3
"""Avaliação honesta do classificador clássico no dataset EGFxSet.

Separa estritamente as casas de calibração das casas de teste (nunca avalia
nas casas usadas para calibrar). Suporta comparação Antes × Depois do Auto Tuning
(--compare-autotune) e gera relatório completo em docs/evaluation.md.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from guitar_string_identifier.autotune.session import AutoTuneSession
from guitar_string_identifier.calibration.calibrator import GuitarCalibrator
from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.profiles.store import ProfileStore
from guitar_string_identifier.sources.dataset_source import (
    EGFxSetLabelProvider,
    list_egfxset_files,
)
from guitar_string_identifier.sources.file_source import FileAudioSource


def _run_eval_on_items(
    profile: GuitarProfile,
    test_items: list[tuple[Path, Any]],
    dataset_dir: Path,
) -> list[dict[str, Any]]:
    pipeline = GuitarIdentifierPipeline(
        profile,
        label_provider=EGFxSetLabelProvider(tuning=profile.instrument.tuning),
        whole_file=True,
    )
    records: list[dict[str, Any]] = []
    for path, lbl in test_items:
        res = pipeline.analyze_file(path)
        top1 = res.candidates[0] if res.candidates else None
        records.append(
            {
                "path": str(path.relative_to(dataset_dir)),
                "pickup": lbl.pickup or "?",
                "true_string": lbl.string,
                "true_fret": lbl.fret,
                "true_note": lbl.note,
                "pred_string": res.string,
                "pred_fret": res.fret,
                "top1_string": top1.string if top1 else None,
                "top1_fret": top1.fret if top1 else None,
                "pred_note": res.note,
                "confidence": res.confidence,
                "indeterminate": res.indeterminate,
                "n_candidates": len(res.candidates),
                "top1_correct": (
                    top1 is not None and top1.string == lbl.string and top1.fret == lbl.fret
                ),
                "decided_correct": (
                    not res.indeterminate and res.string == lbl.string and res.fret == lbl.fret
                ),
                "note_correct": res.note == lbl.note,
                "b_meas": res.features.get("B") if res.features else None,
            }
        )
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    note_ok = sum(1 for r in records if r["note_correct"])
    t1_ok = sum(1 for r in records if r["top1_correct"])
    dec = [r for r in records if not r["indeterminate"]]
    dec_ok = sum(1 for r in dec if r["decided_correct"])
    amb = [r for r in records if r["n_candidates"] >= 2]
    amb_t1 = sum(1 for r in amb if r["top1_correct"])
    amb_dec = [r for r in amb if not r["indeterminate"]]
    amb_dec_ok = sum(1 for r in amb_dec if r["decided_correct"])
    return {
        "n": n,
        "note_acc": 100 * note_ok / max(1, n),
        "t1_acc": 100 * t1_ok / max(1, n),
        "dec_acc": 100 * dec_ok / max(1, len(dec)),
        "dec_cov": 100 * len(dec) / max(1, n),
        "amb_n": len(amb),
        "amb_t1_acc": 100 * amb_t1 / max(1, len(amb)),
        "amb_dec_acc": 100 * amb_dec_ok / max(1, len(amb_dec)),
        "amb_dec_cov": 100 * len(amb_dec) / max(1, len(amb)),
        "note_ok": note_ok,
        "t1_ok": t1_ok,
        "dec_ok": dec_ok,
        "dec_n": len(dec),
        "amb_t1_ok": amb_t1,
        "amb_dec_ok": amb_dec_ok,
        "amb_dec_n": len(amb_dec),
    }


def _build_autotune_profile(
    base_profile: GuitarProfile,
    cal_items: list[tuple[Path, Any]],
    target_error_rate: float = 0.10,
) -> tuple[GuitarProfile, list[str]]:
    """Executa Auto Tuning estritamente sobre `cal_items` (sem tocar no teste)."""
    sess = AutoTuneSession(base_profile, mode="dataset")
    loaded = []
    open_seen: set[int] = set()
    for p, lbl in cal_items:
        aud, sr = FileAudioSource(p).read()
        loaded.append((p, lbl, aud, sr))
        if lbl.fret == 0 and lbl.string not in open_seen:
            sess.record_tuning_string(lbl.string, aud, sr)
            open_seen.add(lbl.string)
        sess.record_calibration_note(
            lbl.string, lbl.fret, aud, sr, sample_key=lbl.pickup or str(p)
        )

    # Validação interna por captador (leave-one-pickup-out) APENAS dentro das casas de calibração
    pickups = sorted({lbl.pickup or "default" for _, lbl, _, _ in loaded})
    val_records: list[dict[str, Any]] = []
    if len(pickups) >= 2:
        for held_pk in pickups:
            train_sub = [
                (Note(audio=aud, sr=sr), lbl.string, lbl.fret)
                for _, lbl, aud, sr in loaded
                if (lbl.pickup or "default") != held_pk
            ]
            val_sub = [
                (lbl, aud, sr) for _, lbl, aud, sr in loaded if (lbl.pickup or "default") == held_pk
            ]
            fold_prof = sess.proposed.model_copy(deep=True)
            fold_cal = GuitarCalibrator(analysis_config=fold_prof.analysis).calibrate(train_sub)
            fold_prof.calibration = fold_cal.calibration
            fold_pipe = GuitarIdentifierPipeline(fold_prof, whole_file=True)
            for lbl, aud, sr in val_sub:
                res = fold_pipe.analyze_note(Note(audio=aud, sr=sr))
                if len(res.candidates) >= 2:
                    t1, t2 = res.candidates[0], res.candidates[1]
                    val_records.append(
                        {
                            "confidence": res.confidence,
                            "margin": t1.score - t2.score,
                            "correct": t1.string == lbl.string and t1.fret == lbl.fret,
                            "n_candidates": len(res.candidates),
                        }
                    )

    sess.apply_threshold_calibration(val_records, target_error_rate=target_error_rate)
    prop = sess.build_proposal()
    notes = [f"{c.label}: {c.proposed_value} ({c.how_obtained})" for c in prop.changes]
    return prop.proposed_profile, notes


def main() -> int:
    """Executa avaliação offline no EGFxSet (com separação estrita por casa) e gera relatório."""
    parser = argparse.ArgumentParser(
        description="Avalia o identificador no EGFxSet com separação calibração/teste por casa."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/egfxset"))
    parser.add_argument("--profile", type=str, default="egfxset")
    parser.add_argument("--cal-frets", type=int, nargs="+", default=[0, 12])
    parser.add_argument("--pickup", type=str, default=None)
    parser.add_argument("--report", type=Path, default=Path("docs/evaluation.md"))
    parser.add_argument("--save-profile", action="store_true")
    parser.add_argument(
        "--compare-autotune",
        action="store_true",
        help="Compara Baseline (antes) vs Auto Tuning (depois) nas mesmas casas de teste.",
    )
    args = parser.parse_args()

    store = ProfileStore("configs/profiles")
    base_profile = store.load(args.profile)

    items = list_egfxset_files(args.dataset_dir, effect="Clean", pickup=args.pickup)
    if not items:
        print(f"Nenhum arquivo WAV encontrado em {args.dataset_dir}/Clean.")
        return 1

    cal_fret_set = set(args.cal_frets)
    cal_items = [(p, lbl) for p, lbl in items if lbl.fret in cal_fret_set]
    test_items = [(p, lbl) for p, lbl in items if lbl.fret not in cal_fret_set]

    print(
        f"Total: {len(items)} | Calibração (casas {sorted(cal_fret_set)}): {len(cal_items)} | "
        f"Teste (casas fora da calibração): {len(test_items)}"
    )

    # 1. Baseline (parâmetros manuais padrão, calibrado apenas nas casas de calibração)
    baseline_profile = base_profile.model_copy(deep=True)
    for k in list(baseline_profile.param_modes.keys()):
        baseline_profile.param_modes[k] = "manual"
    baseline_profile.analysis.weight_inharmonicity = 1.0
    baseline_profile.analysis.weight_decay = 0.25
    baseline_profile.analysis.weight_spectral = 0.05
    baseline_profile.analysis.min_confidence = 0.45
    baseline_profile.analysis.min_margin = 0.10
    baseline_profile.instrument.a4_hz = 440.0
    baseline_profile.instrument.tuning_cents_offset = {}

    cal_notes = []
    for path, lbl in cal_items:
        aud, sr = FileAudioSource(path).read()
        cal_notes.append((Note(audio=aud, sr=sr, source_path=str(path)), lbl.string, lbl.fret))

    cal_report = GuitarCalibrator(analysis_config=baseline_profile.analysis).calibrate(
        cal_notes,
        notes_label=f"Calibrado em EGFxSet Clean (casas {sorted(cal_fret_set)}).",
    )
    baseline_profile.calibration = cal_report.calibration
    if args.save_profile:
        store.save(baseline_profile)

    base_records = _run_eval_on_items(baseline_profile, test_items, args.dataset_dir)
    base_sum = _summarize(base_records)

    print("\n=== BASELINE (ANTES DO AUTO TUNING) ===")
    print(
        f"Nota (f0):            {base_sum['note_ok']}/{base_sum['n']} ({base_sum['note_acc']:.1f}%)"
    )
    print(f"Top-1 geral:          {base_sum['t1_ok']}/{base_sum['n']} ({base_sum['t1_acc']:.1f}%)")
    print(
        f"Decidido geral:       {base_sum['dec_ok']}/{base_sum['dec_n']} "
        f"({base_sum['dec_acc']:.1f}%, cob. {base_sum['dec_cov']:.1f}%)"
    )
    print(
        f"Top-1 ambíguas:       {base_sum['amb_t1_ok']}/{base_sum['amb_n']} "
        f"({base_sum['amb_t1_acc']:.1f}%)"
    )

    at_profile: GuitarProfile | None = None
    at_records: list[dict[str, Any]] | None = None
    at_sum: dict[str, Any] | None = None
    at_notes: list[str] = []

    if args.compare_autotune:
        print("\nRodando Auto Tuning (estritamente nas casas de calibração)...")
        at_profile, at_notes = _build_autotune_profile(baseline_profile, cal_items)
        at_records = _run_eval_on_items(at_profile, test_items, args.dataset_dir)
        at_sum = _summarize(at_records)
        print("=== DEPOIS DO AUTO TUNING ===")
        print(
            f"Nota (f0):            {at_sum['note_ok']}/{at_sum['n']} ({at_sum['note_acc']:.1f}%)"
        )
        print(f"Top-1 geral:          {at_sum['t1_ok']}/{at_sum['n']} ({at_sum['t1_acc']:.1f}%)")
        print(
            f"Decidido geral:       {at_sum['dec_ok']}/{at_sum['dec_n']} "
            f"({at_sum['dec_acc']:.1f}%, cob. {at_sum['dec_cov']:.1f}%)"
        )
        print(
            f"Top-1 ambíguas:       {at_sum['amb_t1_ok']}/{at_sum['amb_n']} "
            f"({at_sum['amb_t1_acc']:.1f}%)"
        )

    # Gera relatório Markdown completo
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = [
        "# Relatório de Avaliação — EGFxSet (Clean)\n",
        f"_Gerado em {ts_str} por `scripts/evaluate.py`._\n",
        "## 1. Protocolo experimental (separação calibração/teste)\n",
        (
            "- **Dataset:** EGFxSet (`Clean`, 48 kHz / 24 bits mono, 5 captadores).\n"
            f"- **Casas usadas EXCLUSIVAMENTE na calibração:** `{sorted(cal_fret_set)}` "
            f"({len(cal_items)} arquivos; {cal_report.accepted_count} aceitos).\n"
            f"- **Casas avaliadas (nunca vistas na calibração):** 21 casas restantes "
            f"({base_sum['n']} arquivos; {base_sum['amb_n']} em regiões ambíguas ≥ 2 posições).\n"
            "- **Sem vazamento:** pesos (Fisher), limiares (leave-one-pickup-out interno) e "
            "parâmetros de afinação do Auto Tuning são estimados **somente** nas casas de "
            "calibração.\n"
        ),
    ]

    if at_sum and at_profile and at_records:
        lines.append("## 2. Comparação Antes × Depois do Auto Tuning\n")
        lines.append("| Métrica | Antes (Baseline) | Depois (Auto Tuning) | Variação |")
        lines.append("|---|---:|---:|---:|")

        def _delta(a: float, b: float) -> str:
            d = b - a
            return f"**{d:+.1f} pp**"

        lines.append(
            f"| Identificação da nota (f0) | {base_sum['note_acc']:.1f}% | "
            f"{at_sum['note_acc']:.1f}% | {_delta(base_sum['note_acc'], at_sum['note_acc'])} |"
        )
        lines.append(
            f"| Corda + casa — Top-1 geral | {base_sum['t1_acc']:.1f}% | "
            f"{at_sum['t1_acc']:.1f}% | {_delta(base_sum['t1_acc'], at_sum['t1_acc'])} |"
        )
        lines.append(
            f"| Corda + casa — Decidido (c/ abstenção) | "
            f"{base_sum['dec_acc']:.1f}% (cob. {base_sum['dec_cov']:.1f}%) | "
            f"{at_sum['dec_acc']:.1f}% (cob. {at_sum['dec_cov']:.1f}%) | "
            f"{_delta(base_sum['dec_acc'], at_sum['dec_acc'])} |"
        )
        lines.append(
            f"| Apenas notas ambíguas — Top-1 | {base_sum['amb_t1_acc']:.1f}% | "
            f"{at_sum['amb_t1_acc']:.1f}% | "
            f"{_delta(base_sum['amb_t1_acc'], at_sum['amb_t1_acc'])} |"
        )
        lines.append(
            f"| Apenas notas ambíguas — Decidido | "
            f"{base_sum['amb_dec_acc']:.1f}% (cob. {base_sum['amb_dec_cov']:.1f}%) | "
            f"{at_sum['amb_dec_acc']:.1f}% (cob. {at_sum['amb_dec_cov']:.1f}%) | "
            f"{_delta(base_sum['amb_dec_acc'], at_sum['amb_dec_acc'])} |"
        )
        lines.append("")
        lines.append(
            f"- **Parâmetros escolhidos pelo Auto Tuning (só na calibração):** "
            f"pesos B/decaimento/espectral = "
            f"`{at_profile.analysis.weight_inharmonicity} / "
            f"{at_profile.analysis.weight_decay} / "
            f"{at_profile.analysis.weight_spectral}`, "
            f"limiares `min_confidence={at_profile.analysis.min_confidence}`, "
            f"`min_margin={at_profile.analysis.min_margin}`, "
            f"`A4={at_profile.instrument.a4_hz} Hz`, "
            f"parâmetros de análise em modo `auto`.\n"
        )

        # Acurácia por corda Antes x Depois
        lines.append("### Acurácia Top-1 por corda (Antes × Depois)\n")
        lines.append("| Corda | Nota solta | Antes (Baseline) | Depois (Auto Tuning) | Variação |")
        lines.append("|---:|:---:|---:|---:|---:|")
        for s in range(1, 7):
            rb = [r for r in base_records if r["true_string"] == s]
            ra = [r for r in at_records if r["true_string"] == s]
            ab = 100 * sum(r["top1_correct"] for r in rb) / max(1, len(rb))
            aa = 100 * sum(r["top1_correct"] for r in ra) / max(1, len(ra))
            lines.append(
                f"| {s} | {baseline_profile.instrument.tuning[s]} | {ab:.1f}% | "
                f"{aa:.1f}% | {_delta(ab, aa)} |"
            )
        lines.append("")

        # Destaque Dó4 Antes x Depois (com foco em (3,5) x (4,10))
        lines.append("### Destaque: 5 posições do Dó4 (Antes × Depois)\n")
        lines.append(
            "| Posição | Antes Top-1 | Depois Top-1 | Confusões Antes | Confusões Depois |"
        )
        lines.append("|---:|---:|---:|---|---|")
        for pos in [(2, 1), (3, 5), (4, 10), (5, 15), (6, 20)]:
            rb = [r for r in base_records if (r["true_string"], r["true_fret"]) == pos]
            ra = [r for r in at_records if (r["true_string"], r["true_fret"]) == pos]
            ok_b = sum(r["top1_correct"] for r in rb)
            ok_a = sum(r["top1_correct"] for r in ra)
            err_b = (
                ", ".join(
                    f"({r['top1_string']},{r['top1_fret']})" for r in rb if not r["top1_correct"]
                )
                or "nenhuma"
            )
            err_a = (
                ", ".join(
                    f"({r['top1_string']},{r['top1_fret']})" for r in ra if not r["top1_correct"]
                )
                or "nenhuma"
            )
            lines.append(
                f"| ({pos[0]}, {pos[1]}) | {ok_b}/{len(rb)} | {ok_a}/{len(ra)} | "
                f"{err_b} | {err_a} |"
            )
        lines.append("")

    # Matriz de confusão (do melhor/principal perfil avaliado)
    main_records = at_records if at_records is not None else base_records
    conf_matrix: dict[int, Counter[int | None]] = {s: Counter() for s in range(1, 7)}
    for r in main_records:
        conf_matrix[r["true_string"]][r["top1_string"]] += 1

    lines.append("## 3. Matriz de confusão (corda real × corda prevista Top-1)\n")
    lines.append(
        "| Real \\ Prevista | Corda 1 | Corda 2 | Corda 3 | Corda 4 | "
        "Corda 5 | Corda 6 | Sem cand. |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s_true in range(1, 7):
        row = [str(conf_matrix[s_true][s_pred]) for s_pred in range(1, 7)]
        lines.append(
            f"| **Corda {s_true}** | " + " | ".join(row) + f" | {conf_matrix[s_true][None]} |"
        )
    lines.append("")

    # Curva confiança x erro (notas ambíguas)
    amb_main = [r for r in main_records if r["n_candidates"] >= 2]
    lines.append("## 4. Curva Confiança × Erro (notas ambíguas)\n")
    lines.append("| Faixa de confiança | Quantidade | Taxa de acerto Top-1 | Taxa de erro Top-1 |")
    lines.append("|---|---:|---:|---:|")
    for lo, hi in [
        (0.0, 0.50),
        (0.50, 0.65),
        (0.65, 0.80),
        (0.80, 0.90),
        (0.90, 0.98),
        (0.98, 1.01),
    ]:
        bucket = [r for r in amb_main if lo <= r["confidence"] < hi]
        if not bucket:
            continue
        acc = 100 * sum(r["top1_correct"] for r in bucket) / len(bucket)
        lines.append(
            f"| `[{lo:.2f}, {min(1.0, hi):.2f}]` | {len(bucket)} | {acc:.1f}% | {100 - acc:.1f}% |"
        )
    lines.append("")

    lines.append("## 5. Como reproduzir\n")
    lines.append("```bash")
    cal_str = " ".join(str(f) for f in sorted(cal_fret_set))
    lines.append(f"uv run python scripts/evaluate.py --cal-frets {cal_str} --compare-autotune")
    lines.append("```\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRelatório salvo em {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
