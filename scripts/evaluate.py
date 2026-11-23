#!/usr/bin/env python3
"""Avaliação honesta do classificador clássico no dataset EGFxSet.

Separa estritamente as casas de calibração das casas de teste (nunca avalia
nas casas usadas para calibrar). Gera relatório completo em docs/evaluation.md.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from guitar_string_identifier.calibration.calibrator import GuitarCalibrator
from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.profiles.store import ProfileStore
from guitar_string_identifier.sources.dataset_source import (
    EGFxSetLabelProvider,
    list_egfxset_files,
)
from guitar_string_identifier.sources.file_source import FileAudioSource


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Avalia o identificador no EGFxSet com separação calibração/teste por casa."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/egfxset"),
        help="Diretório raiz do dataset EGFxSet (padrão: data/egfxset).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="egfxset",
        help="Nome do perfil base em configs/profiles/ (padrão: egfxset).",
    )
    parser.add_argument(
        "--cal-frets",
        type=int,
        nargs="+",
        default=[0, 12],
        help="Casas usadas EXCLUSIVAMENTE para calibração (padrão: 0 12).",
    )
    parser.add_argument(
        "--pickup",
        type=str,
        default=None,
        help="Filtrar por um captador específico (padrão: todos os 5 captadores).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/evaluation.md"),
        help="Caminho do relatório Markdown gerado (padrão: docs/evaluation.md).",
    )
    parser.add_argument(
        "--save-profile",
        action="store_true",
        help="Salva a calibração obtida nas casas de calibração no perfil YAML.",
    )
    args = parser.parse_args()

    store = ProfileStore("configs/profiles")
    profile = store.load(args.profile)

    items = list_egfxset_files(args.dataset_dir, effect="Clean", pickup=args.pickup)
    if not items:
        print(f"Nenhum arquivo WAV encontrado em {args.dataset_dir}/Clean.")
        return 1

    cal_fret_set = set(args.cal_frets)
    cal_items = [(p, lbl) for p, lbl in items if lbl.fret in cal_fret_set]
    test_items = [(p, lbl) for p, lbl in items if lbl.fret not in cal_fret_set]

    print(
        f"Total: {len(items)} arquivos | Calibração (casas {sorted(cal_fret_set)}): "
        f"{len(cal_items)} | Teste (casas fora da calibração): {len(test_items)}"
    )

    # 1. Calibração estritamente nas casas de calibração
    calibrator = GuitarCalibrator(analysis_config=profile.analysis)
    cal_notes: list[tuple[Note, int, int]] = []
    for path, lbl in cal_items:
        audio, sr = FileAudioSource(path).read()
        cal_notes.append((Note(audio=audio, sr=sr, source_path=str(path)), lbl.string, lbl.fret))

    cal_report = calibrator.calibrate(
        cal_notes,
        notes_label=(
            f"Calibrado em EGFxSet Clean (casas {sorted(cal_fret_set)}, "
            f"captador={args.pickup or 'todos'})."
        ),
    )
    profile.calibration = cal_report.calibration
    print(
        f"Calibração concluída: {cal_report.accepted_count}/{len(cal_items)} notas aceitas "
        f"({cal_report.rejected_count} rejeitadas)."
    )
    for s in sorted(profile.calibration.b0_by_string):
        print(f"  Corda {s}: B0 = {profile.calibration.b0_by_string[s]:.3e}")
    print(f"  sigma(log B) = {profile.calibration.log_b_sigma:.4f}")

    if args.save_profile:
        store.save(profile)
        print(f"Perfil '{profile.name}' atualizado em configs/profiles/{profile.name}.yaml.")

    # 2. Avaliação estritamente nas casas de teste (nunca vistas na calibração)
    pipeline = GuitarIdentifierPipeline(
        profile,
        label_provider=EGFxSetLabelProvider(tuning=profile.instrument.tuning),
        whole_file=True,
    )

    records: list[dict] = []
    for path, lbl in test_items:
        res = pipeline.analyze_file(path)
        top1 = res.candidates[0] if res.candidates else None
        n_cands = len(res.candidates)
        top1_correct = (
            top1 is not None and top1.string == lbl.string and top1.fret == lbl.fret
        )
        decided_correct = (
            not res.indeterminate
            and res.string == lbl.string
            and res.fret == lbl.fret
        )
        note_correct = res.note == lbl.note
        records.append(
            {
                "path": str(path.relative_to(args.dataset_dir)),
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
                "n_candidates": n_cands,
                "top1_correct": top1_correct,
                "decided_correct": decided_correct,
                "note_correct": note_correct,
                "b_meas": res.features.get("B") if res.features else None,
                "b_res": res.features.get("B_residual_cents") if res.features else None,
            }
        )

    # Métricas globais
    n_total = len(records)
    n_top1_ok = sum(1 for r in records if r["top1_correct"])
    n_note_ok = sum(1 for r in records if r["note_correct"])
    decided = [r for r in records if not r["indeterminate"]]
    n_decided_ok = sum(1 for r in decided if r["decided_correct"])

    # Apenas notas ambíguas (>= 2 posições possíveis na guitarra)
    ambig = [r for r in records if r["n_candidates"] >= 2]
    ambig_top1_ok = sum(1 for r in ambig if r["top1_correct"])
    ambig_decided = [r for r in ambig if not r["indeterminate"]]
    ambig_decided_ok = sum(1 for r in ambig_decided if r["decided_correct"])

    # Matriz de confusão (corda real x corda prevista top-1)
    conf_matrix: dict[int, Counter[int | None]] = {s: Counter() for s in range(1, 7)}
    for r in records:
        conf_matrix[r["true_string"]][r["top1_string"]] += 1

    # Acurácia por corda
    by_string: dict[int, list[dict]] = defaultdict(list)
    by_fret: dict[int, list[dict]] = defaultdict(list)
    by_pickup: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_string[r["true_string"]].append(r)
        by_fret[r["true_fret"]].append(r)
        by_pickup[r["pickup"]].append(r)

    # Curva confiança x erro (apenas notas ambíguas para não inflar com casos triviais)
    conf_bins = [
        (0.0, 0.50),
        (0.50, 0.65),
        (0.65, 0.80),
        (0.80, 0.90),
        (0.90, 0.98),
        (0.98, 1.01),
    ]

    # Destaque: 5 posições do Dó4
    c4_records = [r for r in records if r["true_note"] == "C4"]
    c4_by_pos: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in c4_records:
        c4_by_pos[(r["true_string"], r["true_fret"])].append(r)

    # Imprime resumo no terminal
    print("\n=== RESUMO DA AVALIAÇÃO (CASAS FORA DA CALIBRAÇÃO) ===")
    print(
        f"Acurácia de nota (f0):                {n_note_ok}/{n_total} "
        f"({100 * n_note_ok / n_total:.1f}%)"
    )
    print(
        f"Acurácia top-1 geral (corda+casa):    {n_top1_ok}/{n_total} "
        f"({100 * n_top1_ok / n_total:.1f}%)"
    )
    print(
        f"Acurácia decidido (c/ abstenção):     {n_decided_ok}/{len(decided)} "
        f"({100 * n_decided_ok / max(1, len(decided)):.1f}%) | "
        f"cobertura {100 * len(decided) / n_total:.1f}%"
    )
    print(
        f"Acurácia top-1 em notas AMBÍGUAS:     {ambig_top1_ok}/{len(ambig)} "
        f"({100 * ambig_top1_ok / max(1, len(ambig)):.1f}%)"
    )
    print(
        f"Acurácia decidido em notas AMBÍGUAS:  {ambig_decided_ok}/{len(ambig_decided)} "
        f"({100 * ambig_decided_ok / max(1, len(ambig_decided)):.1f}%) | "
        f"cobertura {100 * len(ambig_decided) / max(1, len(ambig)):.1f}%"
    )

    # Gera relatório Markdown
    lines: list[str] = []
    lines.append("# Relatório de Avaliação — EGFxSet (Clean)\n")
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"_Gerado em {ts_str} por `scripts/evaluate.py`._\n")
    lines.append("## 1. Protocolo experimental (separação calibração/teste)\n")
    lines.append(
        "- **Dataset:** EGFxSet (`Clean`, 48 kHz / 24 bits mono, 5 posições de captador).\n"
        f"- **Casas usadas EXCLUSIVAMENTE na calibração:** `{sorted(cal_fret_set)}` "
        f"({len(cal_items)} arquivos; {cal_report.accepted_count} aceitos, "
        f"{cal_report.rejected_count} rejeitados pelo critério de resíduo).\n"
        f"- **Casas avaliadas (nunca vistas na calibração):** todas as demais casas "
        f"({n_total} arquivos no total; {len(ambig)} em regiões ambíguas com ≥ 2 posições).\n"
        "- **Algoritmo:** 100% processamento clássico de sinais (YIN + FFT longa com "
        "interpolação parabólica + ajuste IRLS de $B$ + taxa de decaimento STFT + "
        "classificador gaussiano / Mahalanobis).\n"
    )

    lines.append("## 2. Parâmetros calibrados (somente nas casas de calibração)\n")
    lines.append(
        "| Corda | Nota solta | $B_0$ estimado | Decaimento base (dB/s) | "
        "Inclinação decaimento (dB/s/casa) |"
    )
    lines.append("|---:|:---:|---:|---:|---:|")
    for s in range(1, 7):
        b0 = profile.calibration.b0_by_string.get(s)
        d0 = profile.calibration.decay_intercept_by_string.get(s)
        ds = profile.calibration.decay_slope_by_string.get(s)
        lines.append(
            f"| {s} | {profile.instrument.tuning[s]} | "
            f"{f'{b0:.3e}' if b0 else '—'} | "
            f"{f'{d0:.2f}' if d0 is not None else '—'} | "
            f"{f'{ds:+.3f}' if ds is not None else '—'} |"
        )
    lines.append(
        f"\n- Desvio padrão empírico calibrado: "
        f"$\\sigma(\\ln B) = {profile.calibration.log_b_sigma:.4f}$, "
        f"$\\sigma(\\text{{decaimento}}) = {profile.calibration.decay_sigma:.2f}$ dB/s.\n"
    )

    lines.append("## 3. Resultados globais\n")
    lines.append("| Métrica | Acertos / Total | Taxa |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Identificação da nota (f0) | {n_note_ok} / {n_total} | "
        f"**{100 * n_note_ok / n_total:.1f}%** |"
    )
    lines.append(
        f"| Corda + casa (Top-1 forçado, todas as notas) | {n_top1_ok} / {n_total} | "
        f"**{100 * n_top1_ok / n_total:.1f}%** |"
    )
    dec_rate = 100 * n_decided_ok / max(1, len(decided))
    lines.append(
        f"| Corda + casa (Decidido, com abstenção) | {n_decided_ok} / {len(decided)} "
        f"(cobertura {100 * len(decided) / n_total:.1f}%) | **{dec_rate:.1f}%** |"
    )
    lines.append(
        f"| Corda + casa (Top-1, **apenas notas ambíguas ≥ 2 posições**) | "
        f"{ambig_top1_ok} / {len(ambig)} | **{100 * ambig_top1_ok / max(1, len(ambig)):.1f}%** |"
    )
    amb_dec_rate = 100 * ambig_decided_ok / max(1, len(ambig_decided))
    lines.append(
        f"| Corda + casa (Decidido, **apenas notas ambíguas ≥ 2 posições**) | "
        f"{ambig_decided_ok} / {len(ambig_decided)} "
        f"(cobertura {100 * len(ambig_decided) / max(1, len(ambig)):.1f}%) | "
        f"**{amb_dec_rate:.1f}%** |"
    )
    lines.append("")

    lines.append("## 4. Matriz de confusão (corda real × corda prevista Top-1)\n")
    lines.append(
        "| Real \\ Prevista | Corda 1 | Corda 2 | Corda 3 | Corda 4 | "
        "Corda 5 | Corda 6 | Sem cand. |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s_true in range(1, 7):
        row = [str(conf_matrix[s_true][s_pred]) for s_pred in range(1, 7)]
        none_cnt = conf_matrix[s_true][None]
        lines.append(f"| **Corda {s_true}** | " + " | ".join(row) + f" | {none_cnt} |")
    lines.append("")

    lines.append("## 5. Acurácia por corda (casas fora da calibração)\n")
    lines.append(
        "| Corda | Nota solta | Top-1 (todas) | Top-1 (ambíguas) | "
        "Decidido (todas) | Cobertura |"
    )
    lines.append("|---:|:---:|---:|---:|---:|---:|")
    for s in range(1, 7):
        rs = by_string[s]
        rs_amb = [r for r in rs if r["n_candidates"] >= 2]
        rs_dec = [r for r in rs if not r["indeterminate"]]
        t1 = 100 * sum(r["top1_correct"] for r in rs) / max(1, len(rs))
        t1_amb = 100 * sum(r["top1_correct"] for r in rs_amb) / max(1, len(rs_amb))
        dec_acc = 100 * sum(r["decided_correct"] for r in rs_dec) / max(1, len(rs_dec))
        cov = 100 * len(rs_dec) / max(1, len(rs))
        lines.append(
            f"| {s} | {profile.instrument.tuning[s]} | {t1:.1f}% ({len(rs)}) | "
            f"{t1_amb:.1f}% ({len(rs_amb)}) | {dec_acc:.1f}% | {cov:.1f}% |"
        )
    lines.append("")

    lines.append("## 6. Acurácia por captador\n")
    lines.append("| Captador | Top-1 (todas) | Top-1 (ambíguas) | Decidido (ambíguas) |")
    lines.append("|---|---:|---:|---:|")
    for pk in sorted(by_pickup.keys()):
        rp = by_pickup[pk]
        rp_amb = [r for r in rp if r["n_candidates"] >= 2]
        rp_amb_dec = [r for r in rp_amb if not r["indeterminate"]]
        a1 = 100 * sum(r["top1_correct"] for r in rp) / max(1, len(rp))
        a2 = 100 * sum(r["top1_correct"] for r in rp_amb) / max(1, len(rp_amb))
        a3 = 100 * sum(r["decided_correct"] for r in rp_amb_dec) / max(1, len(rp_amb_dec))
        lines.append(f"| {pk} | {a1:.1f}% | {a2:.1f}% | {a3:.1f}% |")
    lines.append("")

    lines.append("## 7. Acurácia por casa (somente casas de teste)\n")
    lines.append("| Casa | Notas | Top-1 (todas) | Top-1 (ambíguas) |")
    lines.append("|---:|---:|---:|---:|")
    for fret in sorted(by_fret.keys()):
        rf = by_fret[fret]
        rf_amb = [r for r in rf if r["n_candidates"] >= 2]
        a_all = 100 * sum(r["top1_correct"] for r in rf) / max(1, len(rf))
        a_amb = (
            f"{100 * sum(r['top1_correct'] for r in rf_amb) / len(rf_amb):.1f}%"
            if rf_amb
            else "—"
        )
        lines.append(f"| {fret} | {len(rf)} | {a_all:.1f}% | {a_amb} |")
    lines.append("")

    lines.append("## 8. Curva Confiança × Erro (notas ambíguas)\n")
    lines.append("| Faixa de confiança | Quantidade | Taxa de acerto Top-1 | Taxa de erro Top-1 |")
    lines.append("|---|---:|---:|---:|")
    for lo, hi in conf_bins:
        bucket = [r for r in ambig if lo <= r["confidence"] < hi]
        if not bucket:
            continue
        acc = 100 * sum(r["top1_correct"] for r in bucket) / len(bucket)
        err = 100.0 - acc
        hi_disp = min(1.0, hi)
        lines.append(f"| `[{lo:.2f}, {hi_disp:.2f}]` | {len(bucket)} | {acc:.1f}% | {err:.1f}% |")
    lines.append("")

    lines.append("## 9. Destaque: as 5 posições do Dó4 (C4 ≈ 261,63 Hz)\n")
    lines.append(
        "O Dó4 pode ser tocado em 5 posições distintas na afinação padrão: "
        "`(2, 1)`, `(3, 5)`, `(4, 10)`, `(5, 15)` e `(6, 20)` — nenhuma delas usada "
        "na calibração (casas `0` e `12`).\n"
    )
    lines.append(
        "| Posição (corda, casa) | $B$ médio medido | Acertos Top-1 (5 captadores) | "
        "Decididos corretos | Principais confusões |"
    )
    lines.append("|---:|---:|---:|---:|---|")
    for pos in [(2, 1), (3, 5), (4, 10), (5, 15), (6, 20)]:
        rp = c4_by_pos.get(pos, [])
        if not rp:
            continue
        b_vals = [r["b_meas"] for r in rp if r["b_meas"]]
        b_avg = f"{sum(b_vals)/len(b_vals):.2e}" if b_vals else "—"
        ok_t1 = sum(r["top1_correct"] for r in rp)
        ok_dec = sum(r["decided_correct"] for r in rp)
        errs = [
            f"({r['top1_string']},{r['top1_fret']})"
            for r in rp
            if not r["top1_correct"]
        ]
        err_str = ", ".join(errs) if errs else "nenhuma"
        lines.append(
            f"| ({pos[0]}, {pos[1]}) | `{b_avg}` | {ok_t1} / {len(rp)} | "
            f"{ok_dec} / {len(rp)} | {err_str} |"
        )
    lines.append(
        "\n> **Observação física:** Na Stratocaster com encordoamento padrão (cordas 1–3 lisas "
        "e 4–6 encapadas), a corda 3 (G3 lisa) possui rigidez $B_0$ maior que a corda 4 (D3 "
        "encapada de núcleo fino). Por isso, em `(3, 5)` e `(4, 10)` os valores de $B$ ficam "
        "muito próximos (~2,0×10⁻⁴), criando uma zona fisicamente ambígua onde características "
        "complementares e o mecanismo de abstenção (`indeterminado`) são essenciais.\n"
    )

    lines.append("## 10. Como reproduzir\n")
    lines.append("```bash")
    lines.append("# 1. Baixar os áudios limpos do EGFxSet (se ainda não baixou)")
    lines.append("uv run python scripts/download_egfxset.py")
    lines.append("")
    lines.append("# 2. Rodar a avaliação com separação calibração/teste por casa")
    cal_str = " ".join(str(f) for f in sorted(cal_fret_set))
    lines.append(f"uv run python scripts/evaluate.py --cal-frets {cal_str} --save-profile")
    lines.append("```\n")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRelatório salvo em {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
