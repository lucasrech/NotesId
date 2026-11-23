# Relatório de Avaliação — EGFxSet (Clean)

_Gerado em 2026-11-23 18:57 por `scripts/evaluate.py`._

## 1. Protocolo experimental (separação calibração/teste)

- **Dataset:** EGFxSet (`Clean`, 48 kHz / 24 bits mono, 5 posições de captador).
- **Casas usadas EXCLUSIVAMENTE na calibração:** `[0, 12]` (60 arquivos; 60 aceitos, 0 rejeitados pelo critério de resíduo).
- **Casas avaliadas (nunca vistas na calibração):** todas as demais casas (630 arquivos no total; 585 em regiões ambíguas com ≥ 2 posições possíveis).
- **Algoritmo:** 100% processamento clássico de sinais (YIN + FFT longa com interpolação parabólica + ajuste IRLS de $B$ + taxa de decaimento STFT + classificador gaussiano / Mahalanobis).

## 2. Parâmetros calibrados (somente nas casas de calibração)

| Corda | Nota solta | $B_0$ estimado | Decaimento base (dB/s) | Inclinação decaimento (dB/s/casa) |
|---:|:---:|---:|---:|---:|
| 1 | E4 | 2.695e-05 | 18.00 | +1.673 |
| 2 | B3 | 3.581e-05 | 13.49 | +1.913 |
| 3 | G3 | 1.139e-04 | 10.10 | +3.074 |
| 4 | D3 | 6.601e-05 | 8.21 | +3.196 |
| 5 | A2 | 1.041e-04 | 11.06 | +2.064 |
| 6 | E2 | 1.513e-04 | 6.10 | +2.794 |

- Desvio padrão empírico calibrado: $\sigma(\ln B) = 0.2577$, $\sigma(\text{decaimento}) = 5.05$ dB/s.

## 3. Resultados globais

| Métrica | Acertos / Total | Taxa |
|---|---:|---:|
| Identificação da nota (f0) | 629 / 630 | **99.8%** |
| Corda + casa (Top-1 forçado, todas as notas) | 536 / 630 | **85.1%** |
| Corda + casa (Decidido, com abstenção) | 531 / 616 (cobertura 97.8%) | **86.2%** |
| Corda + casa (Top-1, **apenas notas ambíguas ≥ 2 posições**) | 491 / 585 | **83.9%** |
| Corda + casa (Decidido, **apenas notas ambíguas ≥ 2 posições**) | 486 / 571 (cobertura 97.6%) | **85.1%** |

## 4. Matriz de confusão (corda real × corda prevista Top-1)

| Real \ Prevista | Corda 1 | Corda 2 | Corda 3 | Corda 4 | Corda 5 | Corda 6 | Sem cand. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Corda 1** | 102 | 3 | 0 | 0 | 0 | 0 | 0 |
| **Corda 2** | 18 | 87 | 0 | 0 | 0 | 0 | 0 |
| **Corda 3** | 1 | 3 | 97 | 4 | 0 | 0 | 0 |
| **Corda 4** | 2 | 6 | 30 | 67 | 0 | 0 | 0 |
| **Corda 5** | 0 | 1 | 4 | 13 | 87 | 0 | 0 |
| **Corda 6** | 0 | 0 | 0 | 1 | 7 | 97 | 0 |

## 5. Acurácia por corda (casas fora da calibração)

| Corda | Nota solta | Top-1 (todas) | Top-1 (ambíguas) | Decidido (todas) | Cobertura |
|---:|:---:|---:|---:|---:|---:|
| 1 | E4 | 97.1% (105) | 96.2% (80) | 97.1% | 100.0% |
| 2 | B3 | 82.9% (105) | 82.9% (105) | 82.9% | 100.0% |
| 3 | G3 | 92.4% (105) | 92.4% (105) | 93.3% | 99.0% |
| 4 | D3 | 63.8% (105) | 63.8% (105) | 66.7% | 88.6% |
| 5 | A2 | 82.9% (105) | 82.9% (105) | 83.7% | 99.0% |
| 6 | E2 | 91.4% (105) | 89.4% (85) | 91.4% | 100.0% |

## 6. Acurácia por captador

| Captador | Top-1 (todas) | Top-1 (ambíguas) | Decidido (ambíguas) |
|---|---:|---:|---:|
| Bridge | 85.7% | 84.6% | 84.5% |
| Bridge-Middle | 85.7% | 84.6% | 87.6% |
| Middle | 89.7% | 88.9% | 90.4% |
| Middle-Neck | 81.7% | 80.3% | 81.4% |
| Neck | 82.5% | 81.2% | 81.7% |

## 7. Acurácia por casa (somente casas de teste)

| Casa | Notas | Top-1 (todas) | Top-1 (ambíguas) |
|---:|---:|---:|---:|
| 1 | 30 | 100.0% | 100.0% |
| 2 | 30 | 100.0% | 100.0% |
| 3 | 30 | 100.0% | 100.0% |
| 4 | 30 | 100.0% | 100.0% |
| 5 | 30 | 93.3% | 93.3% |
| 6 | 30 | 93.3% | 93.3% |
| 7 | 30 | 83.3% | 83.3% |
| 8 | 30 | 90.0% | 90.0% |
| 9 | 30 | 93.3% | 93.3% |
| 10 | 30 | 86.7% | 86.7% |
| 11 | 30 | 90.0% | 90.0% |
| 13 | 30 | 76.7% | 76.7% |
| 14 | 30 | 83.3% | 83.3% |
| 15 | 30 | 63.3% | 63.3% |
| 16 | 30 | 66.7% | 66.7% |
| 17 | 30 | 83.3% | 83.3% |
| 18 | 30 | 90.0% | 88.0% |
| 19 | 30 | 86.7% | 84.0% |
| 20 | 30 | 76.7% | 72.0% |
| 21 | 30 | 73.3% | 68.0% |
| 22 | 30 | 56.7% | 48.0% |

## 8. Curva Confiança × Erro (notas ambíguas)

| Faixa de confiança | Quantidade | Taxa de acerto Top-1 | Taxa de erro Top-1 |
|---|---:|---:|---:|
| `[0.00, 0.50]` | 1 | 0.0% | 100.0% |
| `[0.50, 0.65]` | 51 | 52.9% | 47.1% |
| `[0.65, 0.80]` | 51 | 66.7% | 33.3% |
| `[0.80, 0.90]` | 44 | 84.1% | 15.9% |
| `[0.90, 0.98]` | 65 | 84.6% | 15.4% |
| `[0.98, 1.00]` | 373 | 90.6% | 9.4% |

## 9. Destaque: as 5 posições do Dó4 (C4 ≈ 261,63 Hz)

O Dó4 pode ser tocado em 5 posições distintas na afinação padrão: `(2, 1)`, `(3, 5)`, `(4, 10)`, `(5, 15)` e `(6, 20)` — nenhuma delas usada na calibração (casas `0` e `12`).

| Posição (corda, casa) | $B$ médio medido | Acertos Top-1 (5 captadores) | Decididos corretos | Principais confusões |
|---:|---:|---:|---:|---|
| (2, 1) | `3.38e-05` | 5 / 5 | 5 / 5 | nenhuma |
| (3, 5) | `2.01e-04` | 5 / 5 | 5 / 5 | nenhuma |
| (4, 10) | `1.90e-04` | 2 / 5 | 1 / 5 | (3,5), (3,5), (3,5) |
| (5, 15) | `4.43e-04` | 4 / 5 | 4 / 5 | (3,5) |
| (6, 20) | `1.54e-03` | 4 / 5 | 4 / 5 | (5,15) |

> **Observação física:** Na Stratocaster com encordoamento padrão (cordas 1–3 lisas e 4–6 encapadas), a corda 3 (G3 lisa) possui rigidez $B_0$ maior que a corda 4 (D3 encapada de núcleo fino). Por isso, em `(3, 5)` e `(4, 10)` os valores de $B$ ficam muito próximos (~2,0×10⁻⁴), criando uma zona fisicamente ambígua onde características complementares e o mecanismo de abstenção (`indeterminado`) são essenciais.

## 10. Como reproduzir

```bash
# 1. Baixar os áudios limpos do EGFxSet (se ainda não baixou)
uv run python scripts/download_egfxset.py

# 2. Rodar a avaliação com separação calibração/teste por casa
uv run python scripts/evaluate.py --cal-frets 0 12 --save-profile
```
