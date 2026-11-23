# Relatório de Avaliação — EGFxSet (Clean)

_Gerado em 2026-11-23 19:39 por `scripts/evaluate.py`._

## 1. Protocolo experimental (separação calibração/teste)

- **Dataset:** EGFxSet (`Clean`, 48 kHz / 24 bits mono, 5 captadores).
- **Casas usadas EXCLUSIVAMENTE na calibração:** `[0, 12]` (60 arquivos; 60 aceitos).
- **Casas avaliadas (nunca vistas na calibração):** 21 casas restantes (630 arquivos; 585 em regiões ambíguas ≥ 2 posições).
- **Sem vazamento:** pesos (Fisher), limiares (leave-one-pickup-out interno) e parâmetros de afinação do Auto Tuning são estimados **somente** nas casas de calibração.

## 2. Comparação Antes × Depois do Auto Tuning

| Métrica | Antes (Baseline) | Depois (Auto Tuning) | Variação |
|---|---:|---:|---:|
| Identificação da nota (f0) | 99.8% | 99.8% | **+0.0 pp** |
| Corda + casa — Top-1 geral | 85.1% | 87.0% | **+1.9 pp** |
| Corda + casa — Decidido (c/ abstenção) | 86.2% (cob. 97.8%) | 90.4% (cob. 92.4%) | **+4.2 pp** |
| Apenas notas ambíguas — Top-1 | 83.9% | 86.0% | **+2.1 pp** |
| Apenas notas ambíguas — Decidido | 85.1% (cob. 97.6%) | 89.6% (cob. 91.8%) | **+4.5 pp** |

- **Parâmetros escolhidos pelo Auto Tuning (só na calibração):** pesos B/decaimento/espectral = `1.0 / 0.05 / 0.079`, limiares `min_confidence=0.53`, `min_margin=0.05`, `A4=440.91 Hz`, parâmetros de análise em modo `auto`.

### Acurácia Top-1 por corda (Antes × Depois)

| Corda | Nota solta | Antes (Baseline) | Depois (Auto Tuning) | Variação |
|---:|:---:|---:|---:|---:|
| 1 | E4 | 97.1% | 98.1% | **+1.0 pp** |
| 2 | B3 | 82.9% | 82.9% | **+0.0 pp** |
| 3 | G3 | 92.4% | 93.3% | **+1.0 pp** |
| 4 | D3 | 63.8% | 59.0% | **-4.8 pp** |
| 5 | A2 | 82.9% | 93.3% | **+10.5 pp** |
| 6 | E2 | 91.4% | 95.2% | **+3.8 pp** |

### Destaque: 5 posições do Dó4 (Antes × Depois)

| Posição | Antes Top-1 | Depois Top-1 | Confusões Antes | Confusões Depois |
|---:|---:|---:|---|---|
| (2, 1) | 5/5 | 5/5 | nenhuma | nenhuma |
| (3, 5) | 5/5 | 5/5 | nenhuma | nenhuma |
| (4, 10) | 2/5 | 2/5 | (3,5), (3,5), (3,5) | (3,5), (3,5), (3,5) |
| (5, 15) | 4/5 | 4/5 | (3,5) | (4,10) |
| (6, 20) | 4/5 | 5/5 | (5,15) | nenhuma |

## 3. Matriz de confusão (corda real × corda prevista Top-1)

| Real \ Prevista | Corda 1 | Corda 2 | Corda 3 | Corda 4 | Corda 5 | Corda 6 | Sem cand. |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Corda 1** | 103 | 2 | 0 | 0 | 0 | 0 | 0 |
| **Corda 2** | 18 | 87 | 0 | 0 | 0 | 0 | 0 |
| **Corda 3** | 2 | 0 | 98 | 5 | 0 | 0 | 0 |
| **Corda 4** | 1 | 4 | 33 | 62 | 4 | 1 | 0 |
| **Corda 5** | 0 | 2 | 2 | 3 | 98 | 0 | 0 |
| **Corda 6** | 0 | 0 | 0 | 1 | 3 | 101 | 0 |

## 4. Curva Confiança × Erro (notas ambíguas)

| Faixa de confiança | Quantidade | Taxa de acerto Top-1 | Taxa de erro Top-1 |
|---|---:|---:|---:|
| `[0.00, 0.50]` | 2 | 50.0% | 50.0% |
| `[0.50, 0.65]` | 147 | 72.8% | 27.2% |
| `[0.65, 0.80]` | 23 | 73.9% | 26.1% |
| `[0.80, 0.90]` | 4 | 75.0% | 25.0% |
| `[0.90, 0.98]` | 33 | 87.9% | 12.1% |
| `[0.98, 1.00]` | 376 | 92.0% | 8.0% |

## 5. Como reproduzir

```bash
uv run python scripts/evaluate.py --cal-frets 0 12 --compare-autotune
```

> **Análise honesta da confusão `(3, 5) × (4, 10)` (Dó4) e da Corda 4:**
> O Auto Tuning elevou a acurácia Top-1 geral (+1,9 pp, de 85,1% para 87,0%), a acurácia decidida (+4,2 pp, atingindo 90,4% — erro de 9,6%, dentro do alvo de ≤ 10%) e o Dó4 de 20/25 (80%) para 21/25 (84%), com ganhos expressivos nas cordas 5 (+10,5 pp) e 6 (+3,8 pp). Entretanto, a **corda 4 caiu 4,8 pp (63,8% → 59,0%)** e a confusão específica `(4, 10) → (3, 5)` permaneceu em 2/5. Isso ocorre porque o critério de Fisher nas casas 0 e 12 atribuiu peso menor ao decaimento (`0,05` vs `0,25` manual), enquanto os valores previstos de $B$ em `(3, 5)` e `(4, 10)` diferem em apenas ~3% diante de uma dispersão empírica de ~26%. Sem características adicionais específicas por captador (ou abstenção), o cruzamento físico entre a corda 3 lisa e a corda 4 encapada continua sendo o principal gargalo.
