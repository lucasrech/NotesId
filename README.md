<!--
[![CI](https://github.com/SEU_USUARIO/guitar-string-identifier/actions/workflows/ci.yml/badge.svg)](https://github.com/SEU_USUARIO/guitar-string-identifier/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
-->

# Identificador de Corda e Casa de Guitarra

Sistema completo para identificar **qual corda e qual casa** foram tocadas em uma guitarra a partir do áudio de uma nota isolada, usando **exclusivamente algoritmos clássicos de processamento de sinais (sem machine learning)**.

Uma mesma nota (como o **Dó4 ≈ 261,63 Hz**) pode ser tocada em até **5 posições diferentes** na afinação padrão — `(2, 1)`, `(3, 5)`, `(4, 10)`, `(5, 15)` e `(6, 20)`. O sistema diferencia essas posições pelas propriedades físicas da corda, principalmente o **coeficiente de inharmonicidade $B$** ($f_n = n f_0 \sqrt{1 + B n^2}$), complementado por taxas de decaimento por parcial (STFT), estimativa de posição de palhetada e descritores espectrais de baixo peso.

## Principais recursos

- **Núcleo DSP 100% clássico e modular:** `AudioSource → NoteSegmenter → PitchEstimator (YIN) → FeatureExtractors (B + decaimento + palhetada + espectro) → StringClassifier (Mahalanobis / Gaussiano) → Result`.
- **Gerador sintético de corda rígida:** modelo Karplus-Strong com cascata de filtros *all-pass* de dispersão para validar a recuperação de $f_0$ e $B$ antes de testar em áudio real.
- **Calibração por guitarra:** estima $B_0$ por corda a partir de poucas notas (ex.: cordas soltas + casa 12), rejeita notas com resíduo alto e prevê $B$ para todo o braço via $B(k) = B_0 \cdot 2^{k/6}$.
- **Perfis YAML validados com Pydantic:** 3 blocos configuráveis (Captação, Instrumento, Análise) com CRUD completo na interface web.
- **Interface web em pt-BR (FastAPI + WebSocket + Web Audio API):**
  - Quadro central de nota, corda, casa, confiança e conferência automática de acerto (via `LabelProvider`).
  - Modo **Arquivo / Dataset** (upload WAV/FLAC/OGG + navegador do EGFxSet local).
  - Modo **Ao vivo** direto no navegador via Web Audio API (compatível com WSL2), com seletor de placa de captura, medidor de nível e aviso de clipping.
  - Gráfico do espectro comparando parciais medidos contra harmônicos ideais $n \cdot f_0$.
  - Assistente de **Calibração Guiada** ao vivo ou por arquivos.

## Requisitos e instalação com `uv`

Requer Python 3.12 e `libsndfile1` (no Ubuntu/Debian: `sudo apt install libsndfile1`).

```bash
# Clonar e entrar no diretório
cd ~/Projetos/guitar-string-identifier

# Criar/sincronizar o ambiente virtual com uv
uv sync

# Rodar lint e testes unitários
uv run ruff check .
uv run pytest
```

### Como regenerar `requirements.txt`

```bash
uv export --no-hashes --format requirements-txt -o requirements.txt
```

## Dataset (EGFxSet)

Este projeto utiliza o dataset público **EGFxSet** para calibração e avaliação em áudio real:

- **Fonte:** Zenodo — registro 7044411 ([https://zenodo.org/record/7044411](https://zenodo.org/record/7044411)) | Site: [https://egfxset.github.io](https://egfxset.github.io)
- **Citação:** Pedroza, H., Meza, G., & Roman, I. R. (2022). *EGFxSet: Electric guitar tones processed through real effects of distortion, modulation, delay and reverb*. Zenodo. DOI: `10.5281/zenodo.7044411`.
- **Licença do dataset:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`).

### Download dos áudios limpos

```bash
# Listar arquivos disponíveis no Zenodo e seus tamanhos
uv run python scripts/download_egfxset.py --list

# Baixar apenas os áudios limpos (Clean.zip, ~431 MB) + metadados, com verificação MD5
uv run python scripts/download_egfxset.py
```

Estrutura esperada após extração (ignorada pelo Git em `data/egfxset/`):

```text
data/egfxset/
├── egfxset_metadata.csv
└── Clean/
    ├── Bridge/           (1-0.wav ... 6-22.wav)
    ├── Bridge-Middle/
    ├── Middle/
    ├── Middle-Neck/
    └── Neck/
```

Formato verificado dos arquivos: WAV `48000 Hz`, mono, 24 bits (`PCM_24`), ~5,0 s cada (690 arquivos em `Clean/`).

## Como executar

### 1. Avaliação no EGFxSet (calibração separada do teste)

```bash
uv run python scripts/evaluate.py --cal-frets 0 12 --save-profile
```

O script calibra **exclusivamente** nas casas informadas (`0` e `12`), avalia nas 21 casas restantes (630 arquivos) e grava o relatório completo em [docs/evaluation.md](docs/evaluation.md).

**Resumo dos resultados obtidos (calibrado só nas casas 0 e 12, testado em 630 notas inéditas):**
- **Identificação da nota ($f_0$):** 99,8% (629/630)
- **Corda + casa (Top-1 geral):** 85,1% (536/630)
- **Corda + casa (com abstenção `indeterminado`):** 86,2% (531/616, cobertura de 97,8%)
- **Apenas notas ambíguas (≥ 2 posições possíveis):** 83,9% Top-1 / 85,1% decididas
- **Destaque — 5 posições do Dó4:** 20/25 (80%) entre os 5 captadores (5/5 em `(2,1)` e `(3,5)`, 4/5 em `(5,15)` e `(6,20)`, 2/5 em `(4,10)` devido ao cruzamento físico de rigidez entre corda 3 lisa e corda 4 encapada).

### 2. Interface Web

```bash
uv run guitar-string-identifier
# ou:
uv run uvicorn guitar_string_identifier.web.app:app --host 0.0.0.0 --port 8000
```

Acesse no navegador (inclusive no Windows quando rodando no WSL2): **http://localhost:8000**

## Documentação

- [docs/physics.md](docs/physics.md) — Física da corda rígida, lei de inharmonicidade $B$ e características complementares.
- [docs/architecture.md](docs/architecture.md) — Interfaces, módulos, estrutura real do EGFxSet e captura ao vivo no WSL2.
- [docs/decisions.md](docs/decisions.md) — Decisões técnicas e suposições editáveis.
- [docs/evaluation.md](docs/evaluation.md) — Matriz de confusão, acurácia por corda/casa/captador, curva confiança×erro e destaque do Dó4.
- [CONTRIBUTING.md](CONTRIBUTING.md) — Guia curto de contribuição.

## Licença

MIT — veja [LICENSE](LICENSE).
