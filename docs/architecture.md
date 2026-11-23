# Arquitetura do Sistema

## 1. Pipeline de processamento (interfaces trocáveis)

O núcleo de processamento é organizado em estágios desacoplados definidos por classes abstratas em `src/guitar_string_identifier/core/interfaces.py`:

```text
AudioSource -> NoteSegmenter -> PitchEstimator -> FeatureExtractor(s) -> StringClassifier -> Result
                                                                                              ^
LabelProvider (opcional, para conferência de ground truth em datasets) -----------------------+
```

### Tipos de dados principais (`core/types.py`)
- `Note(audio, sr, t0, f0, source_path)`: segmento mono `float64` de uma nota isolada.
- `Features(values)`: dicionário de características físicas extraídas (`B`, `B_std`, `B_residual_cents`, `partials`, `decay_rates`, `pluck_position_rel`, `spectral_centroid_hz`, `spectral_rolloff_hz`, `spectrum_freqs`, `spectrum_db`).
- `Candidate(string, fret, note, midi, expected_f0, score, distance, expected_b)`: posição candidata compatível com a `f0` estimada.
- `Result(string, fret, note, confidence, candidates, f0, indeterminate, features, label)`: resultado final, incluindo suporte explícito à abstenção (`indeterminate=True`).
- `GroundTruthLabel(string, fret, note, pickup, effect)`: rótulo de referência fornecido por um `LabelProvider`.

## 2. Organização dos módulos

```text
src/guitar_string_identifier/
├── core/
│   ├── types.py             # Dataclasses do pipeline
│   ├── interfaces.py        # Contratos ABC (AudioSource, NoteSegmenter, etc.)
│   ├── music.py             # Conversões nota/MIDI/Hz e geração de candidatos
│   └── pipeline.py          # Orquestrador configurável via GuitarProfile
├── sources/
│   ├── file_source.py       # Leitura via soundfile (WAV, FLAC, OGG, AIFF)
│   ├── dataset_source.py    # EGFxSetLabelProvider e listagem filtrada
│   └── live_source.py       # BufferAudioSource (WebSocket) + SoundDeviceAudioSource (opcional)
├── analysis/
│   ├── segmentation.py      # WholeFileSegmenter e OnsetNoteSegmenter (fluxo espectral)
│   ├── pitch.py             # YinPitchEstimator (YIN com interpolação parabólica)
│   ├── inharmonicity.py     # FFT longa + busca progressiva de parciais + ajuste IRLS de B
│   └── complementary.py     # Decaimento STFT, nulos de palhetada, centroide e roll-off
├── classifier/
│   └── classical.py         # GaussianStringClassifier (distância de Mahalanobis / verossimilhança)
├── calibration/
│   └── calibrator.py        # GuitarCalibrator (estima B0 por corda, rejeita notas ruins)
├── profiles/
│   ├── schema.py            # Modelos Pydantic (Captação, Instrumento, Análise, Calibração)
│   └── store.py             # CRUD de arquivos YAML em configs/profiles/
├── synth/
│   └── stiff_string.py      # Karplus-Strong com cascata all-pass de dispersão
└── web/
    ├── app.py               # Backend FastAPI (REST + WebSocket /ws/live)
    └── static/              # Frontend pt-BR sem build (index.html, styles.css, app.js)
```

## 3. Estrutura verificada do dataset EGFxSet

Após download e inspeção direta dos arquivos reais de `Clean.zip` (Zenodo 7044411):
- **Caminho:** `data/egfxset/Clean/<Captador>/<corda>-<casa>.wav`
- **Captadores (5 pastas):** `Bridge`, `Bridge-Middle`, `Middle`, `Middle-Neck`, `Neck`
- **Corda e casa:** corda `1` (E4, mais aguda) a `6` (E2, mais grave); casa `0` a `22`
- **Total de arquivos em `Clean`:** `5 × 6 × 23 = 690` arquivos WAV
- **Formato de áudio verificado:** `48000 Hz`, `1 canal (mono)`, `PCM_24` (24 bits), duração entre `4,93 s` e `5,03 s`
- **Rótulo de nota:** inferido pela `EGFxSetLabelProvider` combinando `(corda, casa)` com a afinação padrão (`E4, B3, G3, D3, A2, E2`).

## 4. Arquitetura Web e Captura ao Vivo (WSL2)

- O backend FastAPI roda dentro do Ubuntu 24 no WSL2 em `0.0.0.0:8000` e fica acessível diretamente pelo navegador do Windows em `http://localhost:8000`.
- **Captura ao vivo no navegador (Web Audio API):** como o WSL2 não acessa de forma confiável as placas de áudio do Windows, a captura é feita no navegador com `navigator.mediaDevices.enumerateDevices()` e `getUserMedia({ echoCancellation: false, noiseSuppression: false, autoGainControl: false })`, enviando blocos PCM `float32` por WebSocket (`/ws/live`) ao backend.
- Uma fonte alternativa no servidor (`SoundDeviceAudioSource`) permanece implementada atrás da mesma interface `AudioSource` para uso futuro em instalações nativas Linux/Windows/macOS.
