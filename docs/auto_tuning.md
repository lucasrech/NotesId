# Auto Tuning — Configuração Automática do Sistema

O **Auto Tuning** mede automaticamente o sinal de áudio e o instrumento usando apenas processamento clássico de sinais e estatística clássica (sem machine learning) para propor a configuração mais adequada do perfil ativo.

> **Segurança:** o Auto Tuning **nunca sobrescreve** o perfil ativo sem confirmação. Ao final, exibe uma tabela comparando **valor atual × valor proposto × como foi obtido**, permitindo *Aplicar ao perfil ativo*, *Salvar como novo perfil* ou *Descartar*.

---

## 1. O que é automático (`auto`) vs o que não pode ser automatizado

### Automatizado pelo Auto Tuning
| Etapa / Parâmetro | Modo | Como é obtido |
|---|---|---|
| **Placa de captura (sugestão)** | Assistente | Heurística de nome (`Focusrite`, `USB`, `Interface`, etc. acima de `Realtek`/`Microfone integrado`) + monitor de nível ao vivo |
| **Canal ativo** | Assistente | Compara RMS por canal em entradas multicanal e escolhe o que recebeu sinal |
| **Taxa de amostragem (`sample_rate`)** | `auto` | Lida de `MediaStreamTrack.getSettings()` / `AudioContext.sampleRate` (ao vivo) ou do cabeçalho do arquivo WAV |
| **Tamanho de buffer (`buffer_size`)** | `auto` | Derivado de `AudioContext.baseLatency` e taxa de amostragem (potência de 2 em `[1024, 4096]`) |
| **Piso de ruído & limiar de ataque** | Assistente | 2 s de silêncio → RMS e pico de ruído → `onset_threshold_db = max(rms + 16 dB, pico + 6 dB)` |
| **Ronco de rede (50/60 Hz)** | Assistente | FFT do silêncio (prioridade 60 Hz no Brasil) → picos > +10 dB são gravados em `hum_notch_freqs_hz` e excluídos (±3,5 Hz) da busca de parciais |
| **Ganho de referência (`reference_gain_db`)** | `auto` | Nota forte → verifica pico e proporção de clipping → sugere ganho relativo visando −6 dBFS |
| **Afinação, desvios por corda e A4** | Assistente | Detecção sem cliques das 6 cordas soltas → nota mais próxima, preset (`Padrão`, `Drop D`, `Eb`, `DADGAD`, etc.), `a4_hz` global pela mediana e `tuning_cents_offset` por corda |
| **Calibração $B_0$ e decaimento** | Assistente | Calibração guiada sem cliques (cordas soltas + casa 12), com rejeição automática por resíduo |
| **Janela FFT (`window_sec`)** | `auto` | Adaptativa por nota: ~60 ciclos de $f_0$ (mais longa em notas graves), limitada pelo sustain útil |
| **Descarte de transiente (`transient_ms`)** | `auto` | Adaptativo por nota: detecta onde a envoltória RMS pós-pico estabiliza |
| **Nº de parciais (`max_partials`)** | `auto` | Adaptativo por nota: todos os parciais acima do piso de ruído, abaixo de Nyquist e fora das faixas de ronco |
| **Tolerância de picos (`peak_tolerance_cents`)** | `auto` | Estreitamento iterativo (janela inicial → 24 cents → 15 cents) conforme $B$ converge |
| **Pesos das características (`weights`)** | `auto` | Critério discriminante de Fisher (variância entre cordas / variância intra-corda) estimado **somente nos dados de calibração** |
| **Confiança mínima e margem (`min_confidence`, `min_margin`)** | `auto` | Busca em grade na validação interna (leave-one-pickup-out *dentro da calibração*) para atingir `target_error_rate` (padrão 10%) com máxima cobertura |

### O que NÃO pode ser automatizado (exibido como aviso na interface)
1. **Ganho físico (knob Gain) e chave Hi-Z / Instrumento** na interface de áudio USB: o navegador não tem acesso de escrita ao hardware da placa.
2. **Qual entrada física** a guitarra está plugada: só é inferida quando o usuário toca uma corda.
3. **"Aprimoramentos de áudio" do Windows** (supressão de ruído/eco do driver): o navegador não consegue desativá-los. Caminho manual no Windows:
   `Configurações → Sistema → Som → Propriedades do dispositivo → Aprimoramentos de áudio → Desativado`.
4. **Comprimento de escala (mm):** não é estimável pelo áudio, pois a razão $B(\text{casa})/B(0) = 2^{\text{casa}/6}$ independe da escala (que fica absorvida em $B_0$ por corda). O campo é apenas informativo.

---

## 2. Rejeição de entrada ruim e segmentação contínua progressiva

- **Modo de nota isolada:** o sistema valida cada segmento antes de responder. Se detectar **silêncio**, **clipping (> 0,5% das amostras saturadas)** ou **múltiplas notas simultâneas (acorde)** — identificadas por picos espectrais fortes fora da série harmônica de $f_0$ — exibe um aviso claro e não emite palpite de corda/casa.
- **Escuta contínua progressiva (`/ws/live`):** usa `AudioWorklet` (`pcm-worklet.js`), sem `ScriptProcessorNode`, com limiar de ataque derivado do piso de ruído e tempo de rearme (`rearm_time_sec`) contra disparos duplos. Emite uma estimativa provisória em **~0,30 s**, refina em **~0,60 s** e finaliza em **~0,95 s** (ou quando o sustain termina), mostrando a confiança evoluir em tempo real.

---

## 3. Protocolo REST e WebSocket

### Endpoints REST
- `POST /api/autotune/rank-devices` — ordena dispositivos pela heurística de nome (`DeviceProbe`).
- `POST /api/autotune/session` — inicia sessão (`mode: "live" | "dataset"`, `profile`).
- `POST /api/autotune/session/{id}/device` — registra placa, taxa real, canal e buffer sugerido.
- `POST /api/autotune/session/{id}/silence` — analisa 2 s de silêncio (piso de ruído + ronco 50/60 Hz).
- `POST /api/autotune/session/{id}/level` — analisa nota forte (pico, clipping e ganho de referência).
- `POST /api/autotune/session/{id}/tuning` — mede uma corda solta e atualiza preset/A4.
- `POST /api/autotune/session/{id}/calibration` — mede uma nota de calibração `(string, fret)`.
- `POST /api/autotune/session/{id}/dataset-auto` — executa Auto Tuning offline usando **apenas** `cal_frets` (padrão `[0, 12]`).
- `GET /api/autotune/session/{id}/proposal` — retorna a proposta (tabela atual × proposto + avisos) sem salvar.
- `POST /api/autotune/session/{id}/apply` — aplica (`apply_active` ou `save_as_new`) somente após confirmação.

### WebSocket `/ws/autotune`
1. Cliente envia JSON `{"type": "init", "session_id": "...", "sample_rate": 48000}`.
2. Cliente envia `{"type": "set_step", "step": "tuning"|"calibration", "string": 6, "fret": 0}`.
3. Cliente transmite frames binários `Float32Array` vindos do `AudioWorklet`.
4. Servidor envia continuamente `{"type": "level", "rms_db", "peak", "clipping"}` e, ao detectar e medir a nota automaticamente, envia `{"type": "tuning_result"|"calibration_result", "data": {...}}`.

---

## 4. Limitações conhecidas (Navegador e WSL2)
- O WSL2 não acessa interfaces de áudio USB/ASIO do Windows de forma confiável; por isso toda captura ao vivo acontece no navegador do Windows via Web Audio API (`AudioWorklet`) e chega ao backend FastAPI no WSL2 via WebSocket.
- Em alguns drivers Windows, `AudioContext.sampleRate` fica fixo na taxa configurada no painel de som do sistema (geralmente 44,1 ou 48 kHz), mesmo pedindo 96 kHz; o Auto Tuning lê e grava a **taxa real efetivamente obtida**.
