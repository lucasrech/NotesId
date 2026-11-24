# Registro de Decisões Técnicas

Este documento registra escolhas de projeto, suposições adotadas diante de ambiguidades e justificativas técnicas.

## 1. Captura ao vivo via Web Audio API no navegador
- **Contexto:** O ambiente de execução é Ubuntu 24 no WSL2, que não expõe de maneira estável e de baixa latência as interfaces de áudio USB/ASIO do Windows.
- **Decisão:** Realizar a captura ao vivo no navegador (Windows) usando `navigator.mediaDevices.enumerateDevices()` e `getUserMedia` com todos os processamentos de voz desativados (`echoCancellation: false`, `noiseSuppression: false`, `autoGainControl: false`), transmitindo blocos PCM `float32` via WebSocket (`/ws/live`) para o backend FastAPI.
- **Alternativa preservada:** `SoundDeviceAudioSource` implementada atrás da interface `AudioSource` para uso futuro em sistemas nativos.

## 2. Subconjunto baixado do EGFxSet
- **Contexto:** O registro Zenodo 7044411 possui 14 arquivos totalizando **5,81 GB** (12 pedais/efeitos + `Clean.zip` + `egfxset_metadata.csv`).
- **Decisão:** O script `scripts/download_egfxset.py` baixa por padrão apenas `Clean.zip` (431,0 MB, 690 arquivos WAV limpos) e `egfxset_metadata.csv`, com retomada (`curl -C -`) e verificação MD5. A flag `--all` permite baixar o restante se desejado.

## 3. Escala assumida no perfil `egfxset.yaml`
- **Suposição:** A guitarra do EGFxSet é uma Fender Stratocaster de 22 casas; assumimos o comprimento de escala padrão Fender de **648,0 mm** (25,5 polegadas).
- **Status:** Valor registrado como suposição editável no arquivo `configs/profiles/egfxset.yaml` e na página de configuração da interface web. Os calibres e tipos das cordas foram deixados como `null` ("a estimar") por padrão.

## 4. Dependências leves (sem `librosa`)
- **Decisão:** Implementamos YIN (via FFT), STFT, fluxo espectral, interpolação parabólica, centroide e roll-off diretamente com `numpy` e `scipy`, evitando adicionar `librosa` (e sua cadeia pesada de dependências como `numba`/`llvmlite`). Isso acelera a instalação com `uv` e mantém o processamento 100% transparente.

## 5. Rastreamento progressivo de parciais e ajuste IRLS de $B$
- **Contexto:** Em casas altas (onde $B$ ultrapassa $5 \times 10^{-4}$), parciais superiores ($n \ge 6$) deslocam-se mais de 40–100 cents acima de $n \cdot f_0$, saindo de janelas de busca fixas.
- **Decisão:** Usar janela Blackman-Harris (lóbulos laterais < −92 dB) com zero-padding $4\times$, rastreamento progressivo dos parciais (atualizando a predição $n f_0 \sqrt{1 + B n^2}$ a partir do 3º parcial) e ajuste robusto IRLS com pesos de Huber na forma linearizada $(f_n / n)^2 = f_0^2 + (f_0^2 B) n^2$.

## 6. Classificação em escala logarítmica ($\ln B$) com abstenção
- **Contexto:** O coeficiente $B$ varia em duas ordens de grandeza ($2{,}5 \times 10^{-5}$ a $2 \times 10^{-3}$) e seus erros relativos são aproximadamente proporcionais ao valor de $B$.
- **Decisão:** Modelar a distância principal em $\ln B$, combinada com taxa de decaimento média por parcial (peso 0,25) e centroide espectral normalizado (peso 0,05) em um classificador gaussiano / Mahalanobis. Quando a confiança fica abaixo de `min_confidence` (0,45) ou a margem entre 1º e 2º candidato fica abaixo de `min_margin` (0,10), o sistema retorna `indeterminado`.

## 7. Protocolo de avaliação com separação estrita por casa
- **Decisão:** Calibrar o perfil `egfxset` usando **apenas as casas 0 e 12** (espelhando o fluxo da calibração guiada na interface) e avaliar exclusivamente nas **21 casas restantes** (630 arquivos), reportando tanto a acurácia geral quanto a acurácia apenas em notas ambíguas (≥ 2 posições possíveis).

## 8. Auto Tuning: correção de escopo sobre escala e modo `auto`/`manual`
- **Escala da guitarra (`scale_length_mm`):** a razão $B(\text{casa})/B_0 = 2^{\text{casa}/6}$ independe do comprimento de escala (que fica absorvido em $B_0$ por corda). Por isso `scale_length_mm` é mantido apenas como campo **informativo**; tipo e calibre de corda são metadados opcionais.
- **AudioWorklet no navegador:** substituímos `ScriptProcessorNode` por `AudioWorkletProcessor` (`pcm-worklet.js`) para captura PCM de baixa latência em thread dedicada.
- **Modo `auto` vs `manual` por parâmetro:** adicionado `param_modes` ao schema `GuitarProfile` com valores padrão `"manual"` para compatibilidade retroativa total com arquivos YAML existentes.
- **Proposta não-destrutiva:** o Auto Tuning nunca sobrescreve o perfil ativo automaticamente; gera um diff completo para o usuário aplicar ao perfil ativo, salvar como novo perfil ou descartar.
- **Sem vazamento de dados:** os pesos das características (`WeightEstimator`, via critério discriminante de Fisher) e os limiares de decisão (`ThresholdCalibrator`, via validação interna leave-one-pickup-out) são estimados exclusivamente sobre as casas de calibração, jamais tocando nas casas de teste.

## 9. Documentação bilíngue, nome público `NotesId` e bug conhecido registrado
- **Nome público vs pacote Python:** o projeto é apresentado publicamente como **NotesId**, mantendo o nome interno do pacote Python (`guitar_string_identifier`) e da pasta para preservar compatibilidade de imports.
- **Seletor de idioma nos READMEs:** o GitHub não executa JavaScript nem detecta o idioma do navegador dentro de arquivos Markdown. Por isso adotamos `README.md` (inglês, padrão do GitHub) e `README.pt-BR.md` (português) com links explícitos `🇺🇸 English | 🇧🇷 Português` no topo, complementados pela página estática opcional `docs/index.html` para GitHub Pages.
- **Lint de documentação (`ruff`):** habilitadas regras `D` (convenção Google) e `ANN` no `pyproject.toml` (ignorando `D105`, `D107`, `ANN401` e testes em `tests/*.py`).
- **Bug conhecido registrado (não corrigido nesta etapa de documentação):** em `src/guitar_string_identifier/web/static/app.js` (`startWorkletCapture`), `sampleRate` usa `settings.sampleRate || audioCtx.sampleRate`. Quando o driver/navegador abre o `AudioContext` em 44100 Hz mas `MediaStreamTrack.getSettings().sampleRate` reporta 48000 Hz, o backend interpreta amostras de 44,1 kHz como 48 kHz (+147 cents). Deve ser corrigido em commit `fix:` separado priorizando `audioCtx.sampleRate`.

## 10. Causa raiz e correção do bug "status fica Parado na Escuta contínua"
- **Causa raiz:** condição de corrida (*race condition*) em `toggleLiveCapture()` (`src/guitar_string_identifier/web/static/app.js`). O código criava `new WebSocket("/ws/live")` **antes** de `await startWorkletCapture(...)` e só registrava `ws.onopen = () => { ... }` **depois** do `await`. Em interfaces USB reais no Windows, `getUserMedia` leva ~100–500 ms, enquanto o WebSocket local abre em ~2 ms: o evento `onopen` disparava antes de o handler existir, então `config` nunca era enviado, o status permanecia `"Parado"` e `"Analisar agora"` permanecia desabilitado, sem nenhum erro no console.
- **Correções aplicadas:**
  1. Inversão da ordem em `toggleLiveCapture()`: abre a placa + `AudioWorklet` (com `await audioCtx.resume()` e checagem de `audioCtx.audioWorklet`) primeiro, e só então cria o `WebSocket` anexando `onopen`/`onmessage`/`onerror`/`onclose` sincronamente.
  2. Máquina de estados explícita no cliente (`stopped → connecting → listening → detected → stopping → stopped` + `error`) com confirmação de ponta a ponta pelo servidor (`{"type": "status", "state": "listening"}` e `heartbeat` periódico).
  3. Correção da taxa de amostragem em `startWorkletCapture()`: passa a usar `audioCtx.sampleRate` diretamente (resolvendo também o deslocamento de +147 cents quando `track.getSettings().sampleRate` diverge do `AudioContext`).
  4. Medidor de nível animado via `requestAnimationFrame` a partir do worklet, aviso de "sem sinal" após 3,5 s abaixo de −50 dBFS, painel de diagnóstico opcional em `?debug=1`, erros visíveis na interface com botão "Tentar novamente" e cabeçalho `Cache-Control: no-cache` em `/` e `/static/*`.

## 11. Reorganização gráfica do README bilíngue e compatibilidade com Windows nativo
- **Download multiplataforma do EGFxSet (`scripts/download_egfxset.py`):** substituída a chamada externa `curl -C -` por implementação em Python puro (`urllib.request` + cabeçalho HTTP `Range`) para funcionar de forma idêntica no Linux, macOS e Windows PowerShell nativo sem depender de binários externos.
- **Estrutura simétrica dos READMEs:** `README.md` e `README.pt-BR.md` compartilham exatamente 9 capítulos numerados (sem emojis dentro dos títulos linkados, garantindo âncoras estáveis no GitHub), sumário clicável no topo, mini-sumários nos capítulos longos (Tutorial e Calibração), callouts GitHub Markdown e blocos `<details>` para Ubuntu nativo, Windows nativo e os 5 métodos de calibração.
- **Imagens anotadas sem contaminar dependências:** Pillow e Playwright permaneceram fora do `pyproject.toml` do projeto principal; apenas os PNGs otimizados (todos < 200 KB) e SVGs próprios de conexão física foram commitados em `docs/images/`.

## 2026-11-23 — Modo experimental "Celular como microfone pela rede local" e opção `--https`
- **Padrão seguro na CLI (`--host 127.0.0.1`):** anteriormente `run_server()` em `src/guitar_string_identifier/web/app.py` fixava `host="0.0.0.0"` sem argumentos CLI nem suporte a TLS. Alteramos `run_server()` para usar `argparse` com `--host` (padrão `127.0.0.1`, local-only por segurança), `--port` (padrão `8000`), `--https` e `--cert-dir` (padrão `.certs/`).
- **Certificado autoassinado multiplataforma via `cryptography`:** adicionamos a dependência `cryptography` (`uv add cryptography` + `uv export` em `requirements.txt`) para gerar certificados TLS autoassinados ECDSA P-256 com SAN (`localhost`, `127.0.0.1`, `::1` e IPs IPv4 locais detectados) em `src/guitar_string_identifier/web/tls.py`, funcionando tanto em Ubuntu quanto em Windows sem exigir `openssl` instalado. `.certs/`, `*.pem`, `*.key` e `*.crt` foram adicionados ao `.gitignore` para impedir versionamento de chaves privadas.
- **Verificação do WebSocket seguro (`wss://`):** confirmado em `src/guitar_string_identifier/web/static/app.js` que ambos os endpoints (`/ws/live` e `/ws/autotune`) já derivam o protocolo dinamicamente via `location.protocol === "https:" ? "wss" : "ws"`. Validado com teste automatizado em `tests/test_https_server.py` subindo Uvicorn com TLS real e testando `https://.../api/health` e `wss://.../ws/live` (desabilitando validação de certificado exclusivamente dentro do teste).
- **Usabilidade mobile em 390 px:** verificada em emulação Chromium 390×844 (zero overflow horizontal). Adicionado `flex-shrink: 0` em `.status-dot` (`styles.css`) para evitar achatamento do ponto verde de status em telas estreitas. Capturas anotadas geradas em `docs/images/mobile-live-capture.png` e `docs/images/mobile-result-view.png`, rotuladas explicitamente como emulação de celular.
- **Honestidade técnica nos READMEs:** documentado que a acurácia no modo celular não foi medida e tende a ser substancialmente menor do que o sinal limpo via interface USB; explicitado o que foi testado (HTTPS/WSS automatizado e layout 390 px) e o que não foi testado (celular real, microfone físico, Safari/iOS e WSL2 `networkingMode=mirrored` / `portproxy`).
