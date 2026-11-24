/**
 * @fileoverview Controlador do frontend web do NotesId (pt-BR, sem etapa de build).
 *
 * Responsabilidades principais:
 * 1. Modo Arquivo / Dataset: upload de WAV/FLAC/OGG e navegador local do EGFxSet.
 * 2. Modo Ao Vivo: captura contínua via AudioWorklet (`pcm-worklet.js`) e envio
 *    de blocos PCM float32 ao WebSocket `/ws/live`, com exibição progressiva.
 * 3. Visualização: quadro central de nota/corda/casa/confiança, conferência de
 *    acerto (ground truth) e gráfico de espectro (Canvas 2D) com parciais vs n·f0.
 * 4. Configuração & Perfis: edição dos 3 blocos, alternância `auto`/`manual` e CRUD.
 * 5. Calibração Guiada e Assistente Auto Tuning (6 etapas + proposta de revisão).
 *
 * Glossário de termos musicais/DSP: `docs/GLOSSARY.pt-BR.md`.
 */
"use strict";

const state = {
  activeProfile: "egfxset",
  editingProfile: null,
  profiles: [],
  datasetFiles: [],
  live: {
    phase: "stopped", // stopped | connecting | listening | detected | stopping | error
    running: false,
    audioCtx: null,
    stream: null,
    workletNode: null,
    ws: null,
    sampleRate: 48000,
    channel: 0,
    lastRmsDb: -90.0,
    lastPeak: 0.0,
    lastClipping: false,
    lastSignalTs: 0,
    meterRafId: null,
    debugEnabled: new URLSearchParams(window.location.search).get("debug") === "1",
    debugLogs: [],
  },
  calibration: {
    steps: [],
    results: {},
  },
  autotune: {
    sessionId: null,
    mode: "live",
    step: 1,
    audioCtx: null,
    stream: null,
    workletNode: null,
    ws: null,
    sampleRate: 48000,
    baseLatency: null,
    tuningOrder: [6, 5, 4, 3, 2, 1],
    tuningIdx: 0,
    tuningResults: {},
    calQueue: [],
    calIdx: 0,
    calResults: {},
    proposal: null,
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

/* ==================== HELPER: AUDIOWORKLET CAPTURE ==================== */
async function startWorkletCapture(deviceId, chunkSize, onChunk, channelIndex = 0) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "API de captura de áudio indisponível. Acesse via http://localhost:8000 ou HTTPS."
    );
  }
  const constraints = {
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: { ideal: 2 },
      sampleRate: { ideal: 96000 },
    },
  };
  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  const track = stream.getAudioTracks()[0];
  const settings = track && track.getSettings ? track.getSettings() : {};
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();

  if (audioCtx.state === "suspended") {
    await audioCtx.resume();
  }
  if (!audioCtx.audioWorklet) {
    stream.getTracks().forEach((t) => t.stop());
    await audioCtx.close();
    throw new Error(
      "AudioWorklet indisponível neste navegador ou contexto não seguro (use localhost ou HTTPS)."
    );
  }

  try {
    await audioCtx.audioWorklet.addModule("/static/pcm-worklet.js?v=2");
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    await audioCtx.close();
    throw new Error(`Falha ao carregar o processador de áudio (pcm-worklet.js): ${err.message}`);
  }

  const source = audioCtx.createMediaStreamSource(stream);
  const workletNode = new AudioWorkletNode(audioCtx, "pcm-capture-processor", {
    processorOptions: { chunkSize: chunkSize || 2048, channelIndex: channelIndex || 0 },
  });
  workletNode.port.onmessage = (ev) => {
    if (onChunk) onChunk(new Float32Array(ev.data));
  };
  source.connect(workletNode);
  // Conecta a um nó silencioso para manter o grafo ativo sem microfonia
  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;
  workletNode.connect(silentGain);
  silentGain.connect(audioCtx.destination);

  return {
    stream,
    audioCtx,
    workletNode,
    // A taxa real das amostras geradas pelo AudioWorklet é sempre audioCtx.sampleRate
    sampleRate: audioCtx.sampleRate,
    trackSampleRate: settings.sampleRate || null,
    channelIndex: channelIndex || 0,
    baseLatency: typeof audioCtx.baseLatency === "number" ? audioCtx.baseLatency : null,
  };
}

async function captureFixedDurationWorklet(deviceId, durationSec) {
  const chunks = [];
  let collected = 0;
  let targetSamples = 48000;
  let resolveDone;
  const donePromise = new Promise((res) => {
    resolveDone = res;
  });

  const cap = await startWorkletCapture(deviceId, 2048, (arr) => {
    chunks.push(arr);
    collected += arr.length;
    if (collected >= targetSamples) {
      resolveDone();
    }
  });
  targetSamples = Math.round(durationSec * cap.sampleRate);
  await donePromise;

  cap.workletNode.disconnect();
  cap.stream.getTracks().forEach((t) => t.stop());
  await cap.audioCtx.close();

  const merged = new Float32Array(collected);
  let off = 0;
  chunks.forEach((c) => {
    merged.set(c, off);
    off += c.length;
  });
  return { samples: merged, sampleRate: cap.sampleRate };
}

/* ==================== NAVEGAÇÃO ==================== */
function initTabs() {
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => b.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#tab-${btn.dataset.tab}`).classList.add("active");
      if (btn.dataset.tab === "calibrate") {
        loadCalibrationPlan();
      }
    });
  });

  $$(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".mode-btn").forEach((b) => b.classList.remove("active"));
      $$(".mode-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $(`#mode-${btn.dataset.mode}`).classList.add("active");
      if (btn.dataset.mode === "live") {
        refreshAudioDevices("#audioDeviceSelect");
      }
    });
  });
}

/* ==================== EXIBIÇÃO DE RESULTADO ==================== */
function renderResult(data) {
  const warnBanner = $("#inputWarningBanner");
  if (data.input_warning) {
    warnBanner.textContent = `⚠ ${data.input_warning}`;
    warnBanner.classList.remove("hidden");
  } else {
    warnBanner.classList.add("hidden");
  }

  $("#provisionalBadge").classList.toggle("hidden", !data.provisional);

  $("#resNote").textContent = data.note || "?";
  $("#resF0").textContent = data.f0
    ? `f0 medida: ${data.f0.toFixed(2)} Hz`
    : "Frequência fundamental não detectada";

  if (data.input_warning) {
    $("#resPosition").textContent = "Entrada rejeitada";
  } else if (data.indeterminate || data.string === null || data.fret === null) {
    $("#resPosition").textContent = "Indeterminado";
  } else {
    $("#resPosition").textContent = `Corda ${data.string} • Casa ${data.fret}`;
  }

  const confPct = Math.round((data.confidence || 0) * 100);
  $("#resConfidenceText").textContent = `${confPct}%`;
  const bar = $("#resConfidenceBar");
  bar.style.width = `${confPct}%`;
  bar.style.background =
    confPct >= 75 ? "var(--green)" : confPct >= 50 ? "var(--amber)" : "var(--red)";

  // Conferência de acerto (Ground Truth)
  const gtBox = $("#gtBox");
  const gtStatus = $("#gtStatus");
  const gtDetails = $("#gtDetails");
  gtBox.className = "gt-box";

  const cmp = data.comparison || { has_label: false };
  if (!cmp.has_label || !data.label) {
    gtBox.classList.add("gt-neutral");
    gtStatus.textContent = "Sem rótulo para conferir";
    gtDetails.textContent = "Este áudio não possui rótulo conhecido de dataset.";
  } else {
    const lbl = data.label;
    const gotPos = data.indeterminate
      ? "Indeterminado"
      : `Corda ${data.string}, Casa ${data.fret}`;
    const expPos = `Corda ${lbl.string}, Casa ${lbl.fret}`;
    const pickupTxt = lbl.pickup ? ` • Captador: ${lbl.pickup}` : "";

    if (cmp.status === "correct") {
      gtBox.classList.add("gt-correct");
      gtStatus.textContent = "✔ Acertou!";
    } else if (cmp.status === "indeterminate") {
      gtBox.classList.add("gt-indeterminate");
      gtStatus.textContent = "⚠ Indeterminado (abstenção)";
    } else {
      gtBox.classList.add("gt-wrong");
      gtStatus.textContent = "✖ Errou";
    }
    gtDetails.textContent = `Esperado: ${expPos} (${lbl.note})${pickupTxt}  |  Obtido: ${gotPos} (${data.note})`;
  }

  // Candidatos
  const cList = $("#candidatesList");
  if (!data.candidates || data.candidates.length === 0) {
    cList.innerHTML = `<p class="muted">Nenhum candidato compatível com f0.</p>`;
  } else {
    cList.innerHTML = data.candidates
      .map((c, idx) => {
        const pct = Math.round(c.score * 100);
        const expB = c.expected_b ? c.expected_b.toExponential(2) : "—";
        const dist = c.distance !== null ? c.distance.toFixed(2) : "—";
        return `
          <div class="cand-row ${idx === 0 ? "top" : ""}">
            <div class="cand-top">
              <strong>#${idx + 1} Corda ${c.string} • Casa ${c.fret} (${c.note})</strong>
              <strong>${pct}%</strong>
            </div>
            <div class="progress-bar" style="height:6px;margin-bottom:5px;">
              <div class="progress-fill" style="width:${pct}%"></div>
            </div>
            <div class="cand-meta">B esperado: ${expB} • Distância Mahalanobis: ${dist}</div>
          </div>
        `;
      })
      .join("");
  }

  // Características físicas
  const f = data.features || {};
  $("#bSummary").textContent = f.B ? `B medido: ${f.B.toExponential(2)}` : "B medido: —";
  $("#featB").textContent = f.B ? f.B.toExponential(2) : "—";
  $("#featRes").textContent =
    f.B_residual_cents !== null && f.B_residual_cents !== undefined
      ? `${f.B_residual_cents.toFixed(2)} cents`
      : "—";
  $("#featPartials").textContent = `${f.n_partials_used ?? 0}`;
  $("#featDecay").textContent =
    f.decay_mean_db_s !== null && f.decay_mean_db_s !== undefined
      ? `${f.decay_mean_db_s.toFixed(1)} dB/s`
      : "—";
  $("#featPluck").textContent =
    f.pluck_position_rel !== null && f.pluck_position_rel !== undefined
      ? `${Math.round(f.pluck_position_rel * 100)}% da escala`
      : "—";
  $("#featSpectral").textContent =
    f.spectral_centroid_hz && f.spectral_rolloff_hz
      ? `${Math.round(f.spectral_centroid_hz)} / ${Math.round(f.spectral_rolloff_hz)} Hz`
      : "—";

  drawSpectrum(data.spectrum, data.partials || []);
}

/* ==================== GRÁFICO DE ESPECTRO ==================== */
function drawSpectrum(spectrum, partials) {
  const canvas = $("#spectrumCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const freqs = (spectrum && spectrum.freqs) || [];
  const dbs = (spectrum && spectrum.db) || [];
  if (freqs.length < 2) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.fillText("Espectro será exibido após a análise de uma nota.", 24, h / 2);
    return;
  }

  const padL = 52;
  const padR = 20;
  const padT = 22;
  const padB = 34;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const maxF = freqs[freqs.length - 1];
  const minDb = -75;
  const maxDb = 5;

  const xOf = (f) => padL + (f / maxF) * plotW;
  const yOf = (db) =>
    padT + (1 - (Math.max(minDb, Math.min(maxDb, db)) - minDb) / (maxDb - minDb)) * plotH;

  ctx.strokeStyle = "rgba(148,163,184,0.16)";
  ctx.lineWidth = 1;
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#94a3b8";
  for (let db = 0; db >= -70; db -= 15) {
    const y = yOf(db);
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(`${db} dB`, 6, y + 4);
  }

  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = "rgba(148,163,184,0.45)";
  partials.forEach((p) => {
    const x = xOf(p.ideal_freq);
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, h - padB);
    ctx.stroke();
  });
  ctx.restore();

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const x = xOf(freqs[i]);
    const y = yOf(dbs[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  partials.forEach((p) => {
    const x = xOf(p.measured_freq);
    const y = yOf(p.magnitude_db);
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fef3c7";
    ctx.font = "10px sans-serif";
    const sign = p.deviation_cents >= 0 ? "+" : "";
    ctx.fillText(
      `n=${p.n} (${sign}${p.deviation_cents.toFixed(1)}c)`,
      x - 18,
      Math.max(padT + 10, y - 9)
    );
  });

  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  for (let k = 0; k <= 6; k++) {
    const fVal = (maxF * k) / 6;
    const x = xOf(fVal);
    ctx.fillText(`${Math.round(fVal)} Hz`, x - 16, h - 10);
  }
}

/* ==================== DATASET & UPLOAD ==================== */
async function loadDatasetFiles() {
  const pickup = $("#filterPickup").value;
  const string = $("#filterString").value;
  const fret = $("#filterFret").value;

  const params = new URLSearchParams();
  if (pickup) params.set("pickup", pickup);
  if (string) params.set("string", string);
  if (fret !== "") params.set("fret", fret);

  const data = await api(`/api/dataset/files?${params.toString()}`);
  state.datasetFiles = data.files || [];

  const pkSel = $("#filterPickup");
  if (pkSel.options.length <= 1 && data.pickups && data.pickups.length) {
    data.pickups.forEach((pk) => {
      const opt = document.createElement("option");
      opt.value = pk;
      opt.textContent = pk;
      pkSel.appendChild(opt);
    });
  }
  renderDatasetList(state.datasetFiles, data.available);
}

function renderDatasetList(files, available = true) {
  const list = $("#datasetList");
  $("#datasetCount").textContent = `${files.length} arquivos`;
  if (!available) {
    list.innerHTML = `<p class="muted" style="padding:12px">Dataset EGFxSet não encontrado em <code>data/egfxset/Clean</code>.</p>`;
    return;
  }
  if (!files.length) {
    list.innerHTML = `<p class="muted" style="padding:12px">Nenhum arquivo corresponde aos filtros.</p>`;
    return;
  }
  list.innerHTML = files
    .slice(0, 200)
    .map(
      (f) => `
      <div class="dataset-item" data-path="${f.rel_path}">
        <div>
          <strong>Corda ${f.string} • Casa ${f.fret}</strong>
          <span class="tag">${f.note}</span>
        </div>
        <span class="muted">${f.pickup || ""} • ${f.name}</span>
      </div>
    `
    )
    .join("");

  $$(".dataset-item").forEach((el) => {
    el.addEventListener("click", async () => {
      $$(".dataset-item").forEach((i) => i.classList.remove("selected"));
      el.classList.add("selected");
      try {
        const res = await api("/api/analyze/dataset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rel_path: el.dataset.path,
            profile: state.activeProfile,
          }),
        });
        renderResult(res);
      } catch (err) {
        alert(`Erro ao analisar arquivo: ${err.message}`);
      }
    });
  });
}

function initFileAndDataset() {
  const zone = $("#uploadZone");
  const input = $("#audioFileInput");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) uploadAndAnalyze(e.dataTransfer.files[0]);
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadAndAnalyze(input.files[0]);
  });

  ["#filterPickup", "#filterString", "#filterFret"].forEach((sel) => {
    $(sel).addEventListener("change", loadDatasetFiles);
  });
  $("#btnClearFilters").addEventListener("click", () => {
    $("#filterPickup").value = "";
    $("#filterString").value = "";
    $("#filterFret").value = "";
    loadDatasetFiles();
  });
  $("#btnShowC4").addEventListener("click", async () => {
    const data = await api("/api/dataset/files");
    const c4Only = (data.files || []).filter((f) => f.note === "C4");
    renderDatasetList(c4Only, data.available);
  });
}

async function uploadAndAnalyze(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("profile", state.activeProfile);
  try {
    const res = await api("/api/analyze/upload", { method: "POST", body: fd });
    renderResult(res);
  } catch (err) {
    alert(`Erro ao analisar upload: ${err.message}`);
  }
}

/* ==================== DISPOSITIVOS E CAPTURA AO VIVO (AUDIOWORKLET) ==================== */
async function refreshAudioDevices(selectCss = "#audioDeviceSelect") {
  const sel = $(selectCss);
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    sel.innerHTML = `<option value="">Web Audio API indisponível neste navegador</option>`;
    return [];
  }
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch (_) {}

  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices
    .filter((d) => d.kind === "audioinput")
    .map((d) => ({ deviceId: d.deviceId, label: d.label || "" }));

  if (!inputs.length) {
    sel.innerHTML = `<option value="">Nenhuma entrada de áudio detectada</option>`;
    return [];
  }

  // Classifica pela heurística do backend (DeviceProbe)
  let ranked = inputs.map((d, i) => ({
    device_id: d.deviceId,
    label: d.label || `Entrada #${i + 1}`,
    score: 0,
    reason: "",
  }));
  try {
    const resp = await api("/api/autotune/rank-devices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ devices: inputs }),
    });
    ranked = resp.devices || ranked;
  } catch (_) {}

  sel.innerHTML = "";
  ranked.forEach((dev) => {
    const opt = document.createElement("option");
    opt.value = dev.device_id;
    opt.textContent = dev.score > 0 ? `★ ${dev.label} (sugerida)` : dev.label;
    sel.appendChild(opt);
  });
  return ranked;
}

function liveDebugLog(msg) {
  if (!state.live.debugEnabled) return;
  const ts = new Date().toLocaleTimeString("pt-BR");
  state.live.debugLogs.push(`[${ts}] ${msg}`);
  if (state.live.debugLogs.length > 14) state.live.debugLogs.shift();
  const panel = $("#liveDebugPanel");
  const pre = $("#liveDebugPre");
  if (!panel || !pre) return;
  panel.classList.remove("hidden");
  const l = state.live;
  const wsStates = ["CONNECTING", "OPEN", "CLOSING", "CLOSED"];
  const wsState = l.ws ? wsStates[l.ws.readyState] || l.ws.readyState : "none";
  const ctxState = l.audioCtx ? l.audioCtx.state : "none";
  const header = `fase=${l.phase} | AudioContext=${ctxState} | sr=${l.sampleRate} Hz | ch=${l.channel} | WS=${wsState}`;
  pre.textContent = `${header}\n${state.live.debugLogs.join("\n")}`;
}

function showLiveError(message) {
  const box = $("#liveErrorBox");
  const txt = $("#liveErrorText");
  if (!box || !txt) return;
  txt.textContent = `⚠ ${message}`;
  box.classList.remove("hidden");
  liveDebugLog(`ERRO: ${message}`);
}

function clearLiveError() {
  const box = $("#liveErrorBox");
  if (box) box.classList.add("hidden");
}

function formatCaptureError(err) {
  const name = (err && err.name) || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Permissão de microfone/placa negada pelo navegador. Clique no cadeado da barra de endereço e libere o áudio.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "Placa de captura selecionada não encontrada. Clique em 'Atualizar' e selecione outra entrada.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "A placa de áudio está em uso exclusivo por outro aplicativo ou desconectou. Feche outros apps de áudio e tente novamente.";
  }
  return (err && err.message) || "Erro desconhecido ao abrir captura de áudio.";
}

/**
 * Máquina de estados explícita da escuta contínua:
 * stopped -> connecting -> listening -> detected -> stopping -> stopped (+ error).
 */
function setLiveState(nextPhase, customLabel = null) {
  state.live.phase = nextPhase;
  const statusEl = $("#liveStatus");
  const dotEl = $("#liveStatusDot");
  const toggleBtn = $("#btnLiveToggle");
  const analyzeBtn = $("#btnLiveAnalyzeNow");

  if (dotEl) {
    dotEl.className = `status-dot ${nextPhase}`;
  }
  if (statusEl) {
    statusEl.className = `live-status ${nextPhase}`;
    const defaultLabels = {
      stopped: "Parado",
      connecting: "Conectando...",
      listening: `Escutando (${state.live.sampleRate} Hz)`,
      detected: `Escutando • Nota detectada`,
      stopping: "Encerrando...",
      error: "Erro na captura",
    };
    statusEl.textContent = customLabel || defaultLabels[nextPhase] || nextPhase;
  }

  if (toggleBtn) {
    if (nextPhase === "connecting") {
      toggleBtn.textContent = "Conectando...";
      toggleBtn.disabled = true;
    } else if (nextPhase === "listening" || nextPhase === "detected") {
      toggleBtn.textContent = "Parar escuta";
      toggleBtn.disabled = false;
    } else if (nextPhase === "stopping") {
      toggleBtn.textContent = "Encerrando...";
      toggleBtn.disabled = true;
    } else {
      toggleBtn.textContent = "Iniciar escuta contínua";
      toggleBtn.disabled = false;
    }
  }

  if (analyzeBtn) {
    analyzeBtn.disabled = !(nextPhase === "listening" || nextPhase === "detected");
  }

  liveDebugLog(`estado -> ${nextPhase}${customLabel ? ` (${customLabel})` : ""}`);
}

function startLiveMeterLoop() {
  if (state.live.meterRafId) cancelAnimationFrame(state.live.meterRafId);
  const renderFrame = () => {
    const l = state.live;
    if (!l.running) {
      l.meterRafId = null;
      return;
    }
    const db = l.lastRmsDb;
    $("#meterDb").textContent = db <= -89 ? "-∞ dB" : `${db.toFixed(1)} dB`;
    const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
    $("#meterFill").style.width = `${pct}%`;
    $("#clippingBadge").classList.toggle("hidden", !l.lastClipping);

    // Dica se estiver escutando mas sem sinal audível por mais de 3,5 s
    const noSignal =
      (l.phase === "listening" || l.phase === "detected") &&
      l.lastSignalTs > 0 &&
      performance.now() - l.lastSignalTs > 3500;
    const hintEl = $("#liveNoSignalHint");
    if (hintEl) hintEl.classList.toggle("hidden", !noSignal);

    l.meterRafId = requestAnimationFrame(renderFrame);
  };
  state.live.meterRafId = requestAnimationFrame(renderFrame);
}

async function toggleLiveCapture() {
  if (state.live.running || state.live.phase === "connecting") {
    stopLiveCapture();
    return;
  }

  // Garante que não há conflito com o assistente do Auto Tuning
  stopAutoTuneAudio();
  clearLiveError();
  setLiveState("connecting", "Abrindo placa de áudio...");

  const deviceId = $("#audioDeviceSelect").value;
  const channelIdx =
    (state.editingProfile && state.editingProfile.capture && state.editingProfile.capture.channel) ||
    0;

  let cap = null;
  try {
    // Passo 1: abre placa + AudioWorklet ANTES do WebSocket para evitar race condition no ws.onopen
    let activeWs = null;
    cap = await startWorkletCapture(
      deviceId,
      2048,
      (floatChunk) => {
        // Medição local imediata para animar o medidor via requestAnimationFrame
        let sum = 0;
        let pk = 0;
        for (let i = 0; i < floatChunk.length; i++) {
          const v = floatChunk[i];
          const av = v < 0 ? -v : v;
          if (av > pk) pk = av;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / Math.max(1, floatChunk.length) + 1e-18);
        const rmsDb = 20 * Math.log10(rms + 1e-12);
        state.live.lastRmsDb = rmsDb;
        state.live.lastPeak = pk;
        state.live.lastClipping = pk >= 0.985;
        if (rmsDb > -50.0) {
          state.live.lastSignalTs = performance.now();
        }
        if (activeWs && activeWs.readyState === WebSocket.OPEN) {
          activeWs.send(floatChunk.buffer);
        }
      },
      channelIdx
    );

    state.live.sampleRate = cap.sampleRate;
    state.live.channel = cap.channelIndex;
    state.live.audioCtx = cap.audioCtx;
    state.live.stream = cap.stream;
    state.live.workletNode = cap.workletNode;
    state.live.running = true;
    state.live.lastSignalTs = performance.now();

    $("#liveSampleRateInfo").textContent = `Taxa real obtida: ${cap.sampleRate} Hz (AudioWorklet)`;
    setLiveState("connecting", "Conectando ao servidor...");
    startLiveMeterLoop();

    // Passo 2: abre WebSocket e registra TODOS os handlers sincronamente (sem await no meio)
    const wsProto = location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProto}://${location.host}/ws/live`;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    activeWs = ws;
    state.live.ws = ws;

    ws.onopen = () => {
      liveDebugLog(`WS aberto: ${wsUrl}`);
      ws.send(
        JSON.stringify({
          type: "config",
          sample_rate: cap.sampleRate,
          profile: state.activeProfile,
        })
      );
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "status" && msg.state === "listening") {
        liveDebugLog(`Servidor confirmou: listening (${msg.sample_rate} Hz)`);
        setLiveState("listening", `Escutando (${cap.sampleRate} Hz)`);
      } else if (msg.type === "ready" && state.live.phase === "connecting") {
        liveDebugLog(`Servidor pronto: ${msg.profile} (${msg.sample_rate} Hz)`);
        setLiveState("listening", `Escutando (${cap.sampleRate} Hz)`);
      } else if (msg.type === "heartbeat") {
        liveDebugLog(`heartbeat rms=${msg.rms_db} dB`);
      } else if (msg.type === "level") {
        // Sincroniza clipping reportado pelo servidor
        if (msg.clipping) state.live.lastClipping = true;
      } else if (msg.type === "result") {
        setLiveState("detected", `Escutando • ${msg.data.note || "nota"}`);
        renderResult(msg.data);
      } else if (msg.type === "error") {
        showLiveError(msg.message || "Erro retornado pelo servidor de análise.");
        stopLiveCapture("error");
      }
    };

    ws.onerror = () => {
      liveDebugLog("WS onerror disparado");
      if (state.live.running) {
        showLiveError(
          "Não foi possível conectar ao servidor (/ws/live). Verifique se o backend está rodando."
        );
        stopLiveCapture("error");
      }
    };

    ws.onclose = (ev) => {
      liveDebugLog(`WS fechado (code=${ev.code})`);
      if (state.live.running && state.live.phase !== "stopping") {
        showLiveError("A conexão com o servidor (/ws/live) foi encerrada inesperadamente.");
        stopLiveCapture("error");
      }
    };
  } catch (err) {
    if (cap) {
      if (cap.workletNode) cap.workletNode.disconnect();
      if (cap.audioCtx) cap.audioCtx.close();
      if (cap.stream) cap.stream.getTracks().forEach((t) => t.stop());
    }
    showLiveError(formatCaptureError(err));
    stopLiveCapture("error");
  }
}

function stopLiveCapture(finalPhase = "stopped") {
  const l = state.live;
  if (l.phase !== "stopped" && l.phase !== "error") {
    setLiveState("stopping");
  }
  l.running = false;
  if (l.meterRafId) {
    cancelAnimationFrame(l.meterRafId);
    l.meterRafId = null;
  }
  if (l.workletNode) {
    try {
      l.workletNode.disconnect();
    } catch (_) {}
  }
  if (l.audioCtx) {
    try {
      l.audioCtx.close();
    } catch (_) {}
  }
  if (l.stream) l.stream.getTracks().forEach((t) => t.stop());
  if (l.ws && l.ws.readyState <= 1) {
    try {
      l.ws.close();
    } catch (_) {}
  }
  l.audioCtx = null;
  l.stream = null;
  l.workletNode = null;
  l.ws = null;
  l.lastRmsDb = -90.0;
  l.lastClipping = false;

  $("#meterDb").textContent = "-∞ dB";
  $("#meterFill").style.width = "0%";
  $("#clippingBadge").classList.add("hidden");
  const hintEl = $("#liveNoSignalHint");
  if (hintEl) hintEl.classList.add("hidden");

  setLiveState(finalPhase === "error" ? "error" : "stopped");
}

/* ==================== PERFIS & CONFIGURAÇÃO (AUTO/MANUAL) ==================== */
const MODE_INPUT_MAP = {
  sample_rate: ["#cfgSampleRate"],
  buffer_size: ["#cfgBufferSize"],
  reference_gain_db: ["#cfgGain"],
  window_sec: ["#cfgWindowSec"],
  transient_ms: ["#cfgTransientMs"],
  max_partials: ["#cfgMaxPartials"],
  peak_tolerance_cents: ["#cfgPeakTol"],
  min_confidence: ["#cfgMinConf"],
  min_margin: ["#cfgMinMargin"],
  weights: ["#cfgWeightB", "#cfgWeightDecay", "#cfgWeightSpec"],
};

function renderModePills() {
  const modes = (state.editingProfile && state.editingProfile.param_modes) || {};
  $$(".mode-pill").forEach((btn) => {
    const key = btn.dataset.modeKey;
    const isAuto = modes[key] === "auto";
    btn.textContent = isAuto ? "auto" : "manual";
    btn.classList.toggle("is-auto", isAuto);
    (MODE_INPUT_MAP[key] || []).forEach((sel) => {
      const inp = $(sel);
      if (inp) {
        inp.readOnly = isAuto;
        inp.classList.toggle("field-auto", isAuto);
      }
    });
  });
}

async function loadProfilesList(selectName = null) {
  const data = await api("/api/profiles");
  state.activeProfile = data.active;
  state.profiles = data.profiles;

  $("#activeProfileBadge").textContent = data.active;
  const sel = $("#profileSelect");
  sel.innerHTML = "";
  data.profiles.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name === data.active ? `${name} (ativo)` : name;
    sel.appendChild(opt);
  });

  const toLoad = selectName && data.profiles.includes(selectName) ? selectName : data.active;
  sel.value = toLoad;
  await loadProfileIntoForm(toLoad);
}

async function loadProfileIntoForm(name) {
  const prof = await api(`/api/profiles/${encodeURIComponent(name)}`);
  state.editingProfile = prof;

  const isCal = prof.calibration && Object.keys(prof.calibration.b0_by_string || {}).length >= 6;
  if (name === state.activeProfile) {
    $("#calibrationBadge").classList.toggle("calibrated", isCal);
    $("#calibrationBadge").title = isCal ? "Perfil calibrado" : "Perfil não calibrado";
  }

  $("#cfgDescription").value = prof.description || "";
  $("#cfgSourceMode").value = prof.capture.source_mode;
  $("#cfgDeviceName").value = prof.capture.device_name || "";
  $("#cfgSampleRate").value = prof.capture.sample_rate;
  $("#cfgChannel").value = prof.capture.channel;
  $("#cfgGain").value = prof.capture.reference_gain_db;
  $("#cfgBufferSize").value = prof.capture.buffer_size;
  $("#cfgOnsetDb").value = prof.capture.onset_threshold_db ?? -36.0;
  $("#cfgRearmSec").value = prof.capture.rearm_time_sec ?? 0.35;

  const nfTxt =
    prof.capture.noise_floor_db !== null && prof.capture.noise_floor_db !== undefined
      ? `Piso de ruído: ${prof.capture.noise_floor_db} dBFS`
      : "Piso de ruído: não medido";
  const humList = prof.capture.hum_notch_freqs_hz || [];
  const humTxt = humList.length
    ? ` • Ronco excluído: ${prof.capture.hum_freq_hz || 60} Hz (${humList.length} faixas)`
    : " • Sem ronco excluído";
  $("#cfgNoiseHumInfo").textContent = nfTxt + humTxt;

  $("#cfgModelName").value = prof.instrument.model_name;
  $("#cfgNumFrets").value = prof.instrument.num_frets;
  $("#cfgScaleMm").value = prof.instrument.scale_length_mm;
  $("#cfgA4Hz").value = prof.instrument.a4_hz ?? 440.0;
  $("#cfgPickup").value = prof.instrument.pickup_config || "";

  const tbody = $("#cfgStringsBody");
  tbody.innerHTML = "";
  const offsets = prof.instrument.tuning_cents_offset || {};
  for (let s = 1; s <= 6; s++) {
    const note = prof.instrument.tuning[s] || "";
    const off = offsets[s];
    const offBadge =
      off !== undefined && off !== null ? ` <span class="hint">(${off >= 0 ? "+" : ""}${off}c)</span>` : "";
    const spec = (prof.instrument.strings && prof.instrument.strings[s]) || {};
    tbody.innerHTML += `
      <tr>
        <td><strong>${s}</strong>${offBadge}</td>
        <td><input type="text" data-str-tune="${s}" value="${note}" required /></td>
        <td>
          <select data-str-type="${s}">
            <option value="" ${!spec.string_type ? "selected" : ""}>Opcional</option>
            <option value="plain" ${spec.string_type === "plain" ? "selected" : ""}>Lisa</option>
            <option value="wound" ${spec.string_type === "wound" ? "selected" : ""}>Encapada</option>
          </select>
        </td>
        <td>
          <input type="number" step="0.001" min="0.006" max="0.080"
                 data-str-gauge="${s}" placeholder="Opcional"
                 value="${spec.gauge_inches ?? ""}" />
        </td>
      </tr>
    `;
  }

  const a = prof.analysis;
  $("#cfgWindowSec").value = a.window_sec;
  $("#cfgTransientMs").value = a.transient_ms;
  $("#cfgMaxPartials").value = a.max_partials;
  $("#cfgPeakTol").value = a.peak_tolerance_cents;
  $("#cfgMinConf").value = a.min_confidence;
  $("#cfgMinMargin").value = a.min_margin;
  $("#cfgWeightB").value = a.weight_inharmonicity;
  $("#cfgWeightDecay").value = a.weight_decay;
  $("#cfgWeightSpec").value = a.weight_spectral;

  renderModePills();

  const cal = prof.calibration || {};
  const b0Map = cal.b0_by_string || {};
  const b0Keys = Object.keys(b0Map);
  if (b0Keys.length) {
    const items = b0Keys
      .sort()
      .map((k) => `C${k}: ${Number(b0Map[k]).toExponential(2)}`)
      .join(" • ");
    $("#cfgCalibrationSummary").innerHTML = `
      <strong>Calibração salva (${cal.notes_used || 0} notas):</strong><br/>
      ${items}<br/>
      <span class="hint">σ(ln B) = ${cal.log_b_sigma} • ${cal.calibration_notes || ""}</span>
    `;
  } else {
    $("#cfgCalibrationSummary").innerHTML = `
      <strong>Sem calibração salva.</strong> Use o <em>Auto Tuning</em> ou <em>Calibração Guiada</em>.
    `;
  }
}

function collectProfileFromForm() {
  const prof = structuredClone(state.editingProfile);
  prof.description = $("#cfgDescription").value;
  prof.capture = {
    ...prof.capture,
    source_mode: $("#cfgSourceMode").value,
    device_name: $("#cfgDeviceName").value.trim() || null,
    sample_rate: parseInt($("#cfgSampleRate").value, 10),
    channel: parseInt($("#cfgChannel").value, 10),
    reference_gain_db: parseFloat($("#cfgGain").value),
    buffer_size: parseInt($("#cfgBufferSize").value, 10),
    onset_threshold_db: parseFloat($("#cfgOnsetDb").value),
    rearm_time_sec: parseFloat($("#cfgRearmSec").value),
  };

  const tuning = {};
  const strings = {};
  for (let s = 1; s <= 6; s++) {
    tuning[s] = $(`[data-str-tune="${s}"]`).value.trim();
    const tVal = $(`[data-str-type="${s}"]`).value || null;
    const gRaw = $(`[data-str-gauge="${s}"]`).value;
    strings[s] = {
      string_type: tVal,
      gauge_inches: gRaw !== "" ? parseFloat(gRaw) : null,
    };
  }
  prof.instrument = {
    ...prof.instrument,
    model_name: $("#cfgModelName").value.trim(),
    tuning,
    num_frets: parseInt($("#cfgNumFrets").value, 10),
    scale_length_mm: parseFloat($("#cfgScaleMm").value),
    a4_hz: parseFloat($("#cfgA4Hz").value),
    strings,
    pickup_config: $("#cfgPickup").value.trim() || null,
  };

  prof.analysis = {
    ...prof.analysis,
    window_sec: parseFloat($("#cfgWindowSec").value),
    transient_ms: parseFloat($("#cfgTransientMs").value),
    max_partials: parseInt($("#cfgMaxPartials").value, 10),
    peak_tolerance_cents: parseFloat($("#cfgPeakTol").value),
    min_confidence: parseFloat($("#cfgMinConf").value),
    min_margin: parseFloat($("#cfgMinMargin").value),
    weight_inharmonicity: parseFloat($("#cfgWeightB").value),
    weight_decay: parseFloat($("#cfgWeightDecay").value),
    weight_spectral: parseFloat($("#cfgWeightSpec").value),
  };
  return prof;
}

function initConfigEvents() {
  $("#profileSelect").addEventListener("change", (e) => loadProfileIntoForm(e.target.value));

  $$(".mode-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (!state.editingProfile) return;
      const key = btn.dataset.modeKey;
      state.editingProfile.param_modes = state.editingProfile.param_modes || {};
      state.editingProfile.param_modes[key] =
        state.editingProfile.param_modes[key] === "auto" ? "manual" : "auto";
      renderModePills();
    });
  });

  $("#configForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const updated = collectProfileFromForm();
      await api(`/api/profiles/${encodeURIComponent(updated.name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      });
      $("#cfgSaveFeedback").textContent = "Perfil salvo!";
      setTimeout(() => ($("#cfgSaveFeedback").textContent = ""), 2500);
      await loadProfilesList(updated.name);
    } catch (err) {
      alert(`Erro ao salvar perfil: ${err.message}`);
    }
  });

  $("#btnActivateProfile").addEventListener("click", async () => {
    const name = $("#profileSelect").value;
    await api(`/api/profiles/${encodeURIComponent(name)}/activate`, { method: "POST" });
    await loadProfilesList(name);
  });

  $("#btnNewProfile").addEventListener("click", async () => {
    const name = prompt("Nome do novo perfil (sem espaços ou barras):");
    if (!name) return;
    try {
      await api("/api/profiles", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name.trim(), description: "Novo perfil de guitarra." }),
      });
      await loadProfilesList(name.trim());
    } catch (err) {
      alert(`Erro ao criar perfil: ${err.message}`);
    }
  });

  $("#btnDuplicateProfile").addEventListener("click", async () => {
    const src = $("#profileSelect").value;
    const dst = prompt(`Duplicar '${src}' como:`, `${src}_copia`);
    if (!dst) return;
    try {
      await api(`/api/profiles/${encodeURIComponent(src)}/duplicate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: dst.trim() }),
      });
      await loadProfilesList(dst.trim());
    } catch (err) {
      alert(`Erro ao duplicar: ${err.message}`);
    }
  });

  $("#btnRenameProfile").addEventListener("click", async () => {
    const src = $("#profileSelect").value;
    const dst = prompt(`Renomear '${src}' para:`, src);
    if (!dst || dst.trim() === src) return;
    try {
      await api(`/api/profiles/${encodeURIComponent(src)}/rename`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_name: dst.trim() }),
      });
      await loadProfilesList(dst.trim());
    } catch (err) {
      alert(`Erro ao renomear: ${err.message}`);
    }
  });

  $("#btnDeleteProfile").addEventListener("click", async () => {
    const src = $("#profileSelect").value;
    if (!confirm(`Excluir permanentemente o perfil '${src}'?`)) return;
    try {
      await api(`/api/profiles/${encodeURIComponent(src)}`, { method: "DELETE" });
      await loadProfilesList();
    } catch (err) {
      alert(`Erro ao excluir: ${err.message}`);
    }
  });
}

/* ==================== CALIBRAÇÃO GUIADA (ABA 3) ==================== */
async function loadCalibrationPlan() {
  const data = await api(`/api/calibration/plan?profile=${encodeURIComponent(state.activeProfile)}`);
  state.calibration.steps = data.steps || [];
  renderCalibrationTable();
}

function renderCalibrationTable() {
  const tbody = $("#calStepsBody");
  tbody.innerHTML = state.calibration.steps
    .map((st, idx) => {
      const key = `${st.string}-${st.fret}`;
      const r = state.calibration.results[key];
      const bMeas = r && r.b_measured ? r.b_measured.toExponential(2) : "—";
      const b0Imp = r && r.b0_implied ? r.b0_implied.toExponential(2) : "—";
      const resTxt = r && r.residual_cents !== null ? `${r.residual_cents.toFixed(2)} c` : "—";
      let statusHtml = `<span class="status-pending">Pendente</span>`;
      if (r) {
        statusHtml = r.accepted
          ? `<span class="status-ok">✔ Aceita</span>`
          : `<span class="status-err" title="${r.reason}">✖ Rejeitada (${r.reason})</span>`;
      }
      return `
        <tr>
          <td>#${idx + 1}</td>
          <td><strong>Corda ${st.string} • Casa ${st.fret}</strong></td>
          <td>${st.note} (${st.expected_f0} Hz)</td>
          <td>
            <button type="button" class="btn btn-secondary btn-sm" data-cal-live="${key}">
              Gravar (1,2 s)
            </button>
            <label class="btn btn-ghost btn-sm" style="cursor:pointer">
              WAV
              <input type="file" accept=".wav,.flac,.ogg,audio/*" data-cal-file="${key}" hidden />
            </label>
          </td>
          <td>${bMeas}</td>
          <td>${b0Imp}</td>
          <td>${resTxt}</td>
          <td>${statusHtml}</td>
        </tr>
      `;
    })
    .join("");

  $$("[data-cal-live]").forEach((btn) => {
    btn.addEventListener("click", () => recordCalibrationNote(btn.dataset.calLive, btn));
  });
  $$("[data-cal-file]").forEach((inp) => {
    inp.addEventListener("change", () => {
      if (inp.files.length) uploadCalibrationNote(inp.dataset.calFile, inp.files[0]);
    });
  });

  const acceptedCount = Object.values(state.calibration.results).filter((r) => r.accepted).length;
  $("#btnApplyCalibration").disabled = acceptedCount === 0;
}

async function recordCalibrationNote(key, btn) {
  const [sStr, fStr] = key.split("-");
  const string = parseInt(sStr, 10);
  const fret = parseInt(fStr, 10);
  btn.disabled = true;
  btn.textContent = "Gravando...";
  try {
    const deviceId = $("#audioDeviceSelect").value;
    const rec = await captureFixedDurationWorklet(deviceId, 1.2);
    const res = await api("/api/calibration/measure-pcm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile: state.activeProfile,
        string,
        fret,
        sample_rate: rec.sampleRate,
        samples: Array.from(rec.samples),
      }),
    });
    state.calibration.results[key] = res;
    renderCalibrationTable();
  } catch (err) {
    alert(`Erro na gravação de calibração: ${err.message}`);
    btn.disabled = false;
    btn.textContent = "Gravar (1,2 s)";
  }
}

async function uploadCalibrationNote(key, file) {
  const [sStr, fStr] = key.split("-");
  const fd = new FormData();
  fd.append("file", file);
  fd.append("string", sStr);
  fd.append("fret", fStr);
  fd.append("profile", state.activeProfile);
  try {
    const res = await api("/api/calibration/measure-upload", { method: "POST", body: fd });
    state.calibration.results[key] = res;
    renderCalibrationTable();
  } catch (err) {
    alert(`Erro ao medir arquivo de calibração: ${err.message}`);
  }
}

async function applyCalibration() {
  const samples = Object.values(state.calibration.results);
  try {
    const resp = await api("/api/calibration/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile: state.activeProfile,
        samples,
        notes_label: `Calibração guiada via interface web (${samples.filter((s) => s.accepted).length} notas aceitas).`,
      }),
    });
    const fb = $("#calFeedback");
    fb.textContent = `Calibração aplicada ao perfil '${state.activeProfile}' (${resp.accepted} aceitas, ${resp.rejected} rejeitadas).`;
    fb.classList.remove("hidden");
    await loadProfilesList(state.activeProfile);
  } catch (err) {
    alert(`Erro ao aplicar calibração: ${err.message}`);
  }
}

/* ==================== ASSISTENTE AUTO TUNING ==================== */
function setWizardStep(stepNum) {
  state.autotune.step = stepNum;
  $$(".w-panel").forEach((p) => p.classList.remove("active"));
  $(`#wPanel-${stepNum}`).classList.add("active");
  $$(".w-step").forEach((el) => {
    const n = parseInt(el.dataset.wstep, 10);
    el.classList.toggle("active", n === stepNum);
  });
}

function setWizardStepIcon(stepNum, status) {
  const el = $(`.w-step[data-wstep="${stepNum}"]`);
  if (!el) return;
  el.classList.remove("done", "warn", "err");
  const icon = el.querySelector(".w-icon");
  if (status === "ok") {
    el.classList.add("done");
    icon.textContent = "✔";
  } else if (status === "warn") {
    el.classList.add("warn");
    icon.textContent = "⚠";
  } else if (status === "err") {
    el.classList.add("err");
    icon.textContent = "✖";
  } else {
    icon.textContent = "○";
  }
}

function stopAutoTuneAudio() {
  const at = state.autotune;
  if (at.workletNode) at.workletNode.disconnect();
  if (at.audioCtx) at.audioCtx.close();
  if (at.stream) at.stream.getTracks().forEach((t) => t.stop());
  if (at.ws && at.ws.readyState <= 1) at.ws.close();
  at.workletNode = null;
  at.audioCtx = null;
  at.stream = null;
  at.ws = null;
}

async function openAutoTuneModal() {
  if (state.live.running) stopLiveCapture();
  stopAutoTuneAudio();
  state.autotune.sessionId = null;
  state.autotune.tuningResults = {};
  state.autotune.calResults = {};
  state.autotune.proposal = null;
  for (let i = 1; i <= 6; i++) setWizardStepIcon(i, "pending");
  $("#atSilenceResult").classList.add("hidden");
  $("#atLevelResult").classList.add("hidden");
  $("#btnAtStep2Next").disabled = true;
  $("#btnAtStep3Next").disabled = true;

  $("#autotuneModal").classList.remove("hidden");
  setWizardStep(1);
  await refreshAudioDevices("#atDeviceSelect");
}

function closeAutoTuneModal() {
  stopAutoTuneAudio();
  $("#autotuneModal").classList.add("hidden");
}

async function startAutoTuneMonitor() {
  stopAutoTuneAudio();
  const deviceId = $("#atDeviceSelect").value;
  if (!deviceId) return;
  try {
    const cap = await startWorkletCapture(deviceId, 2048, (chunk) => {
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      const rms = Math.sqrt(sum / chunk.length + 1e-18);
      const db = 20 * Math.log10(rms + 1e-12);
      $("#atMonitorDb").textContent = `${db.toFixed(1)} dB`;
      const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
      $("#atMonitorFill").style.width = `${pct}%`;
    });
    state.autotune.audioCtx = cap.audioCtx;
    state.autotune.stream = cap.stream;
    state.autotune.workletNode = cap.workletNode;
    state.autotune.sampleRate = cap.sampleRate;
    state.autotune.baseLatency = cap.baseLatency;
    const latTxt = cap.baseLatency !== null ? ` • baseLatency ${(cap.baseLatency * 1000).toFixed(1)} ms` : "";
    $("#atDeviceMeta").textContent = `Taxa real: ${cap.sampleRate} Hz${latTxt}`;
  } catch (err) {
    $("#atDeviceMeta").textContent = `Erro ao monitorar placa: ${err.message}`;
  }
}

async function handleAutoTuneStep1Next() {
  const mode = document.querySelector('input[name="atMode"]:checked').value;
  state.autotune.mode = mode;

  const created = await api("/api/autotune/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, profile: state.activeProfile }),
  });
  state.autotune.sessionId = created.session_id;

  if (mode === "dataset") {
    // Modo Arquivo/Dataset: pula hardware (etapas 1–3) e executa calibração + pesos + limiares
    const btn = $("#btnAtStep1Next");
    btn.disabled = true;
    btn.textContent = "Executando Auto Tuning no dataset...";
    try {
      const frets = $("#atCalFrets")
        .value.trim()
        .split(/\s+/)
        .map((x) => parseInt(x, 10))
        .filter((x) => !Number.isNaN(x));
      const targetErr = parseFloat($("#atTargetError").value) || 0.10;
      const proposal = await api(`/api/autotune/session/${created.session_id}/dataset-auto`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cal_frets: frets.length ? frets : [0, 12],
          target_error_rate: targetErr,
        }),
      });
      setWizardStepIcon(1, "ok");
      setWizardStepIcon(4, "ok");
      setWizardStepIcon(5, "ok");
      renderAutoTuneProposal(proposal);
      setWizardStep(6);
    } catch (err) {
      alert(`Erro no Auto Tuning de dataset: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = "Continuar →";
    }
    return;
  }

  // Modo Ao vivo: grava placa/taxa/buffer e avança para etapa 2 (silêncio)
  stopAutoTuneAudio();
  const sel = $("#atDeviceSelect");
  const devLabel = sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].textContent : null;
  await api(`/api/autotune/session/${created.session_id}/device`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      device_name: devLabel,
      sample_rate: state.autotune.sampleRate || 48000,
      channel: 0,
      base_latency_sec: state.autotune.baseLatency,
    }),
  });
  setWizardStepIcon(1, "ok");
  setWizardStep(2);
}

async function handleAutoTuneSilence() {
  const btn = $("#btnAtCaptureSilence");
  btn.disabled = true;
  btn.textContent = "Medindo silêncio (2 s)...";
  try {
    const rec = await captureFixedDurationWorklet($("#atDeviceSelect").value, 2.0);
    const res = await api(`/api/autotune/session/${state.autotune.sessionId}/silence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_rate: rec.sampleRate,
        samples: Array.from(rec.samples),
      }),
    });
    const nf = res.noise_floor;
    const hum = res.hum;
    const box = $("#atSilenceResult");
    box.innerHTML = `
      <p><strong>Piso de ruído:</strong> ${nf.rms_db} dBFS • Limiar de ataque sugerido: <strong>${nf.suggested_onset_threshold_db} dBFS</strong></p>
      <p>${nf.message}</p>
      <p style="margin-top:6px"><strong>Ronco da rede:</strong> ${hum.message}</p>
    `;
    box.classList.remove("hidden");
    setWizardStepIcon(2, nf.status === "too_loud" ? "warn" : "ok");
    $("#btnAtStep2Next").disabled = false;
  } catch (err) {
    alert(`Erro ao medir silêncio: ${err.message}`);
    setWizardStepIcon(2, "err");
  } finally {
    btn.disabled = false;
    btn.textContent = "Medir 2 segundos de silêncio novamente";
  }
}

async function handleAutoTuneLevel() {
  const btn = $("#btnAtCaptureLevel");
  btn.disabled = true;
  btn.textContent = "Capturando nota forte (1,5 s)...";
  try {
    const rec = await captureFixedDurationWorklet($("#atDeviceSelect").value, 1.5);
    const res = await api(`/api/autotune/session/${state.autotune.sessionId}/level`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples: Array.from(rec.samples) }),
    });
    const box = $("#atLevelResult");
    box.innerHTML = `
      <p><strong>Pico:</strong> ${res.peak_db} dBFS • <strong>RMS:</strong> ${res.rms_db} dBFS • <strong>Clipping:</strong> ${res.clipped_samples} amostras</p>
      <p>${res.message}</p>
    `;
    box.classList.remove("hidden");
    setWizardStepIcon(3, res.status === "good" ? "ok" : res.status === "clipping" ? "err" : "warn");
    $("#btnAtStep3Next").disabled = false;
  } catch (err) {
    alert(`Erro ao medir nível: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Medir nota forte novamente";
  }
}

async function startAutoTuneWsStream(onMessage) {
  stopAutoTuneAudio();
  const wsProto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${wsProto}://${location.host}/ws/autotune`);
  ws.binaryType = "arraybuffer";
  const cap = await startWorkletCapture($("#atDeviceSelect").value, 2048, (chunk) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(chunk.buffer);
  });
  await new Promise((resolve) => {
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "init",
          session_id: state.autotune.sessionId,
          sample_rate: cap.sampleRate,
        })
      );
      resolve();
    };
  });
  ws.onmessage = (ev) => onMessage(JSON.parse(ev.data), ws);
  state.autotune.audioCtx = cap.audioCtx;
  state.autotune.stream = cap.stream;
  state.autotune.workletNode = cap.workletNode;
  state.autotune.ws = ws;
  return ws;
}

function renderAutoTuneTuningTable() {
  const wrap = $("#atTuningTableWrap");
  const rows = state.autotune.tuningOrder
    .map((s) => {
      const r = state.autotune.tuningResults[s];
      const f0 = r ? `${r.f0_hz} Hz` : "—";
      const note = r ? `${r.nearest_note} (${r.cents_from_440 >= 0 ? "+" : ""}${r.cents_from_440} c)` : "Aguardando";
      const st = r ? (r.accepted ? "✔ OK" : `✖ ${r.reason}`) : "○";
      return `<tr><td><strong>Corda ${s}</strong></td><td>${f0}</td><td>${note}</td><td>${st}</td></tr>`;
    })
    .join("");
  wrap.innerHTML = `<table class="cal-table"><thead><tr><th>Corda</th><th>f0</th><th>Nota / Desvio</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function startClickFreeTuning() {
  state.autotune.tuningIdx = 0;
  state.autotune.tuningResults = {};
  renderAutoTuneTuningTable();
  $("#btnAtRepeatTuning").disabled = false;

  const ws = await startAutoTuneWsStream((msg, socket) => {
    if (msg.type === "tuning_result") {
      const r = msg.data;
      state.autotune.tuningResults[r.string] = r;
      renderAutoTuneTuningTable();
      if (r.summary) {
        const b = $("#atTuningPresetBanner");
        b.textContent = `Afinação detectada: ${r.summary.preset_name} • Referência A4: ${r.summary.a4_hz} Hz`;
        b.classList.remove("hidden");
      }
      if (r.accepted) {
        state.autotune.tuningIdx += 1;
        if (state.autotune.tuningIdx < state.autotune.tuningOrder.length) {
          const nextS = state.autotune.tuningOrder[state.autotune.tuningIdx];
          $("#atTuningPrompt").textContent = `Agora toque a Corda ${nextS} solta...`;
          socket.send(JSON.stringify({ type: "set_step", step: "tuning", string: nextS, fret: 0 }));
        } else {
          $("#atTuningPrompt").textContent = "✔ Todas as 6 cordas detectadas!";
          setWizardStepIcon(4, "ok");
          stopAutoTuneAudio();
        }
      } else {
        $("#atTuningPrompt").textContent = `✖ ${r.reason} Toque a Corda ${r.string} solta novamente.`;
      }
    }
  });

  const firstS = state.autotune.tuningOrder[0];
  $("#atTuningPrompt").textContent = `Toque a Corda ${firstS} solta (sem clicar)...`;
  ws.send(JSON.stringify({ type: "set_step", step: "tuning", string: firstS, fret: 0 }));
}

function renderAutoTuneCalGrid() {
  const list = $("#atCalProgressList");
  list.innerHTML = state.autotune.calQueue
    .map((item, idx) => {
      const key = `${item.string}-${item.fret}`;
      const r = state.autotune.calResults[key];
      const cls = r ? (r.accepted ? "ok" : "err") : idx === state.autotune.calIdx ? "current" : "";
      const txt = r
        ? r.accepted
          ? `✔ B=${r.b_measured.toExponential(1)} (${r.residual_cents}c)`
          : `✖ ${r.reason}`
        : "Pendente";
      return `<div class="at-cal-chip ${cls}"><strong>C${item.string} • Casa ${item.fret}</strong><br/><span class="hint">${txt}</span></div>`;
    })
    .join("");
}

async function startClickFreeCalibration() {
  state.autotune.calQueue = [];
  for (const s of [6, 5, 4, 3, 2, 1]) {
    state.autotune.calQueue.push({ string: s, fret: 0 });
    state.autotune.calQueue.push({ string: s, fret: 12 });
  }
  state.autotune.calIdx = 0;
  state.autotune.calResults = {};
  renderAutoTuneCalGrid();
  $("#btnAtRepeatCal").disabled = false;

  const ws = await startAutoTuneWsStream((msg, socket) => {
    if (msg.type === "calibration_result") {
      const r = msg.data;
      state.autotune.calResults[`${r.string}-${r.fret}`] = r;
      if (r.accepted) {
        state.autotune.calIdx += 1;
      }
      renderAutoTuneCalGrid();
      if (state.autotune.calIdx < state.autotune.calQueue.length) {
        const cur = state.autotune.calQueue[state.autotune.calIdx];
        $("#atCalPrompt").textContent = r.accepted
          ? `✔ Aceita! Agora toque Corda ${cur.string} • Casa ${cur.fret} (${state.autotune.calIdx + 1}/${state.autotune.calQueue.length})`
          : `✖ Rejeitada (${r.reason}). Repita Corda ${cur.string} • Casa ${cur.fret}.`;
        socket.send(
          JSON.stringify({
            type: "set_step",
            step: "calibration",
            string: cur.string,
            fret: cur.fret,
          })
        );
      } else {
        $("#atCalPrompt").textContent = "✔ Calibração concluída! Avance para o resumo.";
        setWizardStepIcon(5, "ok");
        stopAutoTuneAudio();
      }
    }
  });

  const first = state.autotune.calQueue[0];
  $("#atCalPrompt").textContent = `Toque Corda ${first.string} • Casa ${first.fret} (1/${state.autotune.calQueue.length})...`;
  ws.send(
    JSON.stringify({
      type: "set_step",
      step: "calibration",
      string: first.string,
      fret: first.fret,
    })
  );
}

function renderAutoTuneProposal(proposal) {
  state.autotune.proposal = proposal;
  const warnBox = $("#atWarningsBox");
  if (proposal.warnings && proposal.warnings.length) {
    warnBox.innerHTML = proposal.warnings.map((w) => `<div>⚠ ${w}</div>`).join("");
    warnBox.classList.remove("hidden");
  } else {
    warnBox.classList.add("hidden");
  }

  const fmt = (v) => {
    if (v === null || v === undefined) return "—";
    if (Array.isArray(v)) return v.length ? v.join(", ") : "nenhuma";
    if (typeof v === "object") return Object.entries(v).map(([k, val]) => `${k}:${val}`).join(" ");
    return String(v);
  };

  $("#atSummaryTableBody").innerHTML = (proposal.changes || [])
    .map((c) => {
      const modeBadge = c.mode === "auto" ? ` <span class="badge-est">auto</span>` : "";
      return `
        <tr>
          <td><strong>${c.label}</strong>${modeBadge}</td>
          <td class="muted">${fmt(c.current_value)}</td>
          <td><strong>${fmt(c.proposed_value)}</strong></td>
          <td class="hint">${c.how_obtained}</td>
        </tr>
      `;
    })
    .join("");
  setWizardStepIcon(6, "ok");
}

function initAutoTuneWizard() {
  $("#btnOpenAutoTuneTop").addEventListener("click", openAutoTuneModal);
  $("#btnOpenAutoTuneConfig").addEventListener("click", openAutoTuneModal);
  $("#btnCloseAutoTune").addEventListener("click", closeAutoTuneModal);
  $("#btnAtDiscard").addEventListener("click", closeAutoTuneModal);

  $$('input[name="atMode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const isLive = radio.value === "live";
      $("#atLiveDeviceBox").classList.toggle("hidden", !isLive);
      $("#atDatasetBox").classList.toggle("hidden", isLive);
      if (!isLive) stopAutoTuneAudio();
    });
  });

  $("#btnAtRefreshDevices").addEventListener("click", async () => {
    await refreshAudioDevices("#atDeviceSelect");
    await startAutoTuneMonitor();
  });
  $("#atDeviceSelect").addEventListener("change", startAutoTuneMonitor);

  $("#btnAtStep1Next").addEventListener("click", handleAutoTuneStep1Next);
  $("#btnAtCaptureSilence").addEventListener("click", handleAutoTuneSilence);
  $("#btnAtStep2Next").addEventListener("click", () => setWizardStep(3));
  $("#btnAtCaptureLevel").addEventListener("click", handleAutoTuneLevel);
  $("#btnAtStep3Next").addEventListener("click", () => setWizardStep(4));

  $("#btnAtStartTuning").addEventListener("click", startClickFreeTuning);
  $("#btnAtRepeatTuning").addEventListener("click", () => {
    if (state.autotune.ws && state.autotune.ws.readyState === WebSocket.OPEN) {
      const curS = state.autotune.tuningOrder[Math.min(state.autotune.tuningIdx, 5)];
      state.autotune.ws.send(JSON.stringify({ type: "set_step", step: "tuning", string: curS, fret: 0 }));
    }
  });
  $("#btnAtStep4Next").addEventListener("click", () => {
    stopAutoTuneAudio();
    setWizardStep(5);
  });

  $("#btnAtStartCal").addEventListener("click", startClickFreeCalibration);
  $("#btnAtRepeatCal").addEventListener("click", () => {
    if (state.autotune.ws && state.autotune.ws.readyState === WebSocket.OPEN && state.autotune.calQueue.length) {
      const cur = state.autotune.calQueue[Math.min(state.autotune.calIdx, state.autotune.calQueue.length - 1)];
      state.autotune.ws.send(JSON.stringify({ type: "set_step", step: "calibration", string: cur.string, fret: cur.fret }));
    }
  });

  $("#btnAtStep5Next").addEventListener("click", async () => {
    stopAutoTuneAudio();
    const prop = await api(`/api/autotune/session/${state.autotune.sessionId}/proposal`);
    renderAutoTuneProposal(prop);
    setWizardStep(6);
  });

  $$("[data-wgoto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      stopAutoTuneAudio();
      setWizardStep(parseInt(btn.dataset.wgoto, 10));
    });
  });

  $("#btnAtApplyActive").addEventListener("click", async () => {
    try {
      await api(`/api/autotune/session/${state.autotune.sessionId}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "apply_active" }),
      });
      closeAutoTuneModal();
      await loadProfilesList();
      alert("Proposta aplicada ao perfil ativo com sucesso!");
    } catch (err) {
      alert(`Erro ao aplicar: ${err.message}`);
    }
  });

  $("#btnAtSaveAsNew").addEventListener("click", async () => {
    const newName = prompt("Nome do novo perfil (sem espaços/barras):", `${state.activeProfile}_autotune`);
    if (!newName) return;
    try {
      await api(`/api/autotune/session/${state.autotune.sessionId}/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "save_as_new", new_name: newName.trim() }),
      });
      closeAutoTuneModal();
      await loadProfilesList(newName.trim());
      alert(`Novo perfil '${newName.trim()}' criado e ativado!`);
    } catch (err) {
      alert(`Erro ao salvar novo perfil: ${err.message}`);
    }
  });
}

/* ==================== BOOT ==================== */
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initFileAndDataset();
  initConfigEvents();
  initAutoTuneWizard();

  $("#btnRefreshDevices").addEventListener("click", () => refreshAudioDevices("#audioDeviceSelect"));
  $("#btnLiveToggle").addEventListener("click", toggleLiveCapture);
  const retryBtn = $("#btnLiveRetry");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => {
      clearLiveError();
      toggleLiveCapture();
    });
  }
  if (state.live.debugEnabled) {
    liveDebugLog("Modo diagnóstico (?debug=1) ativo");
  }
  $("#btnLiveAnalyzeNow").addEventListener("click", () => {
    if (state.live.ws && state.live.ws.readyState === WebSocket.OPEN) {
      state.live.ws.send(JSON.stringify({ type: "trigger" }));
    }
  });
  $("#btnApplyCalibration").addEventListener("click", applyCalibration);

  await loadProfilesList();
  await loadDatasetFiles();
  drawSpectrum(null, []);
});
