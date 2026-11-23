"use strict";

const state = {
  activeProfile: "egfxset",
  editingProfile: null,
  profiles: [],
  datasetFiles: [],
  live: {
    running: false,
    audioCtx: null,
    stream: null,
    processor: null,
    ws: null,
  },
  calibration: {
    steps: [],
    results: {},
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
        refreshAudioDevices();
      }
    });
  });
}

/* ==================== EXIBIÇÃO DE RESULTADO ==================== */
function renderResult(data) {
  $("#resNote").textContent = data.note || "?";
  $("#resF0").textContent = data.f0
    ? `f0 medida: ${data.f0.toFixed(2)} Hz`
    : "Frequência fundamental não detectada";

  if (data.indeterminate || data.string === null || data.fret === null) {
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

  // Lista de candidatos
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
  const yOf = (db) => padT + (1 - (Math.max(minDb, Math.min(maxDb, db)) - minDb) / (maxDb - minDb)) * plotH;

  // Grade horizontal
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

  // Harmônicos ideais (linhas tracejadas)
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

  // Curva do espectro
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

  // Pontos dos parciais medidos
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
    ctx.fillText(`n=${p.n} (${sign}${p.deviation_cents.toFixed(1)}c)`, x - 18, Math.max(padT + 10, y - 9));
  });

  // Eixo X (frequência)
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

  // Popula dropdown de captadores apenas na primeira vez
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
    list.innerHTML = `<p class="muted" style="padding:12px">Dataset EGFxSet não encontrado em <code>data/egfxset/Clean</code>. Rode <code>uv run python scripts/download_egfxset.py</code>.</p>`;
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

/* ==================== CAPTURA AO VIVO (WEB AUDIO API) ==================== */
async function refreshAudioDevices() {
  const sel = $("#audioDeviceSelect");
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    sel.innerHTML = `<option value="">Web Audio API indisponível neste navegador</option>`;
    return;
  }
  try {
    // Pede permissão breve para revelar os nomes reais das placas de captura
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch (_) {}

  const devices = await navigator.mediaDevices.enumerateDevices();
  const inputs = devices.filter((d) => d.kind === "audioinput");
  sel.innerHTML = "";
  if (!inputs.length) {
    sel.innerHTML = `<option value="">Nenhuma entrada de áudio detectada</option>`;
    return;
  }
  inputs.forEach((dev, idx) => {
    const opt = document.createElement("option");
    opt.value = dev.deviceId;
    opt.textContent = dev.label || `Entrada de áudio #${idx + 1}`;
    sel.appendChild(opt);
  });
}

async function toggleLiveCapture() {
  if (state.live.running) {
    stopLiveCapture();
    return;
  }
  const deviceId = $("#audioDeviceSelect").value;
  const constraints = {
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
      sampleRate: { ideal: 96000 },
    },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioCtx();
    const actualSr = audioCtx.sampleRate;
    $("#liveSampleRateInfo").textContent = `Taxa real obtida: ${actualSr} Hz`;

    const wsProto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${wsProto}://${location.host}/ws/live`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: "config",
          sample_rate: actualSr,
          profile: state.activeProfile,
        })
      );
      $("#liveStatus").textContent = `Capturando (${actualSr} Hz)...`;
      $("#btnLiveToggle").textContent = "Parar captura";
      $("#btnLiveAnalyzeNow").disabled = false;
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "level") {
        $("#meterDb").textContent = `${msg.rms_db.toFixed(1)} dB`;
        const pct = Math.max(0, Math.min(100, ((msg.rms_db + 60) / 60) * 100));
        $("#meterFill").style.width = `${pct}%`;
        $("#clippingBadge").classList.toggle("hidden", !msg.clipping);
      } else if (msg.type === "result") {
        renderResult(msg.data);
      }
    };

    ws.onclose = () => {
      if (state.live.running) stopLiveCapture();
    };

    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      if (ws.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        ws.send(new Float32Array(inputData).buffer);
      }
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);

    state.live = { running: true, audioCtx, stream, processor, ws };
  } catch (err) {
    alert(`Erro ao iniciar captura ao vivo: ${err.message}`);
  }
}

function stopLiveCapture() {
  const l = state.live;
  if (l.processor) l.processor.disconnect();
  if (l.audioCtx) l.audioCtx.close();
  if (l.stream) l.stream.getTracks().forEach((t) => t.stop());
  if (l.ws && l.ws.readyState <= 1) l.ws.close();
  state.live = { running: false, audioCtx: null, stream: null, processor: null, ws: null };
  $("#btnLiveToggle").textContent = "Iniciar captura";
  $("#btnLiveAnalyzeNow").disabled = true;
  $("#liveStatus").textContent = "Parado";
  $("#meterFill").style.width = "0%";
  $("#clippingBadge").classList.add("hidden");
}

/* ==================== PERFIS & CONFIGURAÇÃO ==================== */
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

  $("#cfgModelName").value = prof.instrument.model_name;
  $("#cfgNumFrets").value = prof.instrument.num_frets;
  $("#cfgScaleMm").value = prof.instrument.scale_length_mm;
  $("#cfgPickup").value = prof.instrument.pickup_config || "";

  const tbody = $("#cfgStringsBody");
  tbody.innerHTML = "";
  for (let s = 1; s <= 6; s++) {
    const note = prof.instrument.tuning[s] || "";
    const spec = (prof.instrument.strings && prof.instrument.strings[s]) || {};
    tbody.innerHTML += `
      <tr>
        <td><strong>${s}</strong></td>
        <td><input type="text" data-str-tune="${s}" value="${note}" required /></td>
        <td>
          <select data-str-type="${s}">
            <option value="" ${!spec.string_type ? "selected" : ""}>A estimar</option>
            <option value="plain" ${spec.string_type === "plain" ? "selected" : ""}>Lisa (plain)</option>
            <option value="wound" ${spec.string_type === "wound" ? "selected" : ""}>Encapada (wound)</option>
          </select>
        </td>
        <td>
          <input type="number" step="0.001" min="0.006" max="0.080"
                 data-str-gauge="${s}" placeholder="A estimar"
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
      <strong>Sem calibração salva.</strong> Use a aba <em>Calibração Guiada</em> para estimar B0 por corda.
    `;
  }
}

function collectProfileFromForm() {
  const prof = structuredClone(state.editingProfile);
  prof.description = $("#cfgDescription").value;
  prof.capture = {
    source_mode: $("#cfgSourceMode").value,
    device_name: $("#cfgDeviceName").value.trim() || null,
    sample_rate: parseInt($("#cfgSampleRate").value, 10),
    channel: parseInt($("#cfgChannel").value, 10),
    reference_gain_db: parseFloat($("#cfgGain").value),
    buffer_size: parseInt($("#cfgBufferSize").value, 10),
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
    model_name: $("#cfgModelName").value.trim(),
    tuning,
    num_frets: parseInt($("#cfgNumFrets").value, 10),
    scale_length_mm: parseFloat($("#cfgScaleMm").value),
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

/* ==================== CALIBRAÇÃO GUIADA ==================== */
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
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioCtx();
    const sr = ctx.sampleRate;
    const targetLen = Math.round(1.2 * sr);
    const chunks = [];
    let collected = 0;

    await new Promise((resolve) => {
      const src = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      proc.onaudioprocess = (e) => {
        const ch = new Float32Array(e.inputBuffer.getChannelData(0));
        chunks.push(ch);
        collected += ch.length;
        if (collected >= targetLen) {
          proc.disconnect();
          src.disconnect();
          stream.getTracks().forEach((t) => t.stop());
          ctx.close();
          resolve();
        }
      };
      src.connect(proc);
      proc.connect(ctx.destination);
    });

    const merged = new Float32Array(collected);
    let offset = 0;
    chunks.forEach((c) => {
      merged.set(c, offset);
      offset += c.length;
    });

    const res = await api("/api/calibration/measure-pcm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile: state.activeProfile,
        string,
        fret,
        sample_rate: sr,
        samples: Array.from(merged),
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

/* ==================== BOOT ==================== */
document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  initFileAndDataset();
  initConfigEvents();

  $("#btnRefreshDevices").addEventListener("click", refreshAudioDevices);
  $("#btnLiveToggle").addEventListener("click", toggleLiveCapture);
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
