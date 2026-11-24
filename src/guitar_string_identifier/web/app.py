"""Backend FastAPI (REST + WebSocket) para o Identificador de Corda e Casa."""

from __future__ import annotations

import io
import json
import math
import time
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from guitar_string_identifier.autotune.device_probe import DeviceProbe
from guitar_string_identifier.autotune.session import AutoTuneProposal, AutoTuneSession
from guitar_string_identifier.calibration.calibrator import (
    CalibrationSampleResult,
    GuitarCalibrator,
)
from guitar_string_identifier.core.music import (
    freq_to_cents,
    midi_to_hz,
    string_fret_to_midi,
    string_fret_to_note,
)
from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import Note, Result
from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.profiles.store import ProfileStore
from guitar_string_identifier.sources.dataset_source import (
    EGFxSetLabelProvider,
    list_egfxset_files,
)
from guitar_string_identifier.sources.file_source import FileAudioSource

STATIC_DIR = Path(__file__).parent / "static"


class RenameRequest(BaseModel):
    """Corpo de requisição para renomear ou duplicar um perfil."""

    new_name: str


class DatasetAnalyzeRequest(BaseModel):
    """Corpo de requisição para analisar um arquivo local do dataset EGFxSet."""

    rel_path: str
    profile: str | None = None


class CalibrationApplySample(BaseModel):
    """Amostra individual já medida enviada para consolidação na calibração."""

    string: int
    fret: int
    accepted: bool
    reason: str = "Aceita."
    f0: float | None = None
    b_measured: float | None = None
    b0_implied: float | None = None
    residual_cents: float = 0.0
    n_partials: int = 0
    decay_mean_db_s: float | None = None
    centroid_ratio: float | None = None


class CalibrationApplyRequest(BaseModel):
    """Requisição para consolidar e salvar a calibração no perfil selecionado."""

    profile: str | None = None
    samples: list[CalibrationApplySample]
    notes_label: str = "Calibração guiada via interface web."


class CalibrationPcmRequest(BaseModel):
    """Requisição com bloco PCM em memória para medir uma nota de calibração."""

    profile: str | None = None
    string: int
    fret: int
    sample_rate: int
    samples: list[float]


class AutoTuneCreateRequest(BaseModel):
    """Parâmetros iniciais para abrir uma sessão do assistente Auto Tuning."""

    mode: Literal["live", "dataset"] = "live"
    profile: str | None = None


class DeviceRankRequest(BaseModel):
    """Lista de dispositivos de entrada do navegador para ordenação heurística."""

    devices: list[dict[str, str]]


class AutoTuneDeviceRequest(BaseModel):
    """Dados da placa, taxa real, canal e latência medidos na Etapa 1 do Auto Tuning."""

    device_name: str | None = None
    sample_rate: int = 48000
    channel: int = 0
    base_latency_sec: float | None = None


class AutoTuneSilenceRequest(BaseModel):
    """Áudio de silêncio (~2 s) enviado na Etapa 2 para piso de ruído e ronco 50/60 Hz."""

    sample_rate: int = 48000
    samples: list[float]


class AutoTuneLevelRequest(BaseModel):
    """Áudio de nota forte enviado na Etapa 3 para checagem de nível e clipping."""

    samples: list[float]


class AutoTuneTuningRequest(BaseModel):
    """Áudio de uma corda solta enviado na Etapa 4 para detecção de afinação."""

    string: int
    sample_rate: int = 48000
    samples: list[float]


class AutoTuneCalibrationRequest(BaseModel):
    """Áudio de uma nota `(corda, casa)` enviado na Etapa 5 do Auto Tuning."""

    string: int
    fret: int
    sample_rate: int = 48000
    samples: list[float]


class AutoTuneDatasetRequest(BaseModel):
    """Configuração do Auto Tuning offline restrito às casas de calibração do dataset."""

    cal_frets: list[int] = [0, 12]
    pickup: str | None = None
    target_error_rate: float = 0.10


class AutoTuneApplyRequest(BaseModel):
    """Confirmação explícita do usuário para aplicar a proposta ou salvar novo perfil."""

    action: Literal["apply_active", "save_as_new"] = "apply_active"
    new_name: str | None = None


def _serialize_proposal(prop: AutoTuneProposal) -> dict[str, Any]:
    return {
        "session_id": prop.session_id,
        "mode": prop.mode,
        "base_profile_name": prop.base_profile_name,
        "step_statuses": prop.step_statuses,
        "warnings": prop.warnings,
        "changes": [
            {
                "param": c.param,
                "label": c.label,
                "current_value": c.current_value,
                "proposed_value": c.proposed_value,
                "mode": c.mode,
                "how_obtained": c.how_obtained,
            }
            for c in prop.changes
        ],
        "proposed_profile": prop.proposed_profile.model_dump(mode="json"),
    }


def _serialize_result(result: Result) -> dict[str, Any]:
    """Converte Result em dicionário JSON amigável para o frontend."""
    feats = result.features
    partials_json = []
    if feats and feats.get("partials"):
        for p in feats.get("partials"):
            partials_json.append(
                {
                    "n": p.n,
                    "ideal_freq": round(p.ideal_freq, 2),
                    "measured_freq": round(p.measured_freq, 2),
                    "magnitude_db": round(p.magnitude_db, 2),
                    "deviation_cents": round(p.deviation_cents, 2),
                    "decay_rate": p.decay_rate,
                }
            )

    candidates_json = [
        {
            "string": c.string,
            "fret": c.fret,
            "note": c.note,
            "expected_f0": round(c.expected_f0, 2),
            "score": round(c.score, 4),
            "distance": round(c.distance, 3) if math.isfinite(c.distance) else None,
            "expected_b": c.expected_b,
        }
        for c in result.candidates
    ]

    label_json = None
    comparison = {
        "has_label": False,
        "status": "no_label",
        "string_match": None,
        "fret_match": None,
        "note_match": None,
    }
    if result.label is not None:
        lbl = result.label
        label_json = {
            "string": lbl.string,
            "fret": lbl.fret,
            "note": lbl.note,
            "pickup": lbl.pickup,
            "effect": lbl.effect,
        }
        s_ok = result.string == lbl.string
        f_ok = result.fret == lbl.fret
        n_ok = result.note == lbl.note
        if result.indeterminate:
            status = "indeterminate"
        elif s_ok and f_ok:
            status = "correct"
        else:
            status = "wrong"
        comparison = {
            "has_label": True,
            "status": status,
            "string_match": s_ok,
            "fret_match": f_ok,
            "note_match": n_ok,
        }

    b_res = feats.get("B_residual_cents") if feats else None
    return {
        "string": result.string,
        "fret": result.fret,
        "note": result.note,
        "f0": round(result.f0, 2) if result.f0 else None,
        "confidence": round(result.confidence, 4),
        "indeterminate": result.indeterminate,
        "input_warning": result.input_warning,
        "provisional": result.provisional,
        "candidates": candidates_json,
        "label": label_json,
        "comparison": comparison,
        "features": {
            "B": feats.get("B") if feats else None,
            "B_std": (
                feats.get("B_std")
                if feats and feats.get("B_std") is not None and math.isfinite(feats.get("B_std"))
                else None
            ),
            "B_residual_cents": (
                round(b_res, 2) if b_res is not None and math.isfinite(b_res) else None
            ),
            "B_r2": round(feats.get("B_r2", 0.0), 4) if feats else None,
            "n_partials_used": feats.get("n_partials_used", 0) if feats else 0,
            "decay_mean_db_s": (
                round(feats.get("decay_mean_db_s"), 2)
                if feats and feats.get("decay_mean_db_s") is not None
                else None
            ),
            "decay_ratio_high_low": (
                round(feats.get("decay_ratio_high_low"), 3)
                if feats and feats.get("decay_ratio_high_low") is not None
                else None
            ),
            "pluck_position_rel": feats.get("pluck_position_rel") if feats else None,
            "spectral_centroid_hz": (
                round(feats.get("spectral_centroid_hz"), 1)
                if feats and feats.get("spectral_centroid_hz") is not None
                else None
            ),
            "spectral_rolloff_hz": (
                round(feats.get("spectral_rolloff_hz"), 1)
                if feats and feats.get("spectral_rolloff_hz") is not None
                else None
            ),
        },
        "partials": partials_json,
        "spectrum": {
            "freqs": feats.get("spectrum_freqs", []) if feats else [],
            "db": feats.get("spectrum_db", []) if feats else [],
        },
    }


def create_app(
    profiles_dir: Path | str = "configs/profiles",
    dataset_dir: Path | str = "data/egfxset",
) -> FastAPI:
    """Factory do aplicativo FastAPI (facilita testes isolados)."""
    app = FastAPI(
        title="Identificador de Corda e Casa de Guitarra",
        description="Processamento clássico de sinais (sem ML) para identificação de corda e casa.",
        version="0.2.0",
    )

    store = ProfileStore(profiles_dir)
    dataset_root = Path(dataset_dir)
    autotune_sessions: dict[str, AutoTuneSession] = {}

    state: dict[str, str] = {"active_profile": "egfxset"}
    if not store.exists("egfxset"):
        names = store.list_profiles()
        if names:
            state["active_profile"] = names[0]
        else:
            default_p = GuitarProfile(name="egfxset", description="Perfil padrão.")
            store.save(default_p)

    def _get_profile(name: str | None = None) -> GuitarProfile:
        target = name or state["active_profile"]
        try:
            return store.load(target)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _get_session(sid: str) -> AutoTuneSession:
        sess = autotune_sessions.get(sid)
        if sess is None:
            raise HTTPException(
                status_code=404, detail=f"Sessão Auto Tuning '{sid}' não encontrada."
            )
        return sess

    # --- Frontend estático (com Cache-Control: no-cache para evitar JS/HTML desatualizado) ---
    @app.middleware("http")
    async def no_cache_static_middleware(request: Request, call_next: Any) -> Response:
        """Garante que HTML/JS/CSS atualizados cheguem imediatamente ao navegador."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "active_profile": state["active_profile"],
            "dataset_available": (dataset_root / "Clean").exists(),
        }

    # --- Perfis CRUD ---
    @app.get("/api/profiles")
    async def list_profiles() -> dict[str, Any]:
        return {
            "active": state["active_profile"],
            "profiles": store.list_profiles(),
        }

    @app.get("/api/profiles/{name}")
    async def get_profile(name: str) -> dict[str, Any]:
        return _get_profile(name).model_dump(mode="json")

    @app.post("/api/profiles")
    async def create_or_save_profile(profile: GuitarProfile) -> dict[str, Any]:
        store.save(profile)
        return profile.model_dump(mode="json")

    @app.put("/api/profiles/{name}")
    async def update_profile(name: str, profile: GuitarProfile) -> dict[str, Any]:
        if profile.name != name:
            profile = profile.model_copy(update={"name": name})
        store.save(profile)
        return profile.model_dump(mode="json")

    @app.post("/api/profiles/{name}/activate")
    async def activate_profile(name: str) -> dict[str, Any]:
        if not store.exists(name):
            raise HTTPException(status_code=404, detail=f"Perfil '{name}' não encontrado.")
        state["active_profile"] = name
        return {"active": name}

    @app.post("/api/profiles/{name}/duplicate")
    async def duplicate_profile(name: str, req: RenameRequest) -> dict[str, Any]:
        try:
            cloned = store.duplicate(name, req.new_name)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return cloned.model_dump(mode="json")

    @app.post("/api/profiles/{name}/rename")
    async def rename_profile(name: str, req: RenameRequest) -> dict[str, Any]:
        try:
            renamed = store.rename(name, req.new_name)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if state["active_profile"] == name:
            state["active_profile"] = renamed.name
        return renamed.model_dump(mode="json")

    @app.delete("/api/profiles/{name}")
    async def delete_profile(name: str) -> dict[str, Any]:
        all_names = store.list_profiles()
        if len(all_names) <= 1:
            raise HTTPException(
                status_code=400,
                detail="Não é possível excluir o único perfil restante.",
            )
        try:
            store.delete(name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if state["active_profile"] == name:
            remaining = store.list_profiles()
            state["active_profile"] = remaining[0]
        return {"deleted": name, "active": state["active_profile"]}

    # --- Navegador do Dataset EGFxSet ---
    @app.get("/api/dataset/files")
    async def get_dataset_files(
        pickup: str | None = None,
        string: int | None = None,
        fret: int | None = None,
    ) -> dict[str, Any]:
        items = list_egfxset_files(
            dataset_root,
            effect="Clean",
            pickup=pickup or None,
            string=string,
            fret=fret,
        )
        pickups = sorted({lbl.pickup for _, lbl in items if lbl.pickup})
        files_json = [
            {
                "rel_path": str(p.relative_to(dataset_root)).replace("\\", "/"),
                "name": p.name,
                "string": lbl.string,
                "fret": lbl.fret,
                "note": lbl.note,
                "pickup": lbl.pickup,
                "effect": lbl.effect,
            }
            for p, lbl in items
        ]
        return {
            "available": (dataset_root / "Clean").exists(),
            "count": len(files_json),
            "pickups": pickups,
            "files": files_json,
        }

    @app.post("/api/analyze/dataset")
    async def analyze_dataset_file(req: DatasetAnalyzeRequest) -> dict[str, Any]:
        profile = _get_profile(req.profile)
        target = (dataset_root / req.rel_path).resolve()
        root_resolved = dataset_root.resolve()
        if not str(target).startswith(str(root_resolved)) or not target.exists():
            raise HTTPException(status_code=404, detail="Arquivo do dataset não encontrado.")

        pipeline = GuitarIdentifierPipeline(
            profile,
            label_provider=EGFxSetLabelProvider(tuning=profile.instrument.tuning),
            whole_file=True,
        )
        result = pipeline.analyze_file(target)
        return _serialize_result(result)

    # --- Upload de arquivo de áudio ---
    @app.post("/api/analyze/upload")
    async def analyze_upload(
        file: Annotated[UploadFile, File(...)],
        profile: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        prof = _get_profile(profile)
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")

        try:
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Não foi possível ler o arquivo de áudio: {exc}",
            ) from exc

        ch = prof.capture.channel
        mono = data[:, ch] if 0 <= ch < data.shape[1] else np.mean(data, axis=1)

        label_provider = EGFxSetLabelProvider(tuning=prof.instrument.tuning)
        pipeline = GuitarIdentifierPipeline(
            prof,
            label_provider=label_provider,
            whole_file=True,
            reject_bad_input=True,
        )
        results = pipeline.analyze_audio(
            np.asarray(mono, dtype=np.float64),
            int(sr),
            source_path=file.filename,
        )
        if not results:
            raise HTTPException(status_code=422, detail="Nenhuma nota detectada no áudio.")

        res = results[0]
        if file.filename:
            res.label = label_provider.get_label(Path(file.filename))
        return _serialize_result(res)

    # --- Calibração guiada (existente) ---
    @app.get("/api/calibration/plan")
    async def calibration_plan(profile: str | None = None) -> dict[str, Any]:
        prof = _get_profile(profile)
        steps = []
        for s in sorted(prof.instrument.tuning.keys()):
            for fret in (0, 12):
                if fret <= prof.instrument.num_frets:
                    midi = string_fret_to_midi(s, fret, tuning=prof.instrument.tuning)
                    steps.append(
                        {
                            "string": s,
                            "fret": fret,
                            "note": string_fret_to_note(s, fret, tuning=prof.instrument.tuning),
                            "expected_f0": round(midi_to_hz(midi, a4_hz=prof.instrument.a4_hz), 2),
                        }
                    )
        return {"profile": prof.name, "steps": steps}

    @app.post("/api/calibration/measure-pcm")
    async def calibration_measure_pcm(req: CalibrationPcmRequest) -> dict[str, Any]:
        prof = _get_profile(req.profile)
        arr = np.asarray(req.samples, dtype=np.float64)
        if arr.size < 2048:
            raise HTTPException(status_code=400, detail="Áudio muito curto para calibração.")
        calibrator = GuitarCalibrator(analysis_config=prof.analysis)
        sample_res = calibrator.measure_sample(
            Note(audio=arr, sr=req.sample_rate), req.string, req.fret
        )
        return {
            "string": sample_res.string,
            "fret": sample_res.fret,
            "accepted": sample_res.accepted,
            "reason": sample_res.reason,
            "f0": round(sample_res.f0, 2) if sample_res.f0 else None,
            "b_measured": sample_res.b_measured,
            "b0_implied": sample_res.b0_implied,
            "residual_cents": (
                round(sample_res.residual_cents, 2)
                if math.isfinite(sample_res.residual_cents)
                else None
            ),
            "n_partials": sample_res.n_partials,
            "decay_mean_db_s": sample_res.decay_mean_db_s,
            "centroid_ratio": sample_res.centroid_ratio,
        }

    @app.post("/api/calibration/measure-upload")
    async def calibration_measure_upload(
        file: Annotated[UploadFile, File(...)],
        string: Annotated[int, Form(...)],
        fret: Annotated[int, Form(...)],
        profile: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        prof = _get_profile(profile)
        raw_bytes = await file.read()
        try:
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float64", always_2d=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Erro ao ler áudio: {exc}") from exc
        mono = np.mean(data, axis=1)
        calibrator = GuitarCalibrator(analysis_config=prof.analysis)
        sample_res = calibrator.measure_sample(Note(audio=mono, sr=int(sr)), string, fret)
        return {
            "string": sample_res.string,
            "fret": sample_res.fret,
            "accepted": sample_res.accepted,
            "reason": sample_res.reason,
            "f0": round(sample_res.f0, 2) if sample_res.f0 else None,
            "b_measured": sample_res.b_measured,
            "b0_implied": sample_res.b0_implied,
            "residual_cents": (
                round(sample_res.residual_cents, 2)
                if math.isfinite(sample_res.residual_cents)
                else None
            ),
            "n_partials": sample_res.n_partials,
            "decay_mean_db_s": sample_res.decay_mean_db_s,
            "centroid_ratio": sample_res.centroid_ratio,
        }

    @app.post("/api/calibration/apply")
    async def calibration_apply(req: CalibrationApplyRequest) -> dict[str, Any]:
        prof = _get_profile(req.profile)
        calibrator = GuitarCalibrator(analysis_config=prof.analysis)
        samples = [
            CalibrationSampleResult(
                string=s.string,
                fret=s.fret,
                accepted=s.accepted,
                reason=s.reason,
                f0=s.f0,
                b_measured=s.b_measured,
                b0_implied=s.b0_implied,
                residual_cents=s.residual_cents,
                n_partials=s.n_partials,
                decay_mean_db_s=s.decay_mean_db_s,
                centroid_ratio=s.centroid_ratio,
            )
            for s in req.samples
        ]
        report = calibrator.calibrate_from_samples(samples, notes_label=req.notes_label)
        if report.accepted_count == 0:
            raise HTTPException(
                status_code=400,
                detail="Nenhuma nota válida para aplicar na calibração.",
            )
        prof.calibration = report.calibration
        store.save(prof)
        return {
            "profile": prof.model_dump(mode="json"),
            "accepted": report.accepted_count,
            "rejected": report.rejected_count,
        }

    # ==================== AUTO TUNING (REST) ====================
    @app.post("/api/autotune/rank-devices")
    async def autotune_rank_devices(req: DeviceRankRequest) -> dict[str, Any]:
        probe = DeviceProbe()
        ranked = probe.rank_devices(req.devices)
        return {
            "devices": [
                {
                    "device_id": d.device_id,
                    "label": d.label,
                    "score": d.score,
                    "reason": d.reason,
                }
                for d in ranked
            ]
        }

    @app.post("/api/autotune/session")
    async def autotune_create_session(req: AutoTuneCreateRequest) -> dict[str, Any]:
        prof = _get_profile(req.profile)
        sess = AutoTuneSession(prof, mode=req.mode)
        autotune_sessions[sess.session_id] = sess
        return {
            "session_id": sess.session_id,
            "mode": sess.mode,
            "profile": prof.name,
            "step_statuses": sess.step_statuses,
            "warnings": sess.warnings,
        }

    @app.post("/api/autotune/session/{sid}/device")
    async def autotune_step_device(sid: str, req: AutoTuneDeviceRequest) -> dict[str, Any]:
        sess = _get_session(sid)
        return sess.apply_device_step(
            device_name=req.device_name,
            sample_rate=req.sample_rate,
            channel=req.channel,
            base_latency_sec=req.base_latency_sec,
        )

    @app.post("/api/autotune/session/{sid}/silence")
    async def autotune_step_silence(sid: str, req: AutoTuneSilenceRequest) -> dict[str, Any]:
        sess = _get_session(sid)
        arr = np.asarray(req.samples, dtype=np.float64)
        return sess.apply_silence_step(arr, req.sample_rate)

    @app.post("/api/autotune/session/{sid}/level")
    async def autotune_step_level(sid: str, req: AutoTuneLevelRequest) -> dict[str, Any]:
        sess = _get_session(sid)
        arr = np.asarray(req.samples, dtype=np.float64)
        return sess.apply_level_step(arr)

    @app.post("/api/autotune/session/{sid}/tuning")
    async def autotune_step_tuning(sid: str, req: AutoTuneTuningRequest) -> dict[str, Any]:
        sess = _get_session(sid)
        arr = np.asarray(req.samples, dtype=np.float64)
        return sess.record_tuning_string(req.string, arr, req.sample_rate)

    @app.post("/api/autotune/session/{sid}/calibration")
    async def autotune_step_calibration(
        sid: str, req: AutoTuneCalibrationRequest
    ) -> dict[str, Any]:
        sess = _get_session(sid)
        arr = np.asarray(req.samples, dtype=np.float64)
        return sess.record_calibration_note(req.string, req.fret, arr, req.sample_rate)

    @app.post("/api/autotune/session/{sid}/dataset-auto")
    async def autotune_dataset_auto(sid: str, req: AutoTuneDatasetRequest) -> dict[str, Any]:
        """Executa Auto Tuning offline no dataset usando APENAS as casas de calibração."""
        sess = _get_session(sid)
        items = list_egfxset_files(dataset_root, effect="Clean", pickup=req.pickup)
        if not items:
            raise HTTPException(
                status_code=404,
                detail="Dataset EGFxSet não encontrado em data/egfxset/Clean.",
            )

        cal_frets = set(req.cal_frets)
        cal_items = [(p, lbl) for p, lbl in items if lbl.fret in cal_frets]
        if not cal_items:
            raise HTTPException(
                status_code=400,
                detail=f"Nenhum arquivo nas casas de calibração {sorted(cal_frets)}.",
            )

        # 1. Taxa de amostragem lida diretamente da fonte (modo auto)
        first_audio, first_sr = FileAudioSource(cal_items[0][0]).read()
        sess.proposed.capture.sample_rate = first_sr
        sess.proposed.param_modes["sample_rate"] = "auto"
        sess.reasons["capture.sample_rate"] = f"Lida diretamente dos arquivos WAV ({first_sr} Hz)"

        # 2. Afinação e referência A4 a partir das cordas soltas (casa 0, primeiro captador)
        open_by_string: dict[int, Path] = {}
        for p, lbl in cal_items:
            if lbl.fret == 0 and lbl.string not in open_by_string:
                open_by_string[lbl.string] = p
        for s_num, path in sorted(open_by_string.items()):
            aud, sr = FileAudioSource(path).read()
            sess.record_tuning_string(s_num, aud, sr)

        # 3. Calibração de B0, decaimento e pesos (Fisher) exclusivamente nas casas de calibração
        cal_loaded: list[tuple[Path, Any, np.ndarray, int]] = []
        for p, lbl in cal_items:
            aud, sr = FileAudioSource(p).read()
            cal_loaded.append((p, lbl, aud, sr))
            sess.record_calibration_note(
                lbl.string, lbl.fret, aud, sr, sample_key=lbl.pickup or str(p)
            )

        # 4. Validação interna por captador (leave-one-pickup-out) nas casas de calibração
        # para calibrar min_confidence e min_margin sem tocar nas casas de teste!
        pickups_in_cal = sorted({lbl.pickup or "default" for _, lbl, _, _ in cal_loaded})
        val_records: list[dict[str, Any]] = []
        if len(pickups_in_cal) >= 2:
            for held_pk in pickups_in_cal:
                train_subset = [
                    (Note(audio=aud, sr=sr), lbl.string, lbl.fret)
                    for _, lbl, aud, sr in cal_loaded
                    if (lbl.pickup or "default") != held_pk
                ]
                val_subset = [
                    (lbl, aud, sr)
                    for _, lbl, aud, sr in cal_loaded
                    if (lbl.pickup or "default") == held_pk
                ]
                fold_prof = sess.proposed.model_copy(deep=True)
                fold_cal = GuitarCalibrator(analysis_config=fold_prof.analysis).calibrate(
                    train_subset
                )
                fold_prof.calibration = fold_cal.calibration
                fold_pipe = GuitarIdentifierPipeline(fold_prof, whole_file=True)
                for lbl, aud, sr in val_subset:
                    res = fold_pipe.analyze_note(Note(audio=aud, sr=sr))
                    if len(res.candidates) >= 2:
                        top1 = res.candidates[0]
                        top2 = res.candidates[1]
                        val_records.append(
                            {
                                "confidence": res.confidence,
                                "margin": top1.score - top2.score,
                                "correct": top1.string == lbl.string and top1.fret == lbl.fret,
                                "n_candidates": len(res.candidates),
                            }
                        )

        sess.proposed.analysis.target_error_rate = req.target_error_rate
        sess.apply_threshold_calibration(val_records, target_error_rate=req.target_error_rate)
        sess.enable_adaptive_analysis()

        prop = sess.build_proposal()
        return _serialize_proposal(prop)

    @app.get("/api/autotune/session/{sid}/proposal")
    async def autotune_get_proposal(sid: str) -> dict[str, Any]:
        sess = _get_session(sid)
        return _serialize_proposal(sess.build_proposal())

    @app.post("/api/autotune/session/{sid}/apply")
    async def autotune_apply(sid: str, req: AutoTuneApplyRequest) -> dict[str, Any]:
        sess = _get_session(sid)
        prop = sess.build_proposal()
        target_profile = prop.proposed_profile

        if req.action == "save_as_new":
            if not req.new_name or not req.new_name.strip():
                raise HTTPException(status_code=400, detail="Informe o nome do novo perfil.")
            new_name = req.new_name.strip()
            if store.exists(new_name):
                raise HTTPException(status_code=400, detail=f"Perfil '{new_name}' já existe.")
            target_profile = target_profile.model_copy(update={"name": new_name})
            store.save(target_profile)
            state["active_profile"] = new_name
        else:
            store.save(target_profile)
            state["active_profile"] = target_profile.name

        autotune_sessions.pop(sid, None)
        return {
            "active": state["active_profile"],
            "profile": target_profile.model_dump(mode="json"),
        }

    # ==================== WEBSOCKET: AUTO TUNING AO VIVO ====================
    @app.websocket("/ws/autotune")
    async def autotune_ws(websocket: WebSocket) -> None:
        """Streaming PCM float32 para etapas sem cliques do Auto Tuning ao vivo."""
        await websocket.accept()
        session: AutoTuneSession | None = None
        sample_rate = 48000
        step_mode = "monitor"  # monitor | tuning | calibration
        target_string = 6
        target_fret = 0

        capturing = False
        chunks: list[np.ndarray] = []
        captured_n = 0
        target_n = int(0.70 * sample_rate)
        cooldown_n = 0
        onset_db = -36.0

        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                if "text" in msg and msg["text"]:
                    payload = json.loads(msg["text"])
                    mtype = payload.get("type")
                    if mtype == "init":
                        sid = payload.get("session_id")
                        session = autotune_sessions.get(sid) if sid else None
                        sample_rate = int(payload.get("sample_rate", 48000))
                        if session and session.proposed.capture.onset_threshold_db:
                            onset_db = session.proposed.capture.onset_threshold_db
                        await websocket.send_json({"type": "ready", "sample_rate": sample_rate})
                    elif mtype == "set_step":
                        step_mode = str(payload.get("step", "monitor"))
                        target_string = int(payload.get("string", 6))
                        target_fret = int(payload.get("fret", 0))
                        target_n = int(0.75 * sample_rate)
                        capturing = False
                        chunks.clear()
                        captured_n = 0
                        cooldown_n = 0
                        await websocket.send_json(
                            {
                                "type": "step_ack",
                                "step": step_mode,
                                "string": target_string,
                                "fret": target_fret,
                            }
                        )
                    continue

                raw = msg.get("bytes")
                if not raw:
                    continue
                block = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
                if block.size == 0:
                    continue

                peak = float(np.max(np.abs(block)))
                rms = float(np.sqrt(np.mean(block**2) + 1e-18))
                rms_db = 20.0 * math.log10(rms + 1e-12)
                await websocket.send_json(
                    {
                        "type": "level",
                        "rms_db": round(rms_db, 1),
                        "peak": round(peak, 4),
                        "clipping": peak >= 0.985,
                    }
                )

                if step_mode not in ("tuning", "calibration") or session is None:
                    continue

                if cooldown_n > 0:
                    cooldown_n = max(0, cooldown_n - block.size)
                    continue

                if not capturing and rms_db > onset_db:
                    capturing = True
                    chunks = [block]
                    captured_n = block.size
                elif capturing:
                    chunks.append(block)
                    captured_n += block.size
                    if captured_n >= target_n:
                        audio = np.concatenate(chunks)
                        if step_mode == "tuning":
                            res = session.record_tuning_string(target_string, audio, sample_rate)
                            await websocket.send_json({"type": "tuning_result", "data": res})
                        elif step_mode == "calibration":
                            # Verifica se a nota tocada está próxima da f0 esperada
                            prof = session.proposed
                            exp_midi = string_fret_to_midi(
                                target_string, target_fret, tuning=prof.instrument.tuning
                            )
                            exp_f0 = midi_to_hz(exp_midi, a4_hz=prof.instrument.a4_hz)
                            res = session.record_calibration_note(
                                target_string, target_fret, audio, sample_rate
                            )
                            if res.get("f0") and abs(freq_to_cents(res["f0"], exp_f0)) > 80.0:
                                res["accepted"] = False
                                res["reason"] = (
                                    f"Nota tocada ({res['f0']:.1f} Hz) não corresponde à "
                                    f"Corda {target_string} Casa {target_fret} ({exp_f0:.1f} Hz)."
                                )
                            await websocket.send_json({"type": "calibration_result", "data": res})
                        capturing = False
                        chunks.clear()
                        captured_n = 0
                        cooldown_n = int(0.45 * sample_rate)
        except WebSocketDisconnect:
            return

    # ==================== WEBSOCKET: CAPTURA AO VIVO CONTÍNUA E PROGRESSIVA ====================
    @app.websocket("/ws/live")
    async def live_audio_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        sample_rate = 48000
        profile = _get_profile(None)
        pipeline = GuitarIdentifierPipeline(profile, whole_file=True, reject_bad_input=True)

        capturing = False
        capture_chunks: list[np.ndarray] = []
        captured_samples = 0
        prev_rms = 0.0
        cooldown_samples = 0
        next_stage_idx = 0

        # Estágios progressivos (s): primeira estimativa ~0,30 s -> refina em 0,60 s -> final 0,95 s
        stage_secs = (0.30, 0.60, 0.95)
        last_heartbeat = time.monotonic()

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "text" in message and message["text"]:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")
                    if msg_type == "config":
                        sample_rate = int(payload.get("sample_rate", 48000))
                        prof_name = payload.get("profile") or state["active_profile"]
                        profile = _get_profile(prof_name)
                        pipeline = GuitarIdentifierPipeline(
                            profile, whole_file=True, reject_bad_input=True
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
                        next_stage_idx = 0
                        last_heartbeat = time.monotonic()
                        await websocket.send_json(
                            {
                                "type": "ready",
                                "sample_rate": sample_rate,
                                "profile": profile.name,
                                "onset_threshold_db": profile.capture.onset_threshold_db,
                            }
                        )
                        await websocket.send_json(
                            {
                                "type": "status",
                                "state": "listening",
                                "sample_rate": sample_rate,
                                "profile": profile.name,
                            }
                        )
                    elif msg_type == "trigger" and capture_chunks:
                        audio = np.concatenate(capture_chunks)
                        res = pipeline.analyze_note(
                            Note(audio=audio, sr=sample_rate), provisional=False
                        )
                        await websocket.send_json(
                            {"type": "result", "data": _serialize_result(res)}
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
                        next_stage_idx = 0
                    continue

                raw_bytes = message.get("bytes")
                if not raw_bytes:
                    continue

                block = np.frombuffer(raw_bytes, dtype=np.float32).astype(np.float64)
                if block.size == 0:
                    continue

                peak = float(np.max(np.abs(block)))
                rms = float(np.sqrt(np.mean(block**2) + 1e-18))
                rms_db = 20.0 * math.log10(rms + 1e-12)
                clipping = peak >= 0.985

                await websocket.send_json(
                    {
                        "type": "level",
                        "rms_db": round(rms_db, 1),
                        "peak": round(peak, 4),
                        "clipping": clipping,
                    }
                )

                now = time.monotonic()
                if now - last_heartbeat >= 1.0:
                    last_heartbeat = now
                    await websocket.send_json(
                        {
                            "type": "heartbeat",
                            "state": "listening",
                            "rms_db": round(rms_db, 1),
                            "peak": round(peak, 4),
                            "clipping": clipping,
                        }
                    )

                onset_thresh = profile.capture.onset_threshold_db
                sustain_end_thresh = onset_thresh - 8.0
                rearm_samples = int(profile.capture.rearm_time_sec * sample_rate)

                if cooldown_samples > 0:
                    cooldown_samples = max(0, cooldown_samples - block.size)

                # Detecção contínua de ataque (sem Iniciar/Parar entre notas)
                if (
                    not capturing
                    and cooldown_samples == 0
                    and rms_db > onset_thresh
                    and (rms > prev_rms * 2.0 or rms_db > onset_thresh + 10.0)
                ):
                    capturing = True
                    capture_chunks = [block]
                    captured_samples = block.size
                    next_stage_idx = 0
                elif capturing:
                    capture_chunks.append(block)
                    captured_samples += block.size
                    dur_sec = captured_samples / sample_rate

                    # Verifica se o sustain terminou precocemente após 0,30 s
                    sustain_ended = dur_sec >= 0.30 and rms_db < sustain_end_thresh
                    reached_final = dur_sec >= stage_secs[-1]

                    if sustain_ended or reached_final:
                        audio = np.concatenate(capture_chunks)
                        res = pipeline.analyze_note(
                            Note(audio=audio, sr=sample_rate), provisional=False
                        )
                        await websocket.send_json(
                            {"type": "result", "data": _serialize_result(res)}
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
                        next_stage_idx = 0
                        cooldown_samples = rearm_samples
                    elif (
                        next_stage_idx < len(stage_secs) - 1
                        and dur_sec >= stage_secs[next_stage_idx]
                    ):
                        # Emite estimativa progressiva (confiança vai subindo conforme sustenta)
                        audio = np.concatenate(capture_chunks)
                        res = pipeline.analyze_note(
                            Note(audio=audio, sr=sample_rate), provisional=True
                        )
                        await websocket.send_json(
                            {"type": "result", "data": _serialize_result(res)}
                        )
                        next_stage_idx += 1

                prev_rms = 0.7 * prev_rms + 0.3 * rms

        except WebSocketDisconnect:
            return

    return app


app = create_app()


def run_server() -> None:
    """Ponto de entrada CLI para rodar o servidor web."""
    uvicorn.run(
        "guitar_string_identifier.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
