"""Backend FastAPI (REST + WebSocket) para o Identificador de Corda e Casa."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from guitar_string_identifier.calibration.calibrator import (
    CalibrationSampleResult,
    GuitarCalibrator,
)
from guitar_string_identifier.core.music import midi_to_hz, string_fret_to_midi, string_fret_to_note
from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import Note, Result
from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.profiles.store import ProfileStore
from guitar_string_identifier.sources.dataset_source import (
    EGFxSetLabelProvider,
    list_egfxset_files,
)

STATIC_DIR = Path(__file__).parent / "static"


class RenameRequest(BaseModel):
    new_name: str


class DatasetAnalyzeRequest(BaseModel):
    rel_path: str
    profile: str | None = None


class CalibrationApplySample(BaseModel):
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
    profile: str | None = None
    samples: list[CalibrationApplySample]
    notes_label: str = "Calibração guiada via interface web."


class CalibrationPcmRequest(BaseModel):
    profile: str | None = None
    string: int
    fret: int
    sample_rate: int
    samples: list[float]


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
        version="0.1.0",
    )

    store = ProfileStore(profiles_dir)
    dataset_root = Path(dataset_dir)

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

    # --- Frontend estático ---
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
        )
        results = pipeline.analyze_audio(
            np.asarray(mono, dtype=np.float64),
            int(sr),
            source_path=file.filename,
        )
        if not results:
            raise HTTPException(status_code=422, detail="Nenhuma nota detectada no áudio.")

        res = results[0]
        # Se o nome do arquivo não segue padrão EGFxSet, garante label=None
        if file.filename:
            res.label = label_provider.get_label(Path(file.filename))
        return _serialize_result(res)

    # --- Calibração guiada ---
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
                            "expected_f0": round(midi_to_hz(midi), 2),
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

    # --- WebSocket de captura ao vivo (PCM float32 vindo do navegador) ---
    @app.websocket("/ws/live")
    async def live_audio_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        sample_rate = 48000
        profile = _get_profile(None)
        pipeline = GuitarIdentifierPipeline(profile, whole_file=True)

        # Buffer circular de captura de nota
        capturing = False
        capture_chunks: list[np.ndarray] = []
        captured_samples = 0
        target_samples = int(0.85 * sample_rate)
        prev_rms = 0.0
        cooldown_samples = 0

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
                        pipeline = GuitarIdentifierPipeline(profile, whole_file=True)
                        target_samples = int(
                            max(0.75, profile.analysis.window_sec + 0.15) * sample_rate
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
                        await websocket.send_json(
                            {
                                "type": "ready",
                                "sample_rate": sample_rate,
                                "profile": profile.name,
                            }
                        )
                    elif msg_type == "trigger" and capture_chunks:
                        # Permite disparo manual de análise no último áudio
                        audio = np.concatenate(capture_chunks)
                        res = pipeline.analyze_note(Note(audio=audio, sr=sample_rate))
                        await websocket.send_json(
                            {"type": "result", "data": _serialize_result(res)}
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
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
                clipping = peak >= 0.98

                await websocket.send_json(
                    {
                        "type": "level",
                        "rms_db": round(rms_db, 1),
                        "peak": round(peak, 4),
                        "clipping": clipping,
                    }
                )

                if cooldown_samples > 0:
                    cooldown_samples = max(0, cooldown_samples - block.size)

                # Detecção simples de ataque (salto de energia RMS acima de -36 dB)
                if (
                    not capturing
                    and cooldown_samples == 0
                    and rms_db > -36.0
                    and (rms > prev_rms * 2.2 or rms_db > -24.0)
                ):
                    capturing = True
                    capture_chunks = [block]
                    captured_samples = block.size
                elif capturing:
                    capture_chunks.append(block)
                    captured_samples += block.size
                    if captured_samples >= target_samples:
                        audio = np.concatenate(capture_chunks)
                        res = pipeline.analyze_note(Note(audio=audio, sr=sample_rate))
                        await websocket.send_json(
                            {"type": "result", "data": _serialize_result(res)}
                        )
                        capturing = False
                        capture_chunks.clear()
                        captured_samples = 0
                        cooldown_samples = int(0.35 * sample_rate)

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
