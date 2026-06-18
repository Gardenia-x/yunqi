#!/bin/bash
# ============================================
# 韵启 — 更新到 Basic Pitch 引擎
# 在 Lighthouse 服务器上执行此脚本
# ============================================
set -e

echo "======================================"
echo "  韵启 — 升级到 Basic Pitch 引擎"
echo "======================================"

# ---------- 1. 安装 Basic Pitch ----------
echo "[1/4] 安装 basic-pitch + onnxruntime..."
cd /home/ubuntu/yunqi
source venv/bin/activate

# 仅安装 ONNX 后端（不装 TensorFlow，节省内存）
pip install onnxruntime resampy -i https://mirrors.cloud.tencent.com/pypi/simple
pip install basic-pitch --no-deps -i https://mirrors.cloud.tencent.com/pypi/simple

# ---------- 2. 同步代码 ----------
echo "[2/4] 同步代码..."

# 更新 simple_midi_generator.py
cat > /home/ubuntu/yunqi/src/simple_midi_generator.py << 'PYEOF'
#!/usr/bin/env python3
"""
简单MIDI生成器 — 支持 Basic Pitch (Spotify) 和 PYIN 两种引擎
"""
import sys
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

import scipy.signal
import librosa
import pretty_midi
import soundfile as sf


class SimpleMIDIGenerator:
    """MIDI生成器，支持 Basic Pitch (Spotify) 和 PYIN 两种方法"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'sr': 44100,
            'hop_length': 512,
            'min_freq': 65,
            'max_freq': 2000,
            'n_thresholds': 200,
            'resolution': 0.1,
            'harmonic_margin': 4,
            'min_duration': 0.06,
            'max_gap': 0.06,
            'semitone_tolerance': 1,
            'pitch_smooth_kernel': 7,
            'confidence_threshold': 0.3,
            'voicing_threshold': 0.5,
            'preemphasis_coef': 0.97,
            'normalize_axis': 0,
            'detection_method': 'basic_pitch',
        }

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        method = self.config.get('detection_method', 'basic_pitch')
        if method == 'basic_pitch':
            return self._process_with_basic_pitch(audio_bytes)
        else:
            return self._process_with_pyin(audio_bytes)

    def _process_with_basic_pitch(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            model_path = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
            _, midi_data, _ = predict(tmp_path, model_path)

            if not midi_data.instruments or not midi_data.instruments[0].notes:
                raise ValueError("Basic Pitch 未生成有效音符")

            notes = midi_data.instruments[0].notes
            conf = self.config.get('confidence_threshold', 0.3)
            notes = [n for n in notes if n.velocity >= conf * 100]
            midi_data.instruments[0].notes = notes

            return midi_data

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _process_with_pyin(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            y, sr = librosa.load(tmp_path, sr=self.config['sr'], mono=True)
            y = librosa.effects.preemphasis(y, coef=self.config['preemphasis_coef'])
            y = librosa.util.normalize(y, axis=self.config['normalize_axis'])

            hop = self.config['hop_length']
            target_length = ((len(y) // hop) + 1) * hop
            try:
                y = librosa.util.fix_length(y, size=target_length, mode='wrap')
            except TypeError:
                y = librosa.util.fix_length(y, target_length)
                if len(y) < target_length:
                    y = np.tile(y, np.ceil(target_length / len(y)).astype(int))[:target_length]
                else:
                    y = y[:target_length]

            if self.config.get('detection_method') == 'melody':
                f0, voiced_flag, amplitude = self._melody_tracking_detection(y, sr)
            else:
                f0, voiced_flag, amplitude = self._pitch_detection(y, sr)

            midi = self._create_midi(f0, voiced_flag, amplitude, sr)
            if not midi.instruments[0].notes:
                raise ValueError("生成的MIDI文件无有效音符")

            return midi

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _pitch_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hop = self.config['hop_length']
        fmin = self.config['min_freq']
        fmax = self.config['max_freq']

        frame_length = max(2048, int(sr / fmin * 2))
        frame_length = 2 ** int(np.ceil(np.log2(frame_length)))

        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            hop_length=hop, frame_length=frame_length,
            n_thresholds=self.config['n_thresholds'],
            resolution=self.config['resolution'],
            fill_na=np.nan,
        )

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        amplitude = np.clip((rms_db + 60) / 60, 0.05, 1.0)

        try:
            harmonic = librosa.effects.harmonic(y, margin=self.config['harmonic_margin'])
            f0_h, vo_flag_h, _ = librosa.pyin(
                harmonic, fmin=fmin, fmax=fmax, sr=sr,
                hop_length=hop, frame_length=frame_length,
                fill_na=np.nan,
            )
        except Exception:
            f0_h = np.full_like(f0, np.nan)
            vo_flag_h = np.zeros_like(voiced_flag)

        conf_thresh = self.config['confidence_threshold']
        use_harmonic = (voiced_prob < conf_thresh) & vo_flag_h
        fused_f0 = np.where(use_harmonic, f0_h, f0)
        has_pitch = ~np.isnan(fused_f0)

        if np.any(has_pitch):
            x = np.arange(len(fused_f0))
            fused_f0 = np.interp(x, x[has_pitch], fused_f0[has_pitch])
            kernel = self.config['pitch_smooth_kernel']
            fused_f0 = scipy.signal.medfilt(fused_f0, kernel_size=kernel)

        energy_voiced = rms > np.median(rms) * 0.15
        final_voiced = has_pitch & energy_voiced & (fused_f0 >= fmin) & (fused_f0 <= fmax)

        return fused_f0, final_voiced, amplitude

    def _melody_tracking_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hop = self.config['hop_length']
        fmin = self.config['min_freq']
        fmax = self.config['max_freq']

        frame_length = max(2048, int(sr / fmin * 2))
        frame_length = 2 ** int(np.ceil(np.log2(frame_length)))

        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, fmin=fmin, fmax=fmax,
            hop_length=hop, n_fft=frame_length,
        )

        n_candidates, n_frames = pitches.shape
        n_top = min(5, n_candidates)

        top_pitches = np.zeros((n_top, n_frames))
        top_mags = np.zeros((n_top, n_frames))
        for i in range(n_frames):
            col_mag = magnitudes[:, i]
            order = np.argsort(col_mag)[::-1][:n_top]
            top_pitches[:, i] = pitches[order, i]
            top_mags[:, i] = col_mag[order]

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        amplitude = np.clip((rms_db + 60) / 60, 0.05, 1.0)

        best_f0 = np.full(n_frames, np.nan)
        start_frame = int(np.argmax(rms))
        if top_mags[0, start_frame] > 0:
            best_f0[start_frame] = top_pitches[0, start_frame]

        prev_pitch = best_f0[start_frame]
        for i in range(start_frame - 1, -1, -1):
            if top_mags[0, i] <= 0:
                continue
            best_score = np.inf
            best_p = np.nan
            for k in range(n_top):
                p = top_pitches[k, i]
                m = top_mags[k, i]
                if p <= 0:
                    continue
                cents_change = abs(1200 * np.log2(max(p, prev_pitch) / min(p, prev_pitch))) if prev_pitch > 0 else 0
                score = cents_change - 50 * (m / (top_mags[0, i] + 1e-9))
                if score < best_score:
                    best_score = score
                    best_p = p
            if best_score < 200:
                best_f0[i] = best_p
                prev_pitch = best_p
            else:
                best_f0[i] = np.nan

        prev_pitch = best_f0[start_frame]
        for i in range(start_frame + 1, n_frames):
            if top_mags[0, i] <= 0:
                continue
            best_score = np.inf
            best_p = np.nan
            for k in range(n_top):
                p = top_pitches[k, i]
                m = top_mags[k, i]
                if p <= 0:
                    continue
                cents_change = abs(1200 * np.log2(max(p, prev_pitch) / min(p, prev_pitch))) if prev_pitch > 0 else 0
                score = cents_change - 50 * (m / (top_mags[0, i] + 1e-9))
                if score < best_score:
                    best_score = score
                    best_p = p
            if best_score < 200:
                best_f0[i] = best_p
                prev_pitch = best_p
            else:
                best_f0[i] = np.nan

        valid = ~np.isnan(best_f0)
        if np.sum(valid) > 0:
            x = np.arange(n_frames)
            best_f0 = np.interp(x, x[valid], best_f0[valid])
            kernel = self.config['pitch_smooth_kernel']
            best_f0 = scipy.signal.medfilt(best_f0, kernel_size=kernel)

        energy_voiced = rms > np.median(rms) * 0.15
        has_pitch = ~np.isnan(best_f0)
        final_voiced = has_pitch & energy_voiced & (best_f0 >= fmin) & (best_f0 <= fmax)

        return best_f0, final_voiced, amplitude

    def _create_midi(
        self, f0: np.ndarray, voiced_flag: np.ndarray,
        amplitude: np.ndarray, sr: int
    ) -> pretty_midi.PrettyMIDI:
        hop = self.config['hop_length']
        min_dur = self.config['min_duration']
        max_gap = self.config['max_gap']
        tolerance = self.config['semitone_tolerance']

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_note = None
        pending_notes = []

        def finalize_note(note_dict):
            dur = note_dict['end'] - note_dict['start']
            if dur >= min_dur:
                avg_amp = np.mean(note_dict['amplitudes'])
                velocity = int(np.clip(avg_amp * 100, 30, 120))
                pending_notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_dict['pitch'],
                    start=note_dict['start'],
                    end=note_dict['end'],
                ))

        for i in range(len(f0)):
            time = i * hop / sr
            is_voiced = voiced_flag[i]
            pitch_hz = f0[i]

            if is_voiced and not np.isnan(pitch_hz) and pitch_hz > 0:
                note_num = int(round(librosa.hz_to_midi(pitch_hz)))
                amp = amplitude[i]

                if current_note is None:
                    current_note = {
                        'pitch': note_num, 'start': time,
                        'end': time, 'amplitudes': [amp],
                    }
                elif abs(note_num - current_note['pitch']) <= tolerance:
                    current_note['end'] = time
                    current_note['amplitudes'].append(amp)
                else:
                    finalize_note(current_note)
                    current_note = {
                        'pitch': note_num,
                        'start': max(current_note['end'], time - max_gap),
                        'end': time, 'amplitudes': [amp],
                    }
            else:
                if current_note is not None:
                    gap = time - current_note['end']
                    if gap > max_gap * 2:
                        finalize_note(current_note)
                        current_note = None
                    else:
                        current_note['end'] = time

        if current_note is not None:
            finalize_note(current_note)

        if pending_notes:
            pending_notes.sort(key=lambda n: n.start)
            merged = [pending_notes[0]]
            for note in pending_notes[1:]:
                prev = merged[-1]
                if note.pitch == prev.pitch and (note.start - prev.end) < max_gap * 0.5:
                    prev.end = max(prev.end, note.end)
                else:
                    merged.append(note)
            instrument.notes = merged

        if not instrument.notes:
            raise ValueError("无有效音符生成，请检查输入音频")

        midi.instruments.append(instrument)
        return midi

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def update_config(self, config_updates: Dict[str, Any]):
        self.config.update(config_updates)
PYEOF

# 更新 app.py
cat > /home/ubuntu/yunqi/web_app/app.py << 'PYEOF'
"""
MP3 → MIDI 转换 Web 服务
基于 Spotify Basic Pitch 复音转录引擎
"""
import os
import sys
import uuid
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pretty_midi
from simple_midi_generator import SimpleMIDIGenerator

from services.visualization import generate_midi_preview
from services.sheet_music import generate_sheet, sheet_music_available

app = FastAPI(title="MP3 → MIDI 转换器", version="3.0")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MAX_UPLOAD_SIZE = 50 * 1024 * 1024

VERSIONS = {
    "accompanied": {
        "name": "有伴奏 / 合唱",
        "description": "适合有伴奏、多乐器、多人合唱的音频 — Spotify Basic Pitch 引擎",
        "config": {
            "detection_method": "basic_pitch",
            "confidence_threshold": 0.3,
        },
    },
    "solo": {
        "name": "独奏 / 清唱",
        "description": "适合单人演唱、独奏乐器等干净音频 — Spotify Basic Pitch 引擎",
        "config": {
            "detection_method": "basic_pitch",
            "confidence_threshold": 0.4,
        },
    },
}

tasks: dict = {}


def convert_mp3_to_midi(audio_path: Path, version: str = "accompanied") -> tuple:
    if version not in VERSIONS:
        raise ValueError(f"未知版本: {version}")
    version_info = VERSIONS[version]
    config = version_info["config"]

    start = time.time()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    generator = SimpleMIDIGenerator(config)
    midi = generator.process_audio(audio_bytes)

    elapsed = time.time() - start
    note_count = len(midi.instruments[0].notes) if midi.instruments else 0

    return midi, note_count, elapsed


def generate_artifacts(
    task_id: str, midi: pretty_midi.PrettyMIDI, audio_bytes: bytes, version_config: dict
) -> dict:
    artifacts = {}

    try:
        if sheet_music_available():
            sheet_path = OUTPUT_DIR / f"{task_id}_sheet.png"
            sheet_path.write_bytes(generate_sheet(midi, "png"))
            artifacts["sheet"] = str(sheet_path)
            artifacts["has_sheet"] = True
        else:
            artifacts["has_sheet"] = False
    except Exception:
        artifacts["has_sheet"] = False

    try:
        preview_path = OUTPUT_DIR / f"{task_id}_preview.wav"
        preview_data = generate_midi_preview(midi)
        preview_path.write_bytes(preview_data)
        artifacts["preview"] = str(preview_path)
        artifacts["has_preview"] = True
    except Exception:
        artifacts["has_preview"] = False

    return artifacts


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    version_list = [
        {"key": k, "name": v["name"], "desc": v["description"]}
        for k, v in VERSIONS.items()
    ]
    has_musescore = sheet_music_available()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "version_list": version_list, "has_musescore": has_musescore},
    )


@app.post("/api/convert")
async def api_convert(file: UploadFile = File(...), version: str = Form("accompanied")):
    if not file.filename or not file.filename.lower().endswith(
        (".mp3", ".wav", ".m4a", ".flac", ".ogg")
    ):
        return JSONResponse(
            {"error": "仅支持 MP3, WAV, M4A, FLAC, OGG 格式"},
            status_code=400,
        )

    if version not in VERSIONS:
        return JSONResponse({"error": f"未知版本: {version}"}, status_code=400)

    file_id = uuid.uuid4().hex[:16]
    safe_name = f"{file_id}_{file.filename}"
    upload_path = UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                {"error": f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB"},
                status_code=400,
            )

        with open(upload_path, "wb") as f:
            f.write(content)

        midi, note_count, elapsed = convert_mp3_to_midi(upload_path, version)

        task_id = uuid.uuid4().hex[:12]
        output_path = OUTPUT_DIR / f"{task_id}.mid"
        midi.write(str(output_path))

        version_config = VERSIONS[version]["config"]
        artifacts = generate_artifacts(task_id, midi, content, version_config)

        tasks[task_id] = {
            "filename": file.filename,
            "version": version,
            "note_count": note_count,
            "elapsed": round(elapsed, 1),
            "output": str(output_path),
            "artifacts": artifacts,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        result = {
            "success": True,
            "task_id": task_id,
            "filename": file.filename,
            "version": version,
            "note_count": note_count,
            "elapsed_seconds": round(elapsed, 1),
            "download_url": f"/api/download/{task_id}",
            "has_sheet": artifacts.get("has_sheet", False),
            "sheet_url": f"/api/sheet/{task_id}" if artifacts.get("has_sheet") else None,
            "has_preview": artifacts.get("has_preview", False),
            "preview_url": f"/api/preview/{task_id}" if artifacts.get("has_preview") else None,
        }

        return result

    except Exception as e:
        return JSONResponse({"error": f"转换失败: {str(e)}"}, status_code=500)

    finally:
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass


@app.get("/api/download/{task_id}")
async def api_download(task_id: str):
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在或已过期"}, status_code=404)

    task = tasks[task_id]
    output_path = Path(task["output"])

    if not output_path.exists():
        return JSONResponse({"error": "文件已被清理"}, status_code=404)

    original_name = Path(task["filename"]).stem
    download_name = f"{original_name}_{task['version']}.mid"

    return FileResponse(
        path=str(output_path),
        filename=download_name,
        media_type="audio/midi",
        headers={"X-Note-Count": str(task["note_count"])},
    )


@app.get("/api/sheet/{task_id}")
async def api_sheet(task_id: str):
    return _serve_artifact(task_id, "sheet", "sheet.png", "image/png")


@app.get("/api/preview/{task_id}")
async def api_preview(task_id: str):
    return _serve_artifact(task_id, "preview", "preview.wav", "audio/wav")


def _serve_artifact(task_id: str, key: str, filename: str, media_type: str):
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在或已过期"}, status_code=404)

    artifacts = tasks[task_id].get("artifacts", {})
    file_path = artifacts.get(key)

    if not file_path or not Path(file_path).exists():
        return JSONResponse({"error": "文件不存在或已清理"}, status_code=404)

    return FileResponse(path=file_path, filename=filename, media_type=media_type)


@app.get("/api/versions")
async def api_versions():
    return {
        version: {
            "name": info["name"],
            "description": info["description"],
        }
        for version, info in VERSIONS.items()
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "musescore": sheet_music_available(),
    }


@app.middleware("http")
async def cleanup_old_files(request: Request, call_next):
    now = time.time()
    max_age = 3600

    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in directory.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > max_age:
                try:
                    f.unlink()
                except Exception:
                    pass

    expired = [
        tid
        for tid, t in tasks.items()
        if not Path(t["output"]).exists()
    ]
    for tid in expired:
        del tasks[tid]

    return await call_next(request)
PYEOF

# 更新 index.html
cat > /home/ubuntu/yunqi/web_app/templates/index.html << 'HTMLEOF'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>韵启 — MP3 → MIDI 转换器</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1><i class="fas fa-music"></i> 韵启</h1>
            <p class="subtitle">上传音频文件，AI 自动识别旋律并生成 MIDI</p>
        </header>

        <main class="main-content">
            <div class="upload-section card">
                <h2><i class="fas fa-upload"></i> 上传音频</h2>
                <div class="upload-form">
                    <div class="file-drop-zone" id="dropZone">
                        <input type="file" id="fileInput" accept=".mp3,.wav,.m4a,.flac,.ogg" hidden>
                        <i class="fas fa-cloud-upload-alt upload-icon"></i>
                        <p class="drop-text">拖拽音频文件到此处，或点击选择</p>
                        <p class="drop-hint">支持 MP3 / WAV / M4A / FLAC / OGG，最大 50MB</p>
                        <div class="file-info" id="fileInfo" style="display:none;"></div>
                    </div>

                    <div class="version-select">
                        <label>算法版本：</label>
                        <select id="versionSelect">
                            {% for v in version_list %}
                            <option value="{{ v.key }}" {% if v.key == 'accompanied' %}selected{% endif %}>
                                {{ v.name }} - {{ v.desc }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <button id="convertBtn" class="btn-primary" disabled>
                        <i class="fas fa-play"></i> 开始转换
                    </button>
                </div>
            </div>

            <div class="progress-section card" id="progressSection" style="display:none;">
                <h3><i class="fas fa-spinner fa-spin"></i> 转换中...</h3>
                <p id="progressText">正在分析音频...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p class="progress-hint">处理可能需要 30-120 秒，请耐心等待</p>
            </div>

            <div class="result-section card" id="resultSection" style="display:none;">
                <h2><i class="fas fa-check-circle"></i> 转换成功!</h2>
                <div class="result-info">
                    <div class="info-row">
                        <span class="info-label">文件名</span>
                        <span class="info-value" id="resultFilename">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">算法版本</span>
                        <span class="info-value" id="resultVersion">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">识别音符数</span>
                        <span class="info-value" id="resultNotes">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">处理耗时</span>
                        <span class="info-value" id="resultTime">-</span>
                    </div>
                </div>
                <div class="result-actions">
                    <a id="downloadLink" class="btn-download" href="#">
                        <i class="fas fa-download"></i> 下载 MIDI
                    </a>
                    <button class="btn-secondary" onclick="resetForm()">
                        <i class="fas fa-redo"></i> 再转换一个
                    </button>
                </div>

                <div class="audio-preview" id="audioPreviewBox" style="display:none;">
                    <h3><i class="fas fa-volume-up"></i> MIDI 音频预览</h3>
                    <audio id="audioPlayer" controls></audio>
                </div>
            </div>

            <div class="visualizations" id="visualizations" style="display:none;">
                <div class="viz-card card" id="sheetCard" style="display:none;">
                    <h3><i class="fas fa-music"></i> 五线谱</h3>
                    <img id="sheetImg" src="" alt="乐谱" class="viz-img">
                    <p class="viz-hint">乐谱由 MuseScore 自动生成，仅供参考</p>
                </div>
            </div>

            <div class="error-section card" id="errorSection" style="display:none;">
                <h3><i class="fas fa-exclamation-triangle"></i> 转换失败</h3>
                <p id="errorMessage"></p>
            </div>
        </main>

        <footer class="footer">
            <p>采用 Spotify Basic Pitch 引擎，支持复音音频 | 有伴奏选「有伴奏/合唱」| 清唱选「独奏/清唱」</p>
        </footer>
    </div>

    <script src="/static/script.js"></script>
</body>
</html>
HTMLEOF

# ---------- 3. 安装 ONNX Runtime ----------
echo "[3/4] 安装 ONNX Runtime..."
# 确保只用 ONNX（不依赖 TensorFlow，节省内存）
python -c "from basic_pitch.inference import predict; print('Basic Pitch OK')" 2>&1 | tail -1

# ---------- 4. 重启服务 ----------
echo "[4/4] 重启服务..."
sudo supervisorctl restart midi-converter
sleep 2
sudo supervisorctl status midi-converter

echo ""
echo "======================================"
echo "  升级完成!"
echo "======================================"
echo ""
echo "  测试: curl http://localhost:8080/health"
echo ""
