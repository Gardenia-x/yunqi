"""
MP3 → MIDI 转换 Web 服务
上传 MP3 音频，自动生成 MIDI 文件 + 乐谱/波形/频谱可视化
基于 SimpleMIDIGenerator (v1.0 原始方法)
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

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pretty_midi
from simple_midi_generator import SimpleMIDIGenerator

# 服务模块
from services.visualization import generate_midi_preview
from services.sheet_music import generate_sheet, sheet_music_available

app = FastAPI(title="MP3 → MIDI 转换器", version="2.0")

# 配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

for d in [UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# 静态文件和模板
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 最大上传 50MB
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

# 版本配置
VERSIONS = {
    "accompanied": {
        "name": "有伴奏 / 合唱",
        "description": "适合有伴奏、多乐器、多人合唱的音频",
        "config": {
            "detection_method": "basic_pitch",
            "confidence_threshold": 0.3,
            "extract_melody": True,  # 从复音转录中提取主旋律线
        },
    },
    "solo": {
        "name": "独奏 / 清唱",
        "description": "适合单人演唱、独奏乐器等干净音频 — PYIN 单音检测",
        "config": {
            "detection_method": "pyin",
            "sr": 44100,
            "hop_length": 512,
            "min_freq": 65,
            "max_freq": 2000,
            "min_duration": 0.06,
            "max_gap": 0.06,
            "semitone_tolerance": 1,
            "pitch_smooth_kernel": 7,
            "harmonic_margin": 4,
            "confidence_threshold": 0.3,
            "voicing_threshold": 0.5,
        },
    },
}

# 任务存储
tasks: dict = {}


def convert_mp3_to_midi(audio_path: Path, version: str = "accompanied") -> tuple:
    """
    将 MP3 文件转换为 MIDI

    Returns:
        (midi, note_count, elapsed_time)
    """
    if version not in VERSIONS:
        raise ValueError(f"未知版本: {version}")
    version_info = VERSIONS[version]
    config = version_info["config"].copy()

    start = time.time()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    extract_melody = config.pop("extract_melody", False)

    generator = SimpleMIDIGenerator(config)
    midi = generator.process_audio(audio_bytes)

    # 复音转录后提取主旋律线（仅保留每个时刻最高音）
    if extract_melody and midi.instruments:
        midi = _extract_melody_line(midi)

    elapsed = time.time() - start
    note_count = len(midi.instruments[0].notes) if midi.instruments else 0

    return midi, note_count, elapsed


def _extract_melody_line(midi: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """
    从复音 MIDI 中提取主旋律线：每个时刻只保留最高音高的音符。
    这是针对 Basic Pitch 输出全复音结果的后处理。
    """
    notes = sorted(midi.instruments[0].notes, key=lambda n: (n.start, -n.pitch))

    if not notes:
        return midi

    melody_notes = []
    i = 0
    while i < len(notes):
        current = notes[i]
        # 找出与当前音符重叠的所有音符
        group = [current]
        j = i + 1
        while j < len(notes) and notes[j].start < current.end:
            group.append(notes[j])
            j += 1

        # 只保留音高最高的音符
        best = max(group, key=lambda n: (n.pitch, n.velocity))
        melody_notes.append(pretty_midi.Note(
            velocity=best.velocity,
            pitch=best.pitch,
            start=best.start,
            end=best.end,
        ))

        # 跳到当前音符结束后
        i = j

    result = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes = melody_notes
    result.instruments.append(instrument)
    return result


def generate_artifacts(
    task_id: str, midi: pretty_midi.PrettyMIDI, audio_bytes: bytes, version_config: dict
) -> dict:
    """
    生成可视化制品（波形、频谱、乐谱、音频预览）
    失败不阻塞主流程，返回可用制品列表
    """
    artifacts = {}

    # 乐谱（需要 MuseScore）
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

    # MIDI 音频预览
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
    """主页：文件上传界面"""
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
async def api_convert(file: UploadFile = File(...), version: str = Form("v1.0")):
    """上传 MP3 并转换为 MIDI，同时生成可视化制品"""
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith(
        (".mp3", ".wav", ".m4a", ".flac", ".ogg")
    ):
        return JSONResponse(
            {"error": "仅支持 MP3, WAV, M4A, FLAC, OGG 格式"},
            status_code=400,
        )

    # 验证版本
    if version not in VERSIONS:
        return JSONResponse({"error": f"未知版本: {version}"}, status_code=400)

    # 保存上传文件
    file_id = uuid.uuid4().hex[:16]
    safe_name = f"{file_id}_{file.filename}"
    upload_path = UPLOAD_DIR / safe_name

    try:
        # 读取并保存
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            return JSONResponse(
                {"error": f"文件过大，最大支持 {MAX_UPLOAD_SIZE // 1024 // 1024}MB"},
                status_code=400,
            )

        with open(upload_path, "wb") as f:
            f.write(content)

        # 转换为 MIDI
        midi, note_count, elapsed = convert_mp3_to_midi(upload_path, version)

        # 保存 MIDI 文件
        task_id = uuid.uuid4().hex[:12]
        output_path = OUTPUT_DIR / f"{task_id}.mid"
        midi.write(str(output_path))

        # 生成可视化制品
        version_config = VERSIONS[version]["config"]
        artifacts = generate_artifacts(task_id, midi, content, version_config)

        # 记录任务
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
            # 可视化制品 URL
            "has_sheet": artifacts.get("has_sheet", False),
            "sheet_url": f"/api/sheet/{task_id}" if artifacts.get("has_sheet") else None,
            "has_preview": artifacts.get("has_preview", False),
            "preview_url": f"/api/preview/{task_id}" if artifacts.get("has_preview") else None,
        }

        return result

    except Exception as e:
        return JSONResponse({"error": f"转换失败: {str(e)}"}, status_code=500)

    finally:
        # 清理上传文件
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass


@app.get("/api/download/{task_id}")
async def api_download(task_id: str):
    """下载生成的 MIDI 文件"""
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
    """获取乐谱图片 PNG"""
    return _serve_artifact(task_id, "sheet", "sheet.png", "image/png")


@app.get("/api/preview/{task_id}")
async def api_preview(task_id: str):
    """获取 MIDI 音频预览 WAV"""
    return _serve_artifact(task_id, "preview", "preview.wav", "audio/wav")


def _serve_artifact(task_id: str, key: str, filename: str, media_type: str):
    """通用制品文件响应"""
    if task_id not in tasks:
        return JSONResponse({"error": "任务不存在或已过期"}, status_code=404)

    artifacts = tasks[task_id].get("artifacts", {})
    file_path = artifacts.get(key)

    if not file_path or not Path(file_path).exists():
        return JSONResponse({"error": "文件不存在或已清理"}, status_code=404)

    return FileResponse(path=file_path, filename=filename, media_type=media_type)


@app.get("/api/versions")
async def api_versions():
    """获取可用版本列表"""
    return {
        version: {
            "name": info["name"],
            "description": info["description"],
        }
        for version, info in VERSIONS.items()
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "musescore": sheet_music_available(),
    }


# 定时清理旧文件
@app.middleware("http")
async def cleanup_old_files(request: Request, call_next):
    """清理超过 1 小时的旧文件"""
    now = time.time()
    max_age = 3600

    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in directory.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > max_age:
                try:
                    f.unlink()
                except Exception:
                    pass

    # 清理过期任务
    expired = [
        tid
        for tid, t in tasks.items()
        if not Path(t["output"]).exists()
    ]
    for tid in expired:
        del tasks[tid]

    return await call_next(request)
