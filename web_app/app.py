"""
MP3 → MIDI 转换 Web 服务
上传 MP3 音频，自动生成 MIDI 文件
基于 SimpleMIDIGenerator (v1.0 原始方法)
"""
import os
import sys
import uuid
import time
import shutil
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

app = FastAPI(title="MP3 → MIDI 转换器", version="1.0")

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
    "v1.0": {
        "name": "v1.0 原始方法",
        "description": "听觉体验最佳，参数保守",
        "config": {
            "sr": 44100,
            "hop_length": 512,
            "min_freq": 80,
            "max_freq": 1200,
            "min_duration": 0.1,
            "max_gap": 0.05,
            "voicing_threshold": 0.6,
            "harmonic_margin": 3,
        },
    },
    "v2.0": {
        "name": "v2.0 参数优化",
        "description": "提高时间分辨率，检测更多细节",
        "config": {
            "sr": 44100,
            "hop_length": 256,
            "min_freq": 80,
            "max_freq": 1200,
            "min_duration": 0.05,
            "max_gap": 0.03,
            "voicing_threshold": 0.4,
            "harmonic_margin": 5,
        },
    },
    "v3.0": {
        "name": "v3.0 短音符增强",
        "description": "增强谐波检测与短音符识别",
        "config": {
            "sr": 44100,
            "hop_length": 256,
            "min_freq": 80,
            "max_freq": 1200,
            "min_duration": 0.05,
            "max_gap": 0.03,
            "voicing_threshold": 0.4,
            "harmonic_margin": 6,
        },
    },
}

# 任务存储（简单内存存储，生产环境建议用 Redis/DB）
tasks: dict = {}


def convert_mp3_to_midi(mp3_path: Path, version: str = "v1.0") -> tuple:
    """
    将 MP3 文件转换为 MIDI

    Returns:
        (midi_bytes, note_count, elapsed_time)
    """
    version_info = VERSIONS.get(version, VERSIONS["v1.0"])
    config = version_info["config"]

    start = time.time()

    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()

    generator = SimpleMIDIGenerator(config)
    midi = generator.process_audio(audio_bytes)

    elapsed = time.time() - start
    note_count = len(midi.instruments[0].notes) if midi.instruments else 0

    # 写入临时文件
    task_id = uuid.uuid4().hex[:12]
    output_path = OUTPUT_DIR / f"{task_id}.mid"
    midi.write(str(output_path))

    return output_path, note_count, elapsed, task_id


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页：文件上传界面"""
    version_list = [
        {"key": k, "name": v["name"], "desc": v["description"]}
        for k, v in VERSIONS.items()
    ]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "version_list": version_list},
    )


@app.post("/api/convert")
async def api_convert(file: UploadFile = File(...), version: str = Form("v1.0")):
    """上传 MP3 并转换为 MIDI"""
    # 验证文件类型
    if not file.filename or not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
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
        output_path, note_count, elapsed, task_id = convert_mp3_to_midi(
            upload_path, version
        )

        # 记录任务
        tasks[task_id] = {
            "filename": file.filename,
            "version": version,
            "note_count": note_count,
            "elapsed": round(elapsed, 1),
            "output": str(output_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return {
            "success": True,
            "task_id": task_id,
            "filename": file.filename,
            "version": version,
            "note_count": note_count,
            "elapsed_seconds": round(elapsed, 1),
            "download_url": f"/api/download/{task_id}",
        }

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

    # 生成下载文件名
    original_name = Path(task["filename"]).stem
    download_name = f"{original_name}_{task['version']}.mid"

    return FileResponse(
        path=str(output_path),
        filename=download_name,
        media_type="audio/midi",
        headers={"X-Note-Count": str(task["note_count"])},
    )


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
    return {"status": "ok", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


# 定时清理旧文件（每次请求时检查）
@app.middleware("http")
async def cleanup_old_files(request: Request, call_next):
    """清理超过 1 小时的旧文件"""
    now = time.time()
    max_age = 3600  # 1 小时

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
