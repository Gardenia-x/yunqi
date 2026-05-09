"""
乐谱生成服务
使用 music21 将 MIDI 转为五线谱 PNG / PDF
需要 MuseScore (或 LilyPond) 作为渲染后端
"""
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from io import BytesIO

import pretty_midi
import numpy as np

# music21 设置
from music21 import environment, converter

_initialized = False
_mscore_path: str | None = None


def _init_music21():
    """初始化 music21 环境，自动查找 MuseScore"""
    global _initialized, _mscore_path

    if _initialized:
        return _mscore_path

    _initialized = True

    # 查找 MuseScore
    candidates = [
        shutil.which("musescore4"),
        shutil.which("musescore"),
        shutil.which("musescore3"),
        "/usr/bin/musescore",
        "/usr/bin/musescore3",
        "/snap/bin/musescore",
        "/opt/musescore/bin/musescore",
    ]
    for path in candidates:
        if path and Path(path).exists():
            _mscore_path = path
            break

    if _mscore_path:
        env = environment.Environment()
        env["musescoreDirectPNGPath"] = _mscore_path
        # 设置临时目录
        temp_dir = Path("/tmp/music21")
        temp_dir.mkdir(exist_ok=True)
        env["directoryScratch"] = str(temp_dir)

    return _mscore_path


def sheet_music_available() -> bool:
    """检查乐谱生成是否可用"""
    path = _init_music21()
    return path is not None


def generate_sheet(midi: pretty_midi.PrettyMIDI, fmt: str = "png") -> bytes:
    """
    将 MIDI 转为五线谱图片

    Args:
        midi: pretty_midi.PrettyMIDI 对象
        fmt: 输出格式 (png, pdf, musicxml)

    Returns:
        文件二进制内容
    """
    mscore = _init_music21()
    if not mscore:
        raise RuntimeError("MuseScore 未安装，无法生成乐谱。请在服务器上安装 MuseScore。")

    if not midi.instruments or not midi.instruments[0].notes:
        raise ValueError("MIDI 不含有效音符，无法生成乐谱")

    temp_dir = Path("/tmp/music21_sheet")
    temp_dir.mkdir(exist_ok=True)
    uid = uuid.uuid4().hex[:10]

    midi_path = temp_dir / f"in_{uid}.mid"
    out_path = temp_dir / f"out_{uid}.{fmt}"

    try:
        midi.write(str(midi_path))

        if fmt == "musicxml":
            score = converter.parse(str(midi_path))
            xml_path = temp_dir / f"out_{uid}.musicxml"
            score.write("musicxml", fp=str(xml_path))
            return xml_path.read_bytes()

        # PNG/PDF 通过 MuseScore CLI
        cmd = [mscore, "--force", str(midi_path.resolve()), "--export-to", str(out_path.resolve())]
        if fmt == "png":
            cmd.extend(["-T", "1"])

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
            encoding="utf-8",
            errors="replace",
        )

        if not out_path.exists():
            # 尝试查找变体文件名
            candidates = list(temp_dir.glob(f"out*{uid}*.{fmt}"))
            if candidates:
                out_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"乐谱生成失败（MuseScore 退出码 {result.returncode}）。"
                    f"输出: {result.stdout[:300]}"
                )

        return out_path.read_bytes()

    finally:
        for f in temp_dir.glob(f"*{uid}*"):
            try:
                f.unlink()
            except Exception:
                pass


def midi_to_sheet_bytes(midi_bytes: bytes, fmt: str = "png") -> bytes:
    """从 MIDI 字节流生成乐谱（便捷接口）"""
    midi = pretty_midi.PrettyMIDI(BytesIO(midi_bytes))
    return generate_sheet(midi, fmt)
