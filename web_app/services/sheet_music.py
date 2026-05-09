"""
乐谱生成服务
使用 music21 + MuseScore 将 MIDI 转为五线谱 PNG
"""
import os
import uuid
import shutil
import subprocess
from pathlib import Path
from io import BytesIO

# 关键：让 Qt/MuseScore 在无图形界面的服务器上离屏渲染
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pretty_midi
from music21 import environment, converter

_initialized = False
_mscore_path: str | None = None


def _init_music21():
    """初始化 music21 环境，自动查找 MuseScore"""
    global _initialized, _mscore_path

    if _initialized:
        return _mscore_path

    _initialized = True

    candidates = [
        shutil.which("musescore4"),
        shutil.which("musescore"),
        shutil.which("musescore3"),
        "/usr/bin/musescore",
        "/usr/bin/musescore3",
    ]
    for path in candidates:
        if path and Path(path).exists():
            _mscore_path = path
            break

    if _mscore_path:
        env = environment.Environment()
        env["musescoreDirectPNGPath"] = _mscore_path
        temp_dir = Path("/tmp/music21")
        temp_dir.mkdir(exist_ok=True)
        env["directoryScratch"] = str(temp_dir)

    return _mscore_path


def sheet_music_available() -> bool:
    """检查乐谱生成是否可用"""
    return _init_music21() is not None


def generate_sheet(midi: pretty_midi.PrettyMIDI, fmt: str = "png") -> bytes:
    """
    将 MIDI 转为五线谱图片
    使用 music21 内置的 MuseScore 转换（自动处理环境变量）
    """
    mscore = _init_music21()
    if not mscore:
        raise RuntimeError("MuseScore 未安装")

    if not midi.instruments or not midi.instruments[0].notes:
        raise ValueError("MIDI 不含有效音符")

    temp_dir = Path("/tmp/music21_sheet")
    temp_dir.mkdir(exist_ok=True)
    uid = uuid.uuid4().hex[:10]

    midi_path = temp_dir / f"in_{uid}.mid"
    out_stem = temp_dir / f"out_{uid}"

    try:
        midi.write(str(midi_path))

        if fmt == "musicxml":
            score = converter.parse(str(midi_path))
            xml_path = temp_dir / f"out_{uid}.musicxml"
            score.write("musicxml", fp=str(xml_path))
            return xml_path.read_bytes()

        # 用 music21 的 write() 生成 PNG——它内部调用 MuseScore，会自动继承 QT_QPA_PLATFORM
        score = converter.parse(str(midi_path))
        score.write("musicxml.png", fp=str(out_stem))

        # music21 会在 out_stem 后面自动加 .png
        actual_path = Path(str(out_stem) + ".png")
        if not actual_path.exists():
            # MuseScore 有时输出到不同位置，搜索一下
            candidates = list(temp_dir.glob(f"out*{uid}*.png"))
            if candidates:
                actual_path = candidates[0]
            else:
                raise FileNotFoundError("MuseScore 未生成 PNG 文件")

        return actual_path.read_bytes()

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
