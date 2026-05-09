"""
音频可视化服务
生成波形图和频谱图的 PNG 图片
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头渲染，不弹出窗口
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import librosa
import librosa.display

# 中文字体初始化
_zh_font = None


def _init_font():
    global _zh_font
    if _zh_font is not None:
        return _zh_font

    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                _zh_font = FontProperties(fname=path)
                return _zh_font
            except Exception:
                pass

    # Fallback: 使用系统 sans-serif
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def generate_waveform(audio_bytes: bytes, sr: int = 44100) -> bytes:
    """
    从音频数据生成波形图 PNG

    Args:
        audio_bytes: 原始音频文件内容
        sr: 采样率

    Returns:
        PNG 图片的字节内容
    """
    font = _init_font()

    # 写入临时文件（librosa 需要文件路径）
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    times = np.linspace(0, len(y) / sr, len(y))

    # 降采样以加速渲染
    step = max(1, len(y) // 4000)
    ax.plot(times[::step], y[::step], color="#7c5cfc", linewidth=0.6)
    ax.fill_between(times[::step], y[::step], alpha=0.15, color="#7c5cfc")
    ax.set_xlim(0, times[-1])
    ax.axhline(y=0, color="#333", linewidth=0.5)

    if font:
        ax.set_title("音频波形", fontproperties=font, color="#ccc")
        ax.set_xlabel("时间 (秒)", fontproperties=font, color="#888")
        ax.set_ylabel("振幅", fontproperties=font, color="#888")
    else:
        ax.set_title("Waveform", color="#ccc")
        ax.set_xlabel("Time (s)", color="#888")
        ax.set_ylabel("Amplitude", color="#888")

    ax.set_facecolor("#0f0c29")
    fig.patch.set_facecolor("#0f0c29")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_spectrogram(audio_bytes: bytes, sr: int = 44100, hop_length: int = 512) -> bytes:
    """
    从音频数据生成频谱图 PNG

    Args:
        audio_bytes: 原始音频文件内容
        sr: 采样率
        hop_length: 帧移长度

    Returns:
        PNG 图片的字节内容
    """
    font = _init_font()

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        y, _ = librosa.load(tmp_path, sr=sr, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        D, y_axis="log", x_axis="time", ax=ax, sr=sr, hop_length=hop_length, cmap="viridis"
    )

    if font:
        ax.set_title("音频频谱分析", fontproperties=font, color="#ccc")
        ax.set_xlabel("时间 (秒)", fontproperties=font, color="#888")
        ax.set_ylabel("频率 (Hz)", fontproperties=font, color="#888")
    else:
        ax.set_title("Spectrogram", color="#ccc")
        ax.set_xlabel("Time (s)", color="#888")
        ax.set_ylabel("Frequency (Hz)", color="#888")

    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.outline.set_edgecolor("#333")
    cbar.ax.tick_params(colors="#888")
    cbar.set_label("dB", color="#888")

    ax.set_facecolor("#0f0c29")
    fig.patch.set_facecolor("#0f0c29")
    ax.tick_params(colors="#888")

    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_midi_preview(midi: "pretty_midi.PrettyMIDI", sr: int = 44100) -> bytes:
    """
    将 MIDI 合成为 WAV 音频预览

    Args:
        midi: pretty_midi.PrettyMIDI 对象
        sr: 采样率

    Returns:
        WAV 音频的字节内容
    """
    import soundfile as sf

    synthesized = midi.synthesize(fs=sr, wave=np.sin)
    peak = np.max(np.abs(synthesized))
    if peak > 0:
        synthesized = 0.8 * synthesized / peak

    buf = BytesIO()
    sf.write(buf, synthesized, sr, format="WAV")
    buf.seek(0)
    return buf.getvalue()
