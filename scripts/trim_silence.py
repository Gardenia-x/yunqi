"""
处理 signal.wav：切除前端弱信号段（VAD检测），降低峰均比 Cm
输出 signal_trimmed.wav 用于 MATLAB 仿真
"""
import wave
import numpy as np
import struct
import os

def read_wav(path):
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        nf = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = wf.readframes(nf)
        if sw == 2:
            dtype = np.int16
        elif sw == 4:
            dtype = np.int32
        else:
            dtype = np.uint8
        arr = np.frombuffer(frames, dtype=dtype).astype(np.float64)
        if ch > 1:
            arr = arr.reshape(-1, ch)[:, 0]
        return arr, sr

def write_wav(path, arr, sr):
    arr_int = np.clip(arr, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr_int.tobytes())

def detect_onset(arr, sr, energy_thresh_ratio=0.50, min_silence_duration=0.3):
    """基于短时能量检测语音起始点（默认50%阈值跳过弱信号前段）"""
    frame_len = int(0.05 * sr)       # 50ms 帧长
    hop_len = int(0.01 * sr)          # 10ms 帧移
    energy = []
    for start in range(0, len(arr) - frame_len + 1, hop_len):
        seg = arr[start:start + frame_len]
        rms = np.sqrt(np.mean(seg ** 2))
        energy.append(rms)
    energy = np.array(energy)

    # 用全局最大能量的一定百分比作为阈值
    threshold = energy.max() * energy_thresh_ratio

    # 找到第一个超过阈值的帧
    onset_frames = np.where(energy > threshold)[0]
    if len(onset_frames) == 0:
        # 降级：用平均能量阈值
        threshold = np.mean(energy)
        onset_frames = np.where(energy > threshold)[0]
        if len(onset_frames) == 0:
            return 0

    onset_idx = onset_frames[0]

    # 从 onset 前一帧开始（保留一点过渡）
    onset_sample = max(0, (onset_idx - 1) * hop_len)
    return onset_sample

def analyze_energy(arr, sr, label=""):
    """分段分析能量"""
    dur = len(arr) / sr
    print(f"\n  {label} ({dur:.1f}s):")
    for sec in range(int(dur) + 1):
        start = int(sec * sr)
        end = int((sec + 1) * sr)
        seg = arr[start:end]
        if len(seg) == 0:
            break
        rms = np.sqrt(np.mean(seg ** 2))
        peak = np.max(np.abs(seg))
        print(f"    第{sec}秒: RMS={rms:.1f} peak={peak:.1f}")
    # 整体统计
    rms_total = np.sqrt(np.mean(arr ** 2))
    peak_total = np.max(np.abs(arr))
    cm = peak_total / (rms_total + 1e-10)
    print(f"    整体 RMS={rms_total:.1f} peak={peak_total:.1f} Cm={cm:.4f}")
    return cm

# ===== 主程序 =====
input_path = r'C:\Users\LENOVO\Desktop\signal.wav'
output_path = r'C:\Users\LENOVO\Desktop\signal_trimmed.wav'

# 读取
print("=" * 60)
print("读取音频文件...")
arr, sr = read_wav(input_path)
print(f"  采样率: {sr} Hz, 长度: {len(arr)/sr:.2f}s")

# 分析原始能量
cm_orig = analyze_energy(arr, sr, "原始信号")

# VAD检测起始点
onset = detect_onset(arr, sr, energy_thresh_ratio=0.50)
print(f"\nVAD检测: 语音起始点 = {onset/sr:.3f}s (第{onset}采样点)")

if onset > int(0.2 * sr):
    # 切除前段弱信号
    arr_trimmed = arr[onset:]
    print(f"切除前 {onset/sr:.3f}s 弱信号，新长度: {len(arr_trimmed)/sr:.2f}s")
else:
    arr_trimmed = arr
    print("弱信号段很短，无需切除")

# 分析处理后能量
cm_new = analyze_energy(arr_trimmed, sr, "处理后信号")

print(f"\n{'='*60}")
print(f"Cm 变化: {cm_orig:.4f} → {cm_new:.4f} (降低 {(1-cm_new/cm_orig)*100:.1f}%)")

# 保存
if len(arr_trimmed) > 0:
    write_wav(output_path, arr_trimmed, sr)
    print(f"已保存: {output_path}")
else:
    print("错误: 处理后的信号为空!")

print("=" * 60)
