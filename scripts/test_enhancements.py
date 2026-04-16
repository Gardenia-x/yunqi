#!/usr/bin/env python3
"""
测试改进后的算法
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_feedback_enhanced import EnhancedParameterSet
import librosa

def test_onset_detection():
    print("测试改进的onset检测")
    print("=" * 50)

    # 加载测试音频
    audio_path = Path("test2.mp3")
    if not audio_path.exists():
        print("音频文件不存在")
        return

    y, sr = librosa.load(audio_path, sr=44100)

    # 使用最佳参数配置
    config = {
        'sr': sr,
        'hop_length': 256,
        'frame_length': 2048,
        'min_freq': 350,
        'max_freq': 1200,
        'pitch_method': 'hybrid',
        'voicing_threshold': 0.45,
        'confidence_weight': 0.65,
        'onset_threshold': 0.55,
        'min_note_duration': 0.09,
        'max_gap': 0.04
    }

    generator = EnhancedMIDIGenerator(config)

    # 测试onset检测
    onsets = generator._detect_onsets(y, sr)
    print(f"检测到onset数量: {len(onsets)}")
    if len(onsets) > 0:
        print(f"第一个onset: {onsets[0]:.3f}s")
        print(f"最后一个onset: {onsets[-1]:.3f}s")
        print(f"onset平均间隔: {np.mean(np.diff(onsets)):.3f}s")

    # 测试音高检测
    print("\n测试音高检测...")
    f0, voiced_flag = generator._enhanced_pitch_detection(y, sr)
    print(f"音高轮廓长度: {len(f0)}")
    print(f"浊音帧比例: {np.sum(voiced_flag)/len(voiced_flag):.1%}")

    # 测试音符分割（使用虚拟数据）
    print("\n测试音符分割...")
    velocity_profile = generator._estimate_velocity(y, sr)
    notes = generator._segment_notes(f0, voiced_flag, sr, onsets, velocity_profile)
    print(f"分割出音符数: {len(notes)}")
    if len(notes) > 0:
        durations = [n['end'] - n['start'] for n in notes]
        print(f"音符平均时长: {np.mean(durations):.3f}s")
        print(f"最短音符: {np.min(durations):.3f}s")
        print(f"最长音符: {np.max(durations):.3f}s")

    return len(onsets), len(notes)

if __name__ == "__main__":
    test_onset_detection()