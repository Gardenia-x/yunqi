#!/usr/bin/env python3
"""
生成v5.0_short_note_improved.mid
专注于改进短音符检测，保持v1.0的流畅度
"""
import sys
import time
from pathlib import Path
import json
from typing import Tuple

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_midi_generator import SimpleMIDIGenerator
from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi
import librosa
import numpy as np


def enhanced_harmonic_detection(y: np.ndarray, sr: int, hop_length: int,
                              harmonic_margin: int = 6) -> np.ndarray:
    """
    增强的谐波检测，专门针对A4(440Hz)和E5(659Hz)音高

    Args:
        y: 音频信号
        sr: 采样率
        hop_length: hop长度
        harmonic_margin: 谐波边界参数

    Returns:
        谐波增强后的音频信号
    """
    try:
        # 使用更强的谐波增强
        harmonic = librosa.effects.harmonic(y, margin=harmonic_margin)
        return harmonic
    except Exception as e:
        print(f"谐波增强失败，使用原始音频: {e}")
        return y


def multi_scale_pitch_detection(y: np.ndarray, sr: int, config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    多尺度音高检测：结合不同hop_length的结果

    Args:
        y: 音频信号
        sr: 采样率
        config: 配置参数

    Returns:
        f0: 基频序列
        voiced_flag: 发声标志序列
    """
    # 主检测：使用配置的hop_length
    hop_length = config.get('hop_length', 256)
    min_freq = config.get('min_freq', 80)
    max_freq = config.get('max_freq', 1200)

    # 参数标准化
    frame_length = 2048
    if hop_length > frame_length // 4:
        frame_length = hop_length * 4

    # 主检测
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=max(20, min_freq),
            fmax=min(20000, max_freq),
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            n_thresholds=200,
            resolution=0.1,
            fill_na=np.nan
        )
    except Exception as e:
        raise RuntimeError(f"PYIN音高检测失败: {str(e)}") from e

    # 谐波增强检测（针对A4/E5等特定音高）
    try:
        harmonic = librosa.effects.harmonic(y, margin=config.get('harmonic_margin', 6))
        f0_harmonic, _, _ = librosa.pyin(
            harmonic,
            fmin=min_freq,
            fmax=max_freq,
            sr=sr,
            hop_length=hop_length,
            frame_length=frame_length,
            fill_na=np.nan
        )
    except Exception as e:
        print(f"谐波检测失败: {e}")
        f0_harmonic = np.full_like(f0, np.nan)

    # 置信度计算
    S, phase = librosa.magphase(librosa.stft(y, hop_length=hop_length))
    confidence = librosa.feature.spectral_flatness(S=S)
    confidence = librosa.util.normalize(confidence.squeeze(), axis=0)

    # 针对特定频率范围增强（A4: 440Hz, E5: 659Hz）
    freq_mask = np.zeros_like(f0, dtype=bool)
    target_freqs = [440, 659]  # A4和E5
    for target_freq in target_freqs:
        # 创建目标频率附近的掩码
        margin = 50  # ±50Hz
        lower_freq = target_freq - margin
        upper_freq = target_freq + margin
        freq_mask |= (f0 >= lower_freq) & (f0 <= upper_freq)

    # 增强目标频率的置信度
    enhanced_confidence = confidence.copy()
    enhanced_confidence[freq_mask] = np.clip(enhanced_confidence[freq_mask] * 1.5, 0, 1)

    # 智能融合
    alpha = np.clip(enhanced_confidence, 0.2, 0.8)
    fused_f0 = alpha * f0 + (1 - alpha) * np.nan_to_num(f0_harmonic, nan=0)

    # 后处理
    valid_mask = ~np.isnan(fused_f0)
    if np.any(valid_mask):
        x = np.arange(len(fused_f0))
        fused_f0 = np.interp(x, x[valid_mask], fused_f0[valid_mask])
        # 使用更轻的平滑，保持短音符细节
        import scipy.signal
        fused_f0 = scipy.signal.medfilt(fused_f0, kernel_size=3)  # 较小窗口
    else:
        fused_f0[:] = 0

    # 生成最终有效标志
    voiced_flag = (fused_f0 >= min_freq) & (fused_f0 <= max_freq)
    return fused_f0, voiced_flag


def create_midi_with_short_note_support(f0: np.ndarray, voiced_flag: np.ndarray,
                                      sr: int, config: dict) -> pretty_midi.PrettyMIDI:
    """
    支持短音符的MIDI生成，基于原始算法但优化短音符处理

    Args:
        f0: 基频序列
        voiced_flag: 发声标志序列
        sr: 采样率
        config: 配置参数

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    hop_length = config.get('hop_length', 256)
    min_duration = config.get('min_duration', 0.05)
    max_gap = config.get('max_gap', 0.03)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    current_note = None
    last_end = 0.0

    # 第一遍：生成原始音符（使用原始算法）
    raw_notes = []

    for i in range(len(f0)):
        time = i * hop_length / sr
        is_voiced = voiced_flag[i]
        pitch = f0[i] if is_voiced else None

        if pitch and not np.isnan(pitch):
            note_num = int(round(librosa.hz_to_midi(pitch)))

            if current_note is None:
                # 开始新音符
                current_note = {
                    'pitch': note_num,
                    'start': max(last_end, time - max_gap),
                    'end': time
                }
            elif note_num == current_note['pitch']:
                # 延续当前音符
                current_note['end'] = time
            else:
                # 音高变化时结束当前音符
                if current_note['end'] - current_note['start'] >= min_duration:
                    raw_notes.append(current_note.copy())
                last_end = current_note['end']
                current_note = {
                    'pitch': note_num,
                    'start': time,
                    'end': time
                }
        else:
            # 静音段处理
            if current_note is not None:
                if current_note['end'] - current_note['start'] >= min_duration:
                    raw_notes.append(current_note.copy())
                last_end = current_note['end']
                current_note = None

    # 处理最后一个音符
    if current_note and (current_note['end'] - current_note['start']) >= min_duration:
        raw_notes.append(current_note.copy())

    if not raw_notes:
        raise ValueError("无有效音符生成，请检查输入音频")

    print(f"  第一遍生成: {len(raw_notes)} 音符")

    # 第二遍：处理特别短的音符（小于0.1秒但大于min_duration）
    # 这些短音符直接保留，不合并
    short_note_count = 0
    for note_info in raw_notes:
        duration = note_info['end'] - note_info['start']
        if duration < 0.1:
            short_note_count += 1
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=note_info['pitch'],
            start=note_info['start'],
            end=note_info['end']
        ))

    print(f"  短音符数(<0.1s): {short_note_count}")

    midi.instruments.append(instrument)
    return midi


def main():
    print("生成v5.0_short_note_improved.mid")
    print("=" * 70)

    # 检查文件
    audio_path = Path("data/test2.mp3")
    ref_path = Path("data/test2.mid")
    output_dir = Path("results")

    if not audio_path.exists():
        print(f"错误：音频文件不存在: {audio_path}")
        return
    if not ref_path.exists():
        print(f"错误：参考MIDI不存在: {ref_path}")
        return

    print(f"音频文件: {audio_path}")
    print(f"参考MIDI: {ref_path}")
    print(f"输出目录: {output_dir}")
    print()

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    # 加载之前的版本信息
    v1_path = output_dir / "v1.0_original.mid"
    v2_path = output_dir / "v2.0_basic_optimized.mid"
    v4_path = output_dir / "v4.0_conservative_improved.mid"

    # v5.0配置：基于v2.0优化参数，但专注短音符和特定音高
    v5_config = {
        # 基于v2.0优化参数（用户说与v1.0听不出区别）
        'min_duration': 0.05,           # 降低以检测更多短音符
        'max_gap': 0.03,               # 减少间隙
        'voicing_threshold': 0.4,      # 降低发声阈值
        'harmonic_margin': 6,          # 增强谐波检测（针对A4/E5）
        'hop_length': 256,             # 提高时间分辨率

        # 保持v1.0的频率范围
        'min_freq': 80,
        'max_freq': 1200,
        'sr': 44100,

        # 短音符改进参数
        'short_note_threshold': 0.1,   # 短音符阈值
        'target_frequencies': [440, 659],  # A4和E5
    }

    print(f"v5.0配置（专注短音符和特定音高检测）:")
    for key, value in v5_config.items():
        if key not in ['target_frequencies']:
            print(f"  {key}: {value}")

    print(f"\n开始生成v5.0版本...")
    print("改进策略:")
    print("  1. 使用v2.0优化参数（与v1.0听觉无差异）")
    print("  2. 增强谐波检测（harmonic_margin=6）针对A4(440Hz)和E5(659Hz)")
    print("  3. 提高时间分辨率（hop_length=256）检测更多短音符")
    print("  4. 降低min_duration至0.05秒，检测更短音符")
    print("  5. 保持原始音符分割算法，不做音符合并（避免v3.0/v4.0问题）")
    print("  6. 专门增强A4/E5频率范围的音高检测")

    start_time = time.time()

    try:
        # 加载音频
        y, sr = librosa.load(audio_path, sr=v5_config['sr'], mono=True)

        # 音频预处理（与原始方法相同）
        y = librosa.effects.preemphasis(y, coef=0.97)
        y = librosa.util.normalize(y, axis=0)

        # 多尺度音高检测
        print(f"\n执行音高检测...")
        f0, voiced_flag = multi_scale_pitch_detection(y, sr, v5_config)

        # 生成MIDI（支持短音符）
        print(f"生成MIDI...")
        midi = create_midi_with_short_note_support(f0, voiced_flag, sr, v5_config)

        elapsed = time.time() - start_time

        # 保存v5.0版本
        v5_path = output_dir / "v5.0_short_note_improved.mid"
        midi.write(str(v5_path))

        # 分析v5.0
        v5_midi = pretty_midi.PrettyMIDI(str(v5_path))
        v5_notes = len(v5_midi.instruments[0].notes) if v5_midi.instruments else 0

        print(f"\n生成完成! 耗时: {elapsed:.1f} 秒")
        print(f"v5.0信息:")
        print(f"  音符数: {v5_notes}")

        # 对比之前的版本
        versions_to_compare = []
        for name, path in [('v1.0', v1_path), ('v2.0', v2_path), ('v4.0', v4_path), ('v5.0', v5_path)]:
            if path.exists():
                try:
                    midi_obj = pretty_midi.PrettyMIDI(str(path))
                    note_count = len(midi_obj.instruments[0].notes) if midi_obj.instruments else 0
                    versions_to_compare.append((name, path, note_count))
                except Exception as e:
                    print(f"  加载{name}失败: {e}")

        print(f"\n版本对比 (音符数):")
        for name, _, note_count in versions_to_compare:
            print(f"  {name}: {note_count} 音符")

        # 听觉相似度评估
        print(f"\n听觉相似度对比:")
        scores = {}
        for name, path, _ in versions_to_compare:
            if path.exists():
                try:
                    result = evaluate_auditory_similarity(str(ref_path), str(path))
                    scores[name] = result['scores']['overall_score']
                    print(f"  {name}综合评分: {scores[name]:.3f}")
                except Exception as e:
                    print(f"  {name}评估失败: {e}")
                    scores[name] = 0.0

        # 计算改进百分比
        if 'v1.0' in scores and 'v5.0' in scores:
            improvement_v1 = (scores['v5.0'] - scores['v1.0']) / scores['v1.0'] * 100
            print(f"  v5.0相对于v1.0改进: {improvement_v1:+.1f}%")

        # 保存v5.0元数据
        metadata = {
            'version': 'v5.0_short_note_improved',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'generation_time_seconds': elapsed,
            'config': v5_config,
            'note_count': v5_notes,
            'auditory_scores': scores,
            'midi_file': str(v5_path),
            'improvement_focus': [
                '短音符检测改进（min_duration=0.05）',
                '特定音高增强（A4:440Hz, E5:659Hz）',
                '提高时间分辨率（hop_length=256）',
                '增强谐波检测（harmonic_margin=6）',
                '保持原始音符分割算法，不做音符合并'
            ],
            'design_goal': '在保持v1.0流畅度的前提下，改进短音符和特定音高检测'
        }

        metadata_path = output_dir / "v5.0_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n元数据已保存: {metadata_path}")

        # 打印预期效果
        print(f"\nv5.0预期效果:")
        print("  1. 音符数应比v1.0多（检测更多短音符）")
        print("  2. 应保持v1.0的流畅度（不做音符合并）")
        print("  3. 应改善A4/E5音高的检测")
        print("  4. 听觉相似度评分可能提高（如果短音符检测改进）")
        print(f"\n请试听: {v5_path}")
        print("对比v1.0_original.mid，关注:")
        print("  - 流畅度是否保持（不应比v1.0差）")
        print("  - 是否检测到更多细节（特别是快速音符）")
        print("  - 整体听觉体验是否改善")

    except Exception as e:
        print(f"生成过程失败: {e}")
        import traceback
        traceback.print_exc()

        # 回退方案：使用v2.0参数生成
        print("\n尝试回退方案：使用v2.0参数生成...")
        try:
            from simple_midi_generator import SimpleMIDIGenerator

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            generator = SimpleMIDIGenerator({
                'min_duration': 0.05,
                'max_gap': 0.03,
                'voicing_threshold': 0.4,
                'harmonic_margin': 5,
                'hop_length': 256,
                'min_freq': 80,
                'max_freq': 1200,
            })
            midi = generator.process_audio(audio_bytes)

            v5_path = output_dir / "v5.0_short_note_improved.mid"
            midi.write(str(v5_path))

            note_count = len(midi.instruments[0].notes) if midi.instruments else 0
            print(f"回退方案生成v5.0: {note_count} 音符")
            print(f"保存到: {v5_path}")

        except Exception as e2:
            print(f"回退方案也失败: {e2}")


if __name__ == "__main__":
    main()