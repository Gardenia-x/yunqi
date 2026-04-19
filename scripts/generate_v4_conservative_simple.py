#!/usr/bin/env python3
"""
生成v4.0_conservative_improved.mid
简单保守改进版：使用v1.0原始方法生成，然后进行轻微后处理
"""
import sys
import time
from pathlib import Path
import json
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_midi_generator import SimpleMIDIGenerator
from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi
import librosa


def conservative_note_merge(notes: list, max_gap: float = 0.02, pitch_threshold: int = 1) -> list:
    """
    保守的音符合并：只合并间隙很小且音高非常接近的音符

    Args:
        notes: 原始音符列表 [{'pitch', 'start', 'end'}]
        max_gap: 最大合并间隙（秒）
        pitch_threshold: 音高合并阈值（半音）

    Returns:
        合并后的音符列表
    """
    if not notes:
        return []

    merged = []
    current = notes[0].copy()

    for i in range(1, len(notes)):
        next_note = notes[i]

        # 计算间隙和音高差
        time_gap = next_note['start'] - current['end']
        pitch_diff = abs(next_note['pitch'] - current['pitch'])

        # 保守合并条件：间隙很小且音高相同或非常接近
        if (0 <= time_gap <= max_gap and  # 有间隙（无重叠）
            pitch_diff <= pitch_threshold):  # 音高非常接近
            # 合并音符
            current['end'] = next_note['end']
            # 音高取较长的那个音符的音高
            current_duration = current['end'] - current['start']
            next_duration = next_note['end'] - next_note['start']
            if next_duration > current_duration:
                current['pitch'] = next_note['pitch']
        else:
            # 不合并，保存当前音符
            merged.append(current.copy())
            current = next_note.copy()

    # 处理最后一个音符
    merged.append(current)
    return merged


def smooth_pitch_sequence(notes: list, window_size: int = 3) -> list:
    """
    轻微的音高平滑：使用小窗口的移动平均

    Args:
        notes: 音符列表
        window_size: 平滑窗口大小

    Returns:
        平滑后的音符列表
    """
    if len(notes) < 3 or window_size < 2:
        return notes

    # 提取音高序列
    pitches = np.array([note['pitch'] for note in notes])

    # 创建平滑窗口
    window = np.ones(window_size) / window_size

    # 边界处理：使用'same'模式
    smoothed_pitches = np.convolve(pitches, window, mode='same')

    # 确保结果为整数（MIDI音符编号）
    smoothed_pitches = np.round(smoothed_pitches).astype(int)

    # 更新音符音高
    smoothed_notes = []
    for i, note in enumerate(notes):
        new_note = note.copy()
        new_note['pitch'] = smoothed_pitches[i]
        smoothed_notes.append(new_note)

    return smoothed_notes


def adjust_note_starts_with_energy(notes: list, audio_signal: np.ndarray, sr: int) -> list:
    """
    基于音频能量轻微调整音符开始时间

    Args:
        notes: 音符列表
        audio_signal: 音频信号
        sr: 采样率

    Returns:
        调整后的音符列表
    """
    if not notes or len(audio_signal) == 0:
        return notes

    adjusted_notes = []

    for note in notes:
        start_time = note['start']
        end_time = note['end']

        # 查找音符开始时间附近的能量上升点
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 确保索引在范围内
        start_sample = max(0, min(start_sample, len(audio_signal) - 100))
        end_sample = max(100, min(end_sample, len(audio_signal)))

        # 分析开始前0.05秒到开始后0.05秒的能量
        lookback_samples = int(0.05 * sr)
        lookahead_samples = int(0.05 * sr)

        analysis_start = max(0, start_sample - lookback_samples)
        analysis_end = min(len(audio_signal), start_sample + lookahead_samples)

        if analysis_end <= analysis_start:
            adjusted_notes.append(note.copy())
            continue

        segment = audio_signal[analysis_start:analysis_end]

        # 计算能量
        energy = np.convolve(np.abs(segment), np.ones(256)/256, mode='same')

        # 归一化
        if np.max(energy) > np.min(energy):
            energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
        else:
            energy_norm = energy

        # 查找能量显著上升点（梯度最大）
        gradient = np.gradient(energy_norm)
        max_gradient_idx = np.argmax(gradient)

        # 如果最大梯度点显著且在前半部分，调整开始时间
        if gradient[max_gradient_idx] > 0.1 and max_gradient_idx < len(gradient) // 2:
            adjusted_start = start_time + (max_gradient_idx - lookback_samples/sr)
            # 确保调整后的开始时间不晚于原开始时间且不早于前0.03秒
            if start_time - 0.03 < adjusted_start < start_time:
                new_note = note.copy()
                new_note['start'] = adjusted_start
                adjusted_notes.append(new_note)
                continue

        adjusted_notes.append(note.copy())

    return adjusted_notes


def apply_conservative_improvements(original_midi: pretty_midi.PrettyMIDI,
                                  audio_signal: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
    """
    对原始MIDI应用保守改进

    Args:
        original_midi: 原始MIDI对象
        audio_signal: 音频信号
        sr: 采样率

    Returns:
        改进后的MIDI对象
    """
    if not original_midi.instruments:
        return original_midi

    # 提取原始音符
    instrument = original_midi.instruments[0]
    original_notes = []
    for note in instrument.notes:
        original_notes.append({
            'pitch': note.pitch,
            'start': note.start,
            'end': note.end
        })

    if not original_notes:
        return original_midi

    print(f"  原始音符数: {len(original_notes)}")

    # 步骤1：轻微音高平滑
    smoothed_notes = smooth_pitch_sequence(original_notes, window_size=3)
    print(f"  音高平滑后: {len(smoothed_notes)} 音符")

    # 步骤2：保守音符合并
    merged_notes = conservative_note_merge(smoothed_notes, max_gap=0.02, pitch_threshold=1)
    print(f"  音符合并后: {len(merged_notes)} 音符")

    # 步骤3：基于能量调整音符开始时间
    adjusted_notes = adjust_note_starts_with_energy(merged_notes, audio_signal, sr)
    print(f"  边界调整后: {len(adjusted_notes)} 音符")

    # 创建新的MIDI对象
    improved_midi = pretty_midi.PrettyMIDI()
    improved_instrument = pretty_midi.Instrument(program=0)

    # 添加改进后的音符
    for note_info in adjusted_notes:
        if note_info['end'] - note_info['start'] >= 0.02:  # 最小时长限制
            improved_instrument.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=note_info['pitch'],
                start=note_info['start'],
                end=note_info['end']
            ))

    # 如果改进后无音符，回退到原始
    if not improved_instrument.notes:
        print("  警告：改进后无有效音符，回退到原始")
        return original_midi

    improved_midi.instruments.append(improved_instrument)
    return improved_midi


def main():
    print("生成v4.0_conservative_improved.mid")
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
    v3_path = output_dir / "v3.0_algorithm_improved.mid"

    # v4.0配置：基于v1.0原始参数
    v4_config = {
        # v1.0原始参数（用户认为听起来最好）
        'sr': 44100,
        'hop_length': 512,              # 原始值
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.1,            # 原始值
        'max_gap': 0.05,                # 原始值
        'voicing_threshold': 0.6,       # 原始值
        'harmonic_margin': 3,           # 原始值
    }

    print(f"v4.0配置（完全使用v1.0原始参数）:")
    for key, value in v4_config.items():
        print(f"  {key}: {value}")

    print(f"\n开始生成v4.0版本...")
    print("保守改进策略:")
    print("  1. 完全使用v1.0原始参数生成MIDI")
    print("  2. 轻微音高平滑（窗口3）")
    print("  3. 保守音符合并（只合并间隙<0.02s且音高差≤1半音）")
    print("  4. 基于音频能量轻微调整音符开始时间")
    print("  5. 确保音符数接近v1.0（~119个），不丢失细节")

    start_time = time.time()

    try:
        # 加载音频用于能量分析
        print(f"\n加载音频...")
        y, sr = librosa.load(audio_path, sr=v4_config['sr'], mono=True)

        # 第一步：使用v1.0原始方法生成MIDI
        print(f"使用v1.0原始方法生成MIDI...")
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        generator = SimpleMIDIGenerator(v4_config)
        original_midi = generator.process_audio(audio_bytes)

        # 第二步：应用保守改进
        print(f"应用保守改进...")
        improved_midi = apply_conservative_improvements(original_midi, y, sr)

        elapsed = time.time() - start_time

        # 保存v4.0版本
        v4_path = output_dir / "v4.0_conservative_improved.mid"
        improved_midi.write(str(v4_path))

        # 分析v4.0
        v4_midi = pretty_midi.PrettyMIDI(str(v4_path))
        v4_notes = len(v4_midi.instruments[0].notes) if v4_midi.instruments else 0

        print(f"\n生成完成! 耗时: {elapsed:.1f} 秒")
        print(f"v4.0信息:")
        print(f"  音符数: {v4_notes}")

        # 对比之前的版本
        versions_to_compare = []
        for name, path in [('v1.0', v1_path), ('v2.0', v2_path), ('v3.0', v3_path), ('v4.0', v4_path)]:
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
        if 'v1.0' in scores and 'v4.0' in scores:
            improvement_v1 = (scores['v4.0'] - scores['v1.0']) / scores['v1.0'] * 100
            print(f"  v4.0相对于v1.0改进: {improvement_v1:+.1f}%")

        if 'v3.0' in scores and 'v4.0' in scores:
            improvement_v3 = (scores['v4.0'] - scores['v3.0']) / scores['v3.0'] * 100
            print(f"  v4.0相对于v3.0改进: {improvement_v3:+.1f}%")

        # 保存v4.0元数据
        metadata = {
            'version': 'v4.0_conservative_improved',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'generation_time_seconds': elapsed,
            'config': v4_config,
            'note_count': v4_notes,
            'auditory_scores': scores,
            'midi_file': str(v4_path),
            'conservative_improvements': [
                '完全使用v1.0原始参数生成',
                '轻微音高平滑（窗口3）',
                '保守音符合并（只合并间隙<0.02s且音高差≤1半音）',
                '基于音频能量轻微调整音符开始时间'
            ],
            'design_goal': '保持v1.0优点，尝试轻微改进流畅度'
        }

        metadata_path = output_dir / "v4.0_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n元数据已保存: {metadata_path}")

        # 打印预期效果
        print(f"\n保守改进预期效果:")
        print("  1. 保持v1.0的听觉优点（用户认为听起来最好）")
        print("  2. 轻微提高流畅度，可能减少微小间隙")
        print("  3. 音符数应接近v1.0（轻微减少）")
        print("  4. 综合评分应接近v1.0（轻微变化）")
        print(f"\n请试听: {v4_path}")
        print("对比v1.0_original.mid，关注:")
        print("  - 整体相似度是否保持（应该非常接近v1.0）")
        print("  - 流畅度是否有轻微改善（可能不明显）")
        print("  - 有无丢失重要的音乐细节")

    except Exception as e:
        print(f"生成过程失败: {e}")
        import traceback
        traceback.print_exc()

        # 回退方案：直接使用v1.0原始方法
        print("\n尝试回退方案：直接复制v1.0...")
        try:
            import shutil
            v4_path = output_dir / "v4.0_conservative_improved.mid"
            shutil.copy2(v1_path, v4_path)
            print(f"回退方案：直接复制v1.0到{v4_path}")
        except Exception as e2:
            print(f"回退方案也失败: {e2}")


if __name__ == "__main__":
    main()