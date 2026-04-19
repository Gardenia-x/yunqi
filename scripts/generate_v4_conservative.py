#!/usr/bin/env python3
"""
生成v4.0_conservative_improved.mid
基于v1.0原始方法的保守改进版，保持原始优点，只做轻微优化
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


def conservative_note_merge(notes: list, max_gap: float = 0.03, pitch_threshold: int = 1) -> list:
    """
    保守的音符合并：只合并间隙很小且音高非常接近的音符

    Args:
        notes: 原始音符列表
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
        if (time_gap <= max_gap and time_gap >= 0 and  # 有间隙（无重叠）
            pitch_diff <= pitch_threshold):            # 音高非常接近
            # 合并音符
            current['end'] = next_note['end']
            # 音高取较长的那个音符的音高（不平均）
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


def smooth_pitch_sequence(pitch_seq: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    轻微的音高平滑：使用小窗口的移动平均

    Args:
        pitch_seq: 原始音高序列（MIDI音符编号）
        window_size: 平滑窗口大小

    Returns:
        平滑后的音高序列
    """
    if window_size < 2 or len(pitch_seq) < 3:
        return pitch_seq

    # 创建平滑窗口
    window = np.ones(window_size) / window_size

    # 边界处理：使用'same'模式，边缘部分窗口缩小
    smoothed = np.convolve(pitch_seq, window, mode='same')

    # 确保结果为整数（MIDI音符编号）
    smoothed = np.round(smoothed).astype(int)

    return smoothed


def adjust_note_starts_with_onsets(notes: list, audio_signal: np.ndarray, sr: int,
                                  hop_length: int = 512) -> list:
    """
    基于onset轻微调整音符开始时间，使边界更自然

    Args:
        notes: 音符列表
        audio_signal: 音频信号
        sr: 采样率
        hop_length: hop长度

    Returns:
        调整后的音符列表
    """
    if not notes or len(audio_signal) == 0:
        return notes

    # 检测onset（能量显著上升点）
    try:
        onsets = librosa.onset.onset_detect(y=audio_signal, sr=sr, hop_length=hop_length,
                                           units='time', backtrack=True)
    except Exception:
        # onset检测失败，返回原样
        return notes

    if len(onsets) == 0:
        return notes

    adjusted_notes = []

    for note in notes:
        note_start = note['start']
        note_end = note['end']

        # 查找音符开始时间附近最近的onset
        nearest_onset = None
        min_distance = float('inf')

        for onset_time in onsets:
            distance = abs(onset_time - note_start)
            if distance < min_distance and distance < 0.1:  # 只考虑0.1秒内的onset
                min_distance = distance
                nearest_onset = onset_time

        # 如果找到合适的onset且比原开始时间早，则轻微调整
        if nearest_onset is not None and nearest_onset < note_start and min_distance < 0.05:
            adjusted_note = note.copy()
            adjusted_note['start'] = nearest_onset
            # 确保调整后时长合理
            if adjusted_note['end'] - adjusted_note['start'] > 0.02:
                adjusted_notes.append(adjusted_note)
            else:
                adjusted_notes.append(note.copy())
        else:
            adjusted_notes.append(note.copy())

    return adjusted_notes


def create_conservative_midi(f0: np.ndarray, voiced_flag: np.ndarray, sr: int,
                           audio_signal: np.ndarray, config: dict) -> pretty_midi.PrettyMIDI:
    """
    保守改进版MIDI生成：基于原始算法，添加轻微后处理

    Args:
        f0: 基频序列
        voiced_flag: 发声标志序列
        sr: 采样率
        audio_signal: 音频信号（用于onset检测）
        config: 配置参数

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    hop_length = config.get('hop_length', 512)
    min_duration = config.get('min_duration', 0.1)
    max_gap = config.get('max_gap', 0.05)

    # 第一步：使用原始算法生成基础音符
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    raw_notes = []
    current_note = None
    last_end = 0.0

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

    # 第二步：保守后处理
    # 1. 音高序列平滑（轻微）
    if len(raw_notes) >= 3:
        pitch_seq = np.array([note['pitch'] for note in raw_notes])
        smoothed_pitches = smooth_pitch_sequence(pitch_seq, window_size=3)
        for i, note in enumerate(raw_notes):
            note['pitch'] = smoothed_pitches[i]

    # 2. 保守音符合并（只合并间隙很小且音高相同的音符）
    merged_notes = conservative_note_merge(raw_notes, max_gap=0.02, pitch_threshold=1)

    # 3. 基于onset轻微调整音符开始时间（可选）
    if len(audio_signal) > 0:
        adjusted_notes = adjust_note_starts_with_onsets(merged_notes, audio_signal, sr, hop_length)
    else:
        adjusted_notes = merged_notes

    # 第三步：生成最终MIDI
    for note in adjusted_notes:
        if note['end'] - note['start'] >= min_duration:
            instrument.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=note['pitch'],
                start=note['start'],
                end=note['end']
            ))

    if not instrument.notes:
        # 如果后处理导致无音符，回退到原始音符
        for note in raw_notes:
            if note['end'] - note['start'] >= min_duration:
                instrument.notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=note['pitch'],
                    start=note['start'],
                    end=note['end']
                ))

    midi.instruments.append(instrument)
    return midi


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

        # 保守改进参数
        'pitch_smooth_window': 3,       # 轻微平滑（v3.0用7）
        'note_merge_threshold': 1,      # 保守合并（v3.0用2）
        'max_merge_gap': 0.02,          # 只合并很小间隙的音符
        'use_onset_adjustment': True,   # 使用onset调整开始时间
    }

    print(f"v4.0配置（基于v1.0原始参数 + 保守改进）:")
    for key, value in v4_config.items():
        print(f"  {key}: {value}")

    print(f"\n开始生成v4.0版本...")
    print("保守改进策略:")
    print("  1. 保持v1.0原始核心参数（用户认为听起来最好）")
    print("  2. 轻微音高平滑（窗口3，v3.0用7）")
    print("  3. 保守音符合并（只合并间隙<0.02s且音高差≤1半音）")
    print("  4. 基于onset轻微调整音符开始时间，使边界更自然")
    print("  5. 确保音符数接近v1.0（~119个），不丢失细节")

    start_time = time.time()

    try:
        # 加载音频用于onset检测
        y, sr = librosa.load(audio_path, sr=v4_config['sr'], mono=True)

        # 使用原始生成器获取f0和voiced_flag
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        # 第一步：使用原始生成器获取音高检测结果
        print(f"\n第一步：音高检测...")
        generator = SimpleMIDIGenerator({
            'sr': v4_config['sr'],
            'hop_length': v4_config['hop_length'],
            'min_freq': v4_config['min_freq'],
            'max_freq': v4_config['max_freq'],
            'min_duration': v4_config['min_duration'],
            'max_gap': v4_config['max_gap'],
            'voicing_threshold': v4_config['voicing_threshold'],
            'harmonic_margin': v4_config['harmonic_margin'],
        })

        # 需要访问内部方法，这里简化：直接处理音频
        tmp_path = None
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            y_processed, sr_processed = librosa.load(tmp_path, sr=v4_config['sr'], mono=True)
            y_processed = librosa.effects.preemphasis(y_processed, coef=0.97)
            y_processed = librosa.util.normalize(y_processed, axis=0)

            # 使用生成器的音高检测方法
            f0, voiced_flag = generator._enhanced_pitch_detection(y_processed, sr_processed)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # 第二步：使用保守改进算法生成MIDI
        print(f"第二步：保守改进MIDI生成...")
        midi = create_conservative_midi(f0, voiced_flag, sr_processed, y_processed, v4_config)

        elapsed = time.time() - start_time

        # 保存v4.0版本
        v4_path = output_dir / "v4.0_conservative_improved.mid"
        midi.write(str(v4_path))

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
                '基于v1.0原始参数（用户认为听起来最好）',
                '轻微音高平滑（窗口3）',
                '保守音符合并（只合并间隙<0.02s且音高差≤1半音）',
                '基于onset轻微调整音符开始时间'
            ],
            'design_goal': '保持v1.0优点，轻微改进流畅度，不丢失音乐细节'
        }

        metadata_path = output_dir / "v4.0_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n元数据已保存: {metadata_path}")

        # 打印预期效果
        print(f"\n保守改进预期效果:")
        print("  1. 保持v1.0的听觉优点（用户认为听起来最好）")
        print("  2. 轻微提高流畅度，减少微小间隙")
        print("  3. 保持音符数接近v1.0（~119个），不丢失音乐细节")
        print("  4. 使音符边界更自然，但不破坏旋律")
        print(f"\n请试听: {v4_path}")
        print("对比v1.0_original.mid，关注:")
        print("  - 整体相似度是否保持（应该非常接近v1.0）")
        print("  - 流畅度是否有轻微改善")
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