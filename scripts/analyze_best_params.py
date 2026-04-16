#!/usr/bin/env python3
"""
分析最佳参数生成的MIDI与参考MIDI的差异
"""
import sys
from pathlib import Path
import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).parent))

def analyze_best_params():
    print("最佳参数缺失音符分析")
    print("=" * 60)

    ref_path = Path("../data/test2.mid")
    best_path = Path("../results/test_high_pitch_focused.mid")  # 最佳参数生成的MIDI

    if not ref_path.exists():
        print(f"参考MIDI不存在: {ref_path}")
        return
    if not best_path.exists():
        print(f"最佳参数MIDI不存在: {best_path}")
        return

    # 加载MIDI文件
    ref_midi = pretty_midi.PrettyMIDI(str(ref_path))
    best_midi = pretty_midi.PrettyMIDI(str(best_path))

    ref_notes = ref_midi.instruments[0].notes if ref_midi.instruments else []
    best_notes = best_midi.instruments[0].notes if best_midi.instruments else []

    print(f"参考MIDI音符数: {len(ref_notes)}")
    print(f"最佳参数MIDI音符数: {len(best_notes)}")
    print(f"缺失音符数: {len(ref_notes) - len(best_notes)}")
    print()

    # 分析音高分布
    ref_pitches = [n.pitch for n in ref_notes]
    best_pitches = [n.pitch for n in best_notes]

    print("音高分布:")
    ref_pitch_counts = {}
    for pitch in ref_pitches:
        ref_pitch_counts[pitch] = ref_pitch_counts.get(pitch, 0) + 1

    best_pitch_counts = {}
    for pitch in best_pitches:
        best_pitch_counts[pitch] = best_pitch_counts.get(pitch, 0) + 1

    # 找出缺失的音高
    missing_pitches = []
    for pitch, count in ref_pitch_counts.items():
        best_count = best_pitch_counts.get(pitch, 0)
        missing = count - best_count
        if missing > 0:
            missing_pitches.append((pitch, missing, count))

    if missing_pitches:
        print("\n缺失的音高（按缺失数量排序）:")
        missing_pitches.sort(key=lambda x: x[1], reverse=True)
        for pitch, missing, total in missing_pitches[:20]:  # 显示前20个
            note_name = pretty_midi.note_number_to_name(pitch)
            percentage = missing/total*100
            print(f"  {note_name} ({pitch}): 缺失{missing}/{total}个 ({percentage:.1f}%)")
    else:
        print("没有特定音高大量缺失")

    # 分析音符时长分布
    ref_durations = [n.end - n.start for n in ref_notes]
    best_durations = [n.end - n.start for n in best_notes]

    print(f"\n音符时长统计:")
    print(f"  参考 - 平均: {np.mean(ref_durations):.3f}s, 最小: {np.min(ref_durations):.3f}s, 最大: {np.max(ref_durations):.3f}s")
    print(f"  最佳 - 平均: {np.mean(best_durations):.3f}s, 最小: {np.min(best_durations):.3f}s, 最大: {np.max(best_durations):.3f}s")

    # 分析短音符缺失情况
    short_thresholds = [0.1, 0.2, 0.3, 0.5]
    for threshold in short_thresholds:
        ref_short = [d for d in ref_durations if d < threshold]
        best_short = [d for d in best_durations if d < threshold]

        print(f"\n短音符分析 (<{threshold}s):")
        print(f"  参考短音符数: {len(ref_short)}/{len(ref_notes)} ({len(ref_short)/len(ref_notes)*100:.1f}%)")
        print(f"  最佳短音符数: {len(best_short)}/{len(best_notes)} ({len(best_short)/len(best_notes)*100:.1f}%)")
        if len(ref_short) > 0:
            print(f"  短音符召回率: {len(best_short)/len(ref_short)*100:.1f}%")

    # 分析时间分布
    ref_times = [n.start for n in ref_notes]
    best_times = [n.start for n in best_notes]

    print(f"\n时间分布:")
    print(f"  参考时间范围: {np.min(ref_times):.1f}s - {np.max(ref_times):.1f}s")
    print(f"  最佳时间范围: {np.min(best_times):.1f}s - {np.max(best_times):.1f}s")

    # 检查是否有时间区域完全缺失音符
    time_bins = np.arange(0, np.max(ref_times) + 5, 5)  # 每5秒一个区间
    missing_bins = []
    for i in range(len(time_bins)-1):
        bin_start, bin_end = time_bins[i], time_bins[i+1]
        ref_in_bin = sum(1 for t in ref_times if bin_start <= t < bin_end)
        best_in_bin = sum(1 for t in best_times if bin_start <= t < bin_end)

        if ref_in_bin > 0 and best_in_bin == 0:
            missing_bins.append((bin_start, bin_end, ref_in_bin))

    if missing_bins:
        print(f"\n警告: 以下时间区间完全缺失音符:")
        for bin_start, bin_end, ref_count in missing_bins:
            print(f"  {bin_start:.1f}s-{bin_end:.1f}s: 参考有{ref_count}个音符，生成0个")
    else:
        print(f"\n所有时间区间都有音符检测到")

    # 音高范围分析
    ref_pitch_min, ref_pitch_max = min(ref_pitches), max(ref_pitches)
    best_pitch_min, best_pitch_max = min(best_pitches), max(best_pitches)

    print(f"\n音高范围:")
    print(f"  参考音高范围: {pretty_midi.note_number_to_name(ref_pitch_min)}({ref_pitch_min}) - {pretty_midi.note_number_to_name(ref_pitch_max)}({ref_pitch_max})")
    print(f"  最佳音高范围: {pretty_midi.note_number_to_name(best_pitch_min)}({best_pitch_min}) - {pretty_midi.note_number_to_name(best_pitch_max)}({best_pitch_max})")

    # 检查音高范围是否匹配
    if ref_pitch_min < best_pitch_min:
        print(f"  警告: 生成的最低音高高于参考最低音高")
    if ref_pitch_max > best_pitch_max:
        print(f"  警告: 生成的最高音高低于参考最高音高")

    # 建议
    print("\n" + "=" * 60)
    print("优化建议:")

    # 短音符检测不足
    ref_short_01 = len([d for d in ref_durations if d < 0.1])
    best_short_01 = len([d for d in best_durations if d < 0.1])
    if ref_short_01 > 0 and best_short_01 < ref_short_01 * 0.5:
        print("1. 极短音符(<0.1s)检测严重不足:")
        print("   - 进一步降低min_note_duration (当前0.09s)")
        print("   - 改进onset检测灵敏度")
        print("   - 使用更小的hop_length以提高时间分辨率")

    # 特定音高缺失
    if missing_pitches:
        top_missing = missing_pitches[:3]
        print("2. 特定音高缺失严重:")
        for pitch, missing, total in top_missing:
            note_name = pretty_midi.note_number_to_name(pitch)
            freq = pretty_midi.note_number_to_hz(pitch)
            print(f"   - {note_name} ({pitch}, {freq:.1f}Hz): 缺失{missing}/{total}")
        print("   - 调整min_freq/max_freq范围")
        print("   - 尝试不同的pitch_method参数")
        print("   - 检查谐波增强设置")

    # 总体召回率
    recall = len(best_notes) / len(ref_notes) if len(ref_notes) > 0 else 0
    if recall < 0.7:
        print(f"3. 总体召回率较低 ({recall:.1%}):")
        print("   - 降低voicing_threshold (当前0.45)")
        print("   - 提高confidence_weight (当前0.65)")
        print("   - 检查音高检测算法是否漏检")

if __name__ == "__main__":
    analyze_best_params()