#!/usr/bin/env python3
"""
分析MIDI文件的音乐相似性
对比baseline和优化版本，理解为什么baseline听起来更像
"""
import numpy as np
import pretty_midi
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_midi_similarity():
    print("MIDI音乐相似性分析")
    print("=" * 70)

    # 加载文件
    ref_path = Path("data/test2.mid")
    baseline_path = Path("results/test_baseline.mid")
    optimized_path = Path("results/test_improved_default.mid")

    ref_midi = pretty_midi.PrettyMIDI(str(ref_path))
    baseline_midi = pretty_midi.PrettyMIDI(str(baseline_path))
    optimized_midi = pretty_midi.PrettyMIDI(str(optimized_path))

    ref_notes = ref_midi.instruments[0].notes if ref_midi.instruments else []
    baseline_notes = baseline_midi.instruments[0].notes if baseline_midi.instruments else []
    optimized_notes = optimized_midi.instruments[0].notes if optimized_midi.instruments else []

    print(f"参考MIDI: {len(ref_notes)} 音符")
    print(f"Baseline: {len(baseline_notes)} 音符")
    print(f"优化版: {len(optimized_notes)} 音符")
    print()

    # 1. 音高分布分析
    print("1. 音高分布分析")
    print("-" * 40)

    ref_pitches = [n.pitch for n in ref_notes]
    baseline_pitches = [n.pitch for n in baseline_notes]
    optimized_pitches = [n.pitch for n in optimized_notes]

    def pitch_to_name(pitch):
        return pretty_midi.note_number_to_name(pitch)

    print(f"参考音高范围: {min(ref_pitches)}-{max(ref_pitches)} ({pitch_to_name(min(ref_pitches))} to {pitch_to_name(max(ref_pitches))})")
    print(f"Baseline音高范围: {min(baseline_pitches)}-{max(baseline_pitches)} ({pitch_to_name(min(baseline_pitches))} to {pitch_to_name(max(baseline_pitches))})")
    print(f"优化版音高范围: {min(optimized_pitches)}-{max(optimized_pitches)} ({pitch_to_name(min(optimized_pitches))} to {pitch_to_name(max(optimized_pitches))})")

    # 音高覆盖率
    ref_pitch_set = set(ref_pitches)
    baseline_coverage = len([p for p in baseline_pitches if p in ref_pitch_set]) / len(ref_pitch_set)
    optimized_coverage = len([p for p in optimized_pitches if p in ref_pitch_set]) / len(ref_pitch_set)

    print(f"\n音高覆盖率:")
    print(f"  Baseline: {baseline_coverage*100:.1f}% (覆盖{len(set(baseline_pitches) & ref_pitch_set)}/{len(ref_pitch_set)}个音高)")
    print(f"  优化版: {optimized_coverage*100:.1f}% (覆盖{len(set(optimized_pitches) & ref_pitch_set)}/{len(ref_pitch_set)}个音高)")

    # 2. 时长分布分析
    print("\n2. 时长分布分析")
    print("-" * 40)

    ref_durations = [n.end - n.start for n in ref_notes]
    baseline_durations = [n.end - n.start for n in baseline_notes]
    optimized_durations = [n.end - n.start for n in optimized_notes]

    def duration_stats(durations):
        if not durations:
            return 0, 0, 0, 0
        return np.mean(durations), np.median(durations), np.min(durations), np.max(durations)

    ref_mean, ref_median, ref_min, ref_max = duration_stats(ref_durations)
    baseline_mean, baseline_median, baseline_min, baseline_max = duration_stats(baseline_durations)
    optimized_mean, optimized_median, optimized_min, optimized_max = duration_stats(optimized_durations)

    print(f"参考平均时长: {ref_mean:.3f}s, 中位数: {ref_median:.3f}s, 范围: {ref_min:.3f}-{ref_max:.3f}s")
    print(f"Baseline平均时长: {baseline_mean:.3f}s, 中位数: {baseline_median:.3f}s, 范围: {baseline_min:.3f}-{baseline_max:.3f}s")
    print(f"优化版平均时长: {optimized_mean:.3f}s, 中位数: {optimized_median:.3f}s, 范围: {optimized_min:.3f}-{optimized_max:.3f}s")

    # 短音符比例
    short_threshold = 0.3
    ref_short = len([d for d in ref_durations if d < short_threshold]) / len(ref_durations)
    baseline_short = len([d for d in baseline_durations if d < short_threshold]) / len(baseline_durations)
    optimized_short = len([d for d in optimized_durations if d < short_threshold]) / len(optimized_durations)

    print(f"\n短音符比例 (<{short_threshold}s):")
    print(f"  参考: {ref_short*100:.1f}% ({int(ref_short*len(ref_durations))}/{len(ref_durations)})")
    print(f"  Baseline: {baseline_short*100:.1f}% ({int(baseline_short*len(baseline_durations))}/{len(baseline_durations)})")
    print(f"  优化版: {optimized_short*100:.1f}% ({int(optimized_short*len(optimized_durations))}/{len(optimized_durations)})")

    # 3. 时序分析（音符密度）
    print("\n3. 时序分析（音符密度）")
    print("-" * 40)

    # 计算每5秒的音符密度
    total_duration = ref_midi.get_end_time()
    window_size = 5.0

    def note_density(notes, total_dur, window):
        density = []
        for t in np.arange(0, total_dur, window):
            count = len([n for n in notes if n.start >= t and n.start < t + window])
            density.append(count)
        return density

    ref_density = note_density(ref_notes, total_duration, window_size)
    baseline_density = note_density(baseline_notes, total_duration, window_size)
    optimized_density = note_density(optimized_notes, total_duration, window_size)

    # 计算密度相关性
    baseline_corr = np.corrcoef(ref_density, baseline_density)[0, 1] if len(ref_density) == len(baseline_density) else 0
    optimized_corr = np.corrcoef(ref_density, optimized_density)[0, 1] if len(ref_density) == len(optimized_density) else 0

    print(f"时间窗口: {window_size}s (共{len(ref_density)}个窗口)")
    print(f"Baseline密度相关性: {baseline_corr:.3f}")
    print(f"优化版密度相关性: {optimized_corr:.3f}")

    # 4. 旋律轮廓分析（音高变化模式）
    print("\n4. 旋律轮廓分析")
    print("-" * 40)

    def pitch_contour(notes):
        """提取旋律轮廓（音高变化序列）"""
        if not notes:
            return []
        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: n.start)
        return [n.pitch for n in sorted_notes]

    ref_contour = pitch_contour(ref_notes)
    baseline_contour = pitch_contour(baseline_notes)
    optimized_contour = pitch_contour(optimized_notes)

    # 简单比较：音高变化的趋势
    def contour_similarity(contour1, contour2):
        """计算两个轮廓的相似度（基于DTW距离）"""
        if not contour1 or not contour2:
            return 0

        # 简单的归一化互相关
        from scipy import signal
        min_len = min(len(contour1), len(contour2))
        c1_norm = (np.array(contour1[:min_len]) - np.mean(contour1[:min_len])) / np.std(contour1[:min_len])
        c2_norm = (np.array(contour2[:min_len]) - np.mean(contour2[:min_len])) / np.std(contour2[:min_len])

        correlation = np.corrcoef(c1_norm, c2_norm)[0, 1]
        return correlation if not np.isnan(correlation) else 0

    baseline_contour_sim = contour_similarity(ref_contour, baseline_contour)
    optimized_contour_sim = contour_similarity(ref_contour, optimized_contour)

    print(f"Baseline旋律轮廓相似度: {baseline_contour_sim:.3f}")
    print(f"优化版旋律轮廓相似度: {optimized_contour_sim:.3f}")

    # 5. 关键发现
    print("\n5. 关键发现")
    print("-" * 40)

    findings = []

    # 音高覆盖
    if baseline_coverage > optimized_coverage:
        findings.append(f"Baseline音高覆盖更好 ({baseline_coverage*100:.1f}% vs {optimized_coverage*100:.1f}%)")

    # 时长分布
    baseline_duration_diff = abs(baseline_mean - ref_mean) / ref_mean
    optimized_duration_diff = abs(optimized_mean - ref_mean) / ref_mean
    if baseline_duration_diff < optimized_duration_diff:
        findings.append(f"Baseline平均时长更接近参考 ({baseline_mean:.3f}s vs 参考{ref_mean:.3f}s)")

    # 密度相关性
    if baseline_corr > optimized_corr:
        findings.append(f"Baseline音符密度相关性更高 ({baseline_corr:.3f} vs {optimized_corr:.3f})")

    # 旋律轮廓
    if baseline_contour_sim > optimized_contour_sim:
        findings.append(f"Baseline旋律轮廓更相似 ({baseline_contour_sim:.3f} vs {optimized_contour_sim:.3f})")

    if findings:
        print("Baseline听起来更像的原因:")
        for i, finding in enumerate(findings, 1):
            print(f"  {i}. {finding}")
    else:
        print("未发现明显差异，可能需要进一步分析")

    # 6. 建议
    print("\n6. 优化建议")
    print("-" * 40)

    suggestions = []

    if optimized_coverage < baseline_coverage:
        suggestions.append("提高音高覆盖率：调整min_freq/max_freq范围")

    if optimized_short < ref_short * 0.5:  # 优化版短音符比例不到参考的一半
        suggestions.append(f"短音符检测不足：当前{optimized_short*100:.1f}%，参考{ref_short*100:.1f}%")

    if optimized_contour_sim < baseline_contour_sim:
        suggestions.append("保持旋律轮廓：避免过度追求F1分数而破坏音乐结构")

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

    return {
        'ref_notes': len(ref_notes),
        'baseline_notes': len(baseline_notes),
        'optimized_notes': len(optimized_notes),
        'baseline_coverage': baseline_coverage,
        'optimized_coverage': optimized_coverage,
        'baseline_duration_sim': 1 - baseline_duration_diff,
        'optimized_duration_sim': 1 - optimized_duration_diff,
        'baseline_density_corr': baseline_corr,
        'optimized_density_corr': optimized_corr,
        'baseline_contour_sim': baseline_contour_sim,
        'optimized_contour_sim': optimized_contour_sim
    }

if __name__ == "__main__":
    analyze_midi_similarity()