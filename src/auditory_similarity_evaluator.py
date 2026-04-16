#!/usr/bin/env python3
"""
听觉相似度评估器
基于音乐相似度分析，专注于听觉相关指标
"""
import numpy as np
import pretty_midi
from typing import List, Dict, Any, Tuple, Optional
from scipy import signal


class AuditorySimilarityEvaluator:
    """
    听觉相似度评估器
    评估生成的MIDI与参考MIDI在听觉上的相似度
    """

    def __init__(self, reference_midi: pretty_midi.PrettyMIDI):
        """
        初始化评估器

        Args:
            reference_midi: 参考MIDI文件
        """
        self.reference_midi = reference_midi
        self.ref_notes = self._extract_notes(reference_midi)

        # 预计算参考特征（提高性能）
        self._precompute_features()

    def _extract_notes(self, midi: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
        """提取MIDI中的音符列表"""
        if midi.instruments:
            return midi.instruments[0].notes
        return []

    def _precompute_features(self):
        """预计算参考MIDI的特征"""
        self.ref_pitches = [n.pitch for n in self.ref_notes]
        self.ref_durations = [n.end - n.start for n in self.ref_notes]
        self.ref_pitch_set = set(self.ref_pitches)
        self.ref_contour = self._extract_pitch_contour(self.ref_notes)

        # 时长统计
        if self.ref_durations:
            self.ref_mean_duration = np.mean(self.ref_durations)
            self.ref_median_duration = np.median(self.ref_durations)
            self.ref_short_ratio = len([d for d in self.ref_durations if d < 0.3]) / len(self.ref_durations)
        else:
            self.ref_mean_duration = 0
            self.ref_median_duration = 0
            self.ref_short_ratio = 0

    def evaluate(self, generated_midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        评估生成MIDI的听觉相似度

        Args:
            generated_midi: 生成的MIDI文件

        Returns:
            包含各项评分指标的字典
        """
        gen_notes = self._extract_notes(generated_midi)

        if not gen_notes:
            return self._empty_scores()

        # 计算各项指标
        pitch_coverage = self._calculate_pitch_coverage(gen_notes)
        duration_similarity = self._calculate_duration_similarity(gen_notes)
        melody_similarity = self._calculate_melody_similarity(gen_notes)
        rhythm_similarity = self._calculate_rhythm_similarity(gen_notes)
        density_correlation = self._calculate_density_correlation(gen_notes)

        # 综合评分（基于计划中的权重）
        weights = {
            'melody_similarity': 0.4,    # 旋律最重要
            'rhythm_similarity': 0.3,    # 节奏流畅度
            'pitch_coverage': 0.2,       # 音高覆盖
            'density_correlation': 0.1,   # 密度相关
        }

        # 计算加权总分
        weighted_score = (
            weights['melody_similarity'] * melody_similarity +
            weights['rhythm_similarity'] * rhythm_similarity +
            weights['pitch_coverage'] * pitch_coverage +
            weights['density_correlation'] * density_correlation
        )

        return {
            'overall_score': weighted_score,
            'melody_similarity': melody_similarity,
            'rhythm_similarity': rhythm_similarity,
            'pitch_coverage': pitch_coverage,
            'density_correlation': density_correlation,
            'duration_similarity': duration_similarity,
            'note_count': len(gen_notes),
            'ref_note_count': len(self.ref_notes),
            'short_note_ratio': self._calculate_short_note_ratio(gen_notes),
        }

    def _empty_scores(self) -> Dict[str, float]:
        """返回空MIDI的评分"""
        return {
            'overall_score': 0.0,
            'melody_similarity': 0.0,
            'rhythm_similarity': 0.0,
            'pitch_coverage': 0.0,
            'density_correlation': 0.0,
            'duration_similarity': 0.0,
            'note_count': 0,
            'ref_note_count': len(self.ref_notes),
            'short_note_ratio': 0.0,
        }

    def _calculate_pitch_coverage(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算音高覆盖率"""
        if not self.ref_pitch_set:
            return 0.0

        gen_pitches = [n.pitch for n in gen_notes]
        gen_pitch_set = set(gen_pitches)

        # 计算覆盖的参考音高比例
        covered = len(gen_pitch_set.intersection(self.ref_pitch_set))
        return covered / len(self.ref_pitch_set)

    def _calculate_duration_similarity(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算时长分布相似度"""
        if not self.ref_durations or not gen_notes:
            return 0.0

        gen_durations = [n.end - n.start for n in gen_notes]
        gen_mean = np.mean(gen_durations)

        # 计算平均时长的相对差异（越小越好，转为相似度）
        if self.ref_mean_duration > 0:
            diff = abs(gen_mean - self.ref_mean_duration) / self.ref_mean_duration
            return max(0.0, 1.0 - diff)  # 差异越小，相似度越高
        return 0.0

    def _calculate_melody_similarity(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算旋律轮廓相似度"""
        if not self.ref_contour or not gen_notes:
            return 0.0

        gen_contour = self._extract_pitch_contour(gen_notes)
        return self._contour_similarity(self.ref_contour, gen_contour)

    def _calculate_rhythm_similarity(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算节奏相似度（基于时长分布和短音符比例）"""
        if not gen_notes:
            return 0.0

        gen_durations = [n.end - n.start for n in gen_notes]

        # 1. 时长分布相似度（已单独计算）
        duration_sim = self._calculate_duration_similarity(gen_notes)

        # 2. 短音符比例相似度
        gen_short_ratio = len([d for d in gen_durations if d < 0.3]) / len(gen_durations)
        short_ratio_sim = 1.0 - abs(gen_short_ratio - self.ref_short_ratio)

        # 综合节奏相似度
        return 0.7 * duration_sim + 0.3 * short_ratio_sim

    def _calculate_density_correlation(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算音符密度相关性"""
        if not gen_notes:
            return 0.0

        # 使用5秒窗口
        window_size = 5.0
        total_duration = max(
            self.reference_midi.get_end_time(),
            max([n.end for n in gen_notes]) if gen_notes else 0
        )

        ref_density = self._calculate_note_density(self.ref_notes, total_duration, window_size)
        gen_density = self._calculate_note_density(gen_notes, total_duration, window_size)

        # 计算相关性
        min_len = min(len(ref_density), len(gen_density))
        if min_len < 3:  # 数据点太少
            return 0.0

        ref_array = np.array(ref_density[:min_len])
        gen_array = np.array(gen_density[:min_len])

        correlation = np.corrcoef(ref_array, gen_array)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0  # 负相关视为不相似

    def _calculate_short_note_ratio(self, gen_notes: List[pretty_midi.Note]) -> float:
        """计算短音符比例"""
        if not gen_notes:
            return 0.0

        gen_durations = [n.end - n.start for n in gen_notes]
        short_count = len([d for d in gen_durations if d < 0.3])
        return short_count / len(gen_durations)

    # ===== 工具函数 =====

    @staticmethod
    def _extract_pitch_contour(notes: List[pretty_midi.Note]) -> List[int]:
        """提取旋律轮廓（音高变化序列）"""
        if not notes:
            return []
        # 按开始时间排序
        sorted_notes = sorted(notes, key=lambda n: n.start)
        return [n.pitch for n in sorted_notes]

    @staticmethod
    def _contour_similarity(contour1: List[int], contour2: List[int]) -> float:
        """计算两个轮廓的相似度（基于归一化互相关）"""
        if not contour1 or not contour2:
            return 0.0

        min_len = min(len(contour1), len(contour2))
        if min_len < 5:  # 数据点太少
            return 0.0

        c1 = np.array(contour1[:min_len])
        c2 = np.array(contour2[:min_len])

        # 归一化
        c1_norm = (c1 - np.mean(c1)) / (np.std(c1) + 1e-10)
        c2_norm = (c2 - np.mean(c2)) / (np.std(c2) + 1e-10)

        correlation = np.corrcoef(c1_norm, c2_norm)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0  # 负相关视为不相似

    @staticmethod
    def _calculate_note_density(notes: List[pretty_midi.Note], total_duration: float,
                               window_size: float) -> List[int]:
        """计算音符密度"""
        density = []
        for t in np.arange(0, total_duration, window_size):
            count = len([n for n in notes if n.start >= t and n.start < t + window_size])
            density.append(count)
        return density

    def generate_report(self, generated_midi: pretty_midi.PrettyMIDI) -> str:
        """生成详细的听觉相似度报告"""
        scores = self.evaluate(generated_midi)
        gen_notes = self._extract_notes(generated_midi)

        report = []
        report.append("听觉相似度评估报告")
        report.append("=" * 60)
        report.append(f"参考音符数: {scores['ref_note_count']}")
        report.append(f"生成音符数: {scores['note_count']}")
        report.append(f"音符数比例: {scores['note_count']/scores['ref_note_count']*100:.1f}%")
        report.append("")

        report.append("各项指标评分:")
        report.append(f"  综合评分: {scores['overall_score']:.3f}")
        report.append(f"  旋律相似度: {scores['melody_similarity']:.3f} (权重: 40%)")
        report.append(f"  节奏相似度: {scores['rhythm_similarity']:.3f} (权重: 30%)")
        report.append(f"  音高覆盖率: {scores['pitch_coverage']:.3f} (权重: 20%)")
        report.append(f"  密度相关性: {scores['density_correlation']:.3f} (权重: 10%)")
        report.append("")

        # 时长分析
        if gen_notes:
            gen_durations = [n.end - n.start for n in gen_notes]
            gen_mean = np.mean(gen_durations)
            gen_short_ratio = scores['short_note_ratio']

            report.append("时长分析:")
            report.append(f"  参考平均时长: {self.ref_mean_duration:.3f}s")
            report.append(f"  生成平均时长: {gen_mean:.3f}s")
            report.append(f"  时长相似度: {scores['duration_similarity']:.3f}")
            report.append(f"  参考短音符比例: {self.ref_short_ratio:.3f}")
            report.append(f"  生成短音符比例: {gen_short_ratio:.3f}")

        return "\n".join(report)


# 便捷函数
def evaluate_auditory_similarity(ref_midi_path: str, gen_midi_path: str) -> Dict[str, Any]:
    """
    便捷函数：评估两个MIDI文件的听觉相似度

    Args:
        ref_midi_path: 参考MIDI文件路径
        gen_midi_path: 生成MIDI文件路径

    Returns:
        评估结果字典
    """
    ref_midi = pretty_midi.PrettyMIDI(ref_midi_path)
    gen_midi = pretty_midi.PrettyMIDI(gen_midi_path)

    evaluator = AuditorySimilarityEvaluator(ref_midi)
    scores = evaluator.evaluate(gen_midi)

    return {
        'scores': scores,
        'report': evaluator.generate_report(gen_midi)
    }


if __name__ == "__main__":
    # 简单测试
    import sys
    from pathlib import Path

    if len(sys.argv) >= 3:
        ref_path = Path(sys.argv[1])
        gen_path = Path(sys.argv[2])

        if ref_path.exists() and gen_path.exists():
            result = evaluate_auditory_similarity(str(ref_path), str(gen_path))
            print(result['report'])
        else:
            print(f"文件不存在: {ref_path if not ref_path.exists() else gen_path}")
    else:
        print("使用方法: python auditory_similarity_evaluator.py <参考MIDI> <生成MIDI>")
        print("示例: python auditory_similarity_evaluator.py data/test2.mid results/v1.0_original.mid")