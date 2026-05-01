#!/usr/bin/env python3
"""
分析各版本听觉相似度评分细节
"""
import sys
from pathlib import Path
import json

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi


def analyze_version(name: str, midi_path: str, ref_path: str):
    """分析单个版本的详细评分"""
    if not Path(midi_path).exists():
        print(f"文件不存在: {midi_path}")
        return None

    try:
        result = evaluate_auditory_similarity(ref_path, midi_path)
        scores = result['scores']

        print(f"\n{name} 详细评分:")
        print(f"  综合评分: {scores['overall_score']:.3f}")
        print(f"  旋律相似度: {scores['melody_similarity']:.3f} (权重: 40%)")
        print(f"  节奏相似度: {scores['rhythm_similarity']:.3f} (权重: 30%)")
        print(f"  音高覆盖率: {scores['pitch_coverage']:.3f} (权重: 20%)")
        print(f"  密度相关性: {scores['density_correlation']:.3f} (权重: 10%)")
        print(f"  时长相似度: {scores['duration_similarity']:.3f}")
        print(f"  短音符比例: {scores['short_note_ratio']:.3f}")
        print(f"  音符数: {scores['note_count']} (参考: {scores['ref_note_count']})")

        return scores
    except Exception as e:
        print(f"分析{name}失败: {e}")
        return None


def main():
    print("各版本听觉相似度评分详细分析")
    print("=" * 60)

    ref_path = "data/test2.mid"
    versions = [
        ("v1.0_original", "results/v1.0_original.mid"),
        ("v2.0_basic_optimized", "results/v2.0_basic_optimized.mid"),
        ("v3.0_short_note_improved", "results/v3.0_short_note_improved.mid"),
    ]

    if not Path(ref_path).exists():
        print(f"参考文件不存在: {ref_path}")
        return

    all_scores = {}
    for name, path in versions:
        scores = analyze_version(name, path, ref_path)
        if scores:
            all_scores[name] = scores

    # 对比分析
    if len(all_scores) >= 2:
        print(f"\n{'='*60}")
        print("版本对比分析:")

        # 找出最佳综合评分
        best_overall = max(all_scores.items(), key=lambda x: x[1]['overall_score'])
        print(f"  最佳综合评分: {best_overall[0]} ({best_overall[1]['overall_score']:.3f})")

        # 对比各子项
        print(f"\n  各子项对比:")
        for metric in ['melody_similarity', 'rhythm_similarity', 'pitch_coverage',
                      'density_correlation', 'duration_similarity']:
            if metric in all_scores['v1.0_original']:
                v1_val = all_scores['v1.0_original'][metric]
                best_val = max(scores[metric] for scores in all_scores.values())
                best_version = [name for name, scores in all_scores.items()
                               if scores[metric] == best_val][0]
                print(f"    {metric}:")
                print(f"      v1.0: {v1_val:.3f}, 最佳: {best_version} ({best_val:.3f})")

        # 音符数对比
        print(f"\n  音符数对比:")
        for name, scores in all_scores.items():
            ratio = scores['note_count'] / scores['ref_note_count'] * 100
            print(f"    {name}: {scores['note_count']} ({ratio:.1f}% of ref)")

        # 短音符比例对比
        print(f"\n  短音符比例对比 (<0.3秒):")
        for name, scores in all_scores.items():
            print(f"    {name}: {scores['short_note_ratio']:.3f}")


if __name__ == "__main__":
    main()