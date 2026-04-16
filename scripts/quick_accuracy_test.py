#!/usr/bin/env python3
"""
快速准确度测试 - 验证系统改进效果
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_feedback_enhanced import EnhancedParameterSet, EnhancedFeedbackSystem
from midi_evaluator import ScoreEvaluator

def test_baseline_accuracy():
    """测试基准准确度（使用默认参数）"""
    print("=" * 60)
    print("基准准确度测试")
    print("=" * 60)

    # 加载测试文件
    audio_path = Path("test2.mp3")
    midi_path = Path("test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("错误：测试文件未找到")
        return None

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    print(f"音频: {audio_path.name}")
    print(f"参考MIDI: {midi_path.name} (138个音符)")
    print()

    # 测试1: 默认参数
    print("1. 默认参数生成器")
    generator = EnhancedMIDIGenerator()
    try:
        midi = generator.process_audio(audio_bytes)
        if midi and midi.instruments and midi.instruments[0].notes:
            midi.write("test_baseline.mid")
            evaluator = ScoreEvaluator(midi_path, Path("test_baseline.mid"))
            accuracy = evaluator.note_accuracy()
            print(f"   生成音符: {len(midi.instruments[0].notes)}")
            print(f"   F1分数: {accuracy['f1']:.3f}")
            print(f"   精确率: {accuracy['precision']:.3f}")
            print(f"   召回率: {accuracy['recall']:.3f}")
            baseline_f1 = accuracy['f1']
        else:
            print("   错误：未生成音符")
            baseline_f1 = 0
    except Exception as e:
        print(f"   错误：{e}")
        baseline_f1 = 0

    # 测试2: 音频感知参数（基于诊断结果）
    print("\n2. 音频感知参数")
    informed_params = EnhancedParameterSet(
        sr=44100,
        hop_length=512,
        min_freq=209,  # 基于音频诊断
        max_freq=941,  # 基于音频诊断
        pitch_method='pyin',
        voicing_threshold=0.6,
        confidence_weight=0.7,
        min_note_duration=0.2,
        max_gap=0.1,
        onset_threshold=0.6
    )
    informed_generator = EnhancedMIDIGenerator(informed_params.to_generator_config())
    try:
        midi = informed_generator.process_audio(audio_bytes)
        if midi and midi.instruments and midi.instruments[0].notes:
            midi.write("test_informed.mid")
            evaluator = ScoreEvaluator(midi_path, Path("test_informed.mid"))
            accuracy = evaluator.note_accuracy()
            print(f"   生成音符: {len(midi.instruments[0].notes)}")
            print(f"   F1分数: {accuracy['f1']:.3f}")
            print(f"   精确率: {accuracy['precision']:.3f}")
            print(f"   召回率: {accuracy['recall']:.3f}")
            informed_f1 = accuracy['f1']
        else:
            print("   错误：未生成音符")
            informed_f1 = 0
    except Exception as e:
        print(f"   错误：{e}")
        informed_f1 = 0

    # 测试3: 快速优化（3次迭代）
    print("\n3. 快速优化（3次迭代）")
    try:
        system = EnhancedFeedbackSystem()
        best_params = system.optimize_for_audio(
            audio_bytes=audio_bytes,
            reference_midi_path=midi_path,
            n_iterations=3,
            strategy="adaptive"
        )

        optimized_generator = system.create_optimized_generator()
        midi = optimized_generator.process_audio(audio_bytes)
        if midi and midi.instruments and midi.instruments[0].notes:
            midi.write("test_optimized.mid")
            evaluator = ScoreEvaluator(midi_path, Path("test_optimized.mid"))
            accuracy = evaluator.note_accuracy()
            print(f"   生成音符: {len(midi.instruments[0].notes)}")
            print(f"   F1分数: {accuracy['f1']:.3f}")
            print(f"   精确率: {accuracy['precision']:.3f}")
            print(f"   召回率: {accuracy['recall']:.3f}")
            print(f"   最佳分数: {system.best_score:.4f}")
            optimized_f1 = accuracy['f1']
        else:
            print("   错误：未生成音符")
            optimized_f1 = 0
    except Exception as e:
        print(f"   错误：{e}")
        import traceback
        traceback.print_exc()
        optimized_f1 = 0

    # 结果汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    results = {
        'baseline_f1': baseline_f1,
        'informed_f1': informed_f1,
        'optimized_f1': optimized_f1 if 'optimized_f1' in locals() else 0,
        'improvement_baseline_to_informed': informed_f1 - baseline_f1 if informed_f1 and baseline_f1 else 0,
        'improvement_informed_to_optimized': (optimized_f1 - informed_f1) if 'optimized_f1' in locals() and informed_f1 else 0,
    }

    print(f"基准F1:        {results['baseline_f1']:.3f}")
    print(f"音频感知F1:    {results['informed_f1']:.3f} ({results['improvement_baseline_to_informed']:+.3f})")
    if 'optimized_f1' in locals():
        print(f"优化后F1:      {results['optimized_f1']:.3f} ({results['improvement_informed_to_optimized']:+.3f})")

    # 保存结果
    with open("quick_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到 quick_test_results.json")
    print(f"生成的MIDI文件：")
    print(f"  test_baseline.mid   - 默认参数")
    print(f"  test_informed.mid   - 音频感知参数")
    print(f"  test_optimized.mid  - 优化后参数")

    return results

if __name__ == "__main__":
    test_baseline_accuracy()