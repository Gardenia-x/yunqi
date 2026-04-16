#!/usr/bin/env python3
"""
最终优化测试 - 测试多个针对性参数组合
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_feedback_enhanced import EnhancedParameterSet
from midi_evaluator import ScoreEvaluator

def test_parameter_sets():
    print("针对性参数优化测试")
    print("=" * 60)

    # 加载文件
    audio_path = Path("../data/test2.mp3")
    midi_path = Path("../data/test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("文件未找到")
        return

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # 定义多个参数组合
    param_sets = {
        # 组合1: 高敏感度（低阈值，短音符）
        "high_sensitivity": EnhancedParameterSet(
            sr=44100,
            hop_length=512,
            min_freq=200,
            max_freq=1000,
            pitch_method='hybrid',
            voicing_threshold=0.4,
            confidence_weight=0.6,
            min_note_duration=0.08,
            max_gap=0.03,
            onset_threshold=0.5
        ),
        # 组合2: 平衡型（中等阈值）
        "balanced": EnhancedParameterSet(
            sr=44100,
            hop_length=512,
            min_freq=209,
            max_freq=941,
            pitch_method='pyin',
            voicing_threshold=0.5,
            confidence_weight=0.7,
            min_note_duration=0.1,
            max_gap=0.05,
            onset_threshold=0.6
        ),
        # 组合3: 针对高音优化（调整频率范围）
        "high_pitch_focused": EnhancedParameterSet(
            sr=44100,
            hop_length=256,  # 更好的时间分辨率
            min_freq=350,    # 聚焦中高音
            max_freq=1200,
            pitch_method='hybrid',
            voicing_threshold=0.45,
            confidence_weight=0.65,
            min_note_duration=0.09,
            max_gap=0.04,
            onset_threshold=0.55
        ),
        # 组合4: 保守型（减少误检）
        "conservative": EnhancedParameterSet(
            sr=44100,
            hop_length=512,
            min_freq=209,
            max_freq=941,
            pitch_method='pyin',
            voicing_threshold=0.6,
            confidence_weight=0.8,
            min_note_duration=0.15,
            max_gap=0.08,
            onset_threshold=0.65
        ),
    }

    results = {}
    best_f1 = 0
    best_name = None

    for name, params in param_sets.items():
        print(f"\n测试组合: {name}")
        print("-" * 40)

        result = test_single_params(params, audio_bytes, midi_path, name)
        results[name] = result

        if 'f1' in result and result['f1'] > best_f1:
            best_f1 = result['f1']
            best_name = name

    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, result in results.items():
        if 'f1' in result:
            print(f"{name:20s} | 音符: {result.get('note_count', 0):3d} | "
                  f"F1: {result['f1']:.3f} | 精确率: {result.get('precision', 0):.3f} | "
                  f"召回率: {result.get('recall', 0):.3f}")

    if best_name:
        print(f"\n最佳组合: {best_name} (F1: {best_f1:.3f})")
        best_result = results[best_name]
        print(f"参数:")
        for key, value in param_sets[best_name].to_dict().items():
            if key not in ['sr', 'frame_length', 'n_thresholds', 'resolution',
                          'fill_na', 'harmonic_margin', 'median_filter_size',
                          'smoothing_window', 'interpolate_missing', 'min_velocity',
                          'max_velocity', 'max_polyphony', 'polyphony_threshold']:
                print(f"  {key}: {value}")

        # 保存最佳参数
        with open("best_parameters.json", 'w') as f:
            json.dump({
                'name': best_name,
                'parameters': param_sets[best_name].to_dict(),
                'results': best_result
            }, f, indent=2)
        print(f"\n最佳参数已保存到 best_parameters.json")

    # 保存所有结果
    with open("all_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)

def test_single_params(params, audio_bytes, ref_midi_path, label):
    """测试单个参数集"""
    try:
        generator = EnhancedMIDIGenerator(params.to_generator_config())
        midi = generator.process_audio(audio_bytes)

        if midi and midi.instruments and midi.instruments[0].notes:
            note_count = len(midi.instruments[0].notes)
            output_path = Path(f"test_{label}.mid")
            midi.write(str(output_path))

            evaluator = ScoreEvaluator(ref_midi_path, output_path)
            accuracy = evaluator.note_accuracy()

            result = {
                'f1': accuracy['f1'],
                'precision': accuracy['precision'],
                'recall': accuracy['recall'],
                'note_count': note_count,
                'params': params.to_dict()
            }

            print(f"  生成音符: {note_count}")
            print(f"  F1分数: {accuracy['f1']:.3f}")
            print(f"  精确率: {accuracy['precision']:.3f}")
            print(f"  召回率: {accuracy['recall']:.3f}")

            return result
        else:
            print(f"  错误：未生成音符")
            return {'error': 'no_notes_generated'}
    except Exception as e:
        print(f"  错误：{e}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_parameter_sets()