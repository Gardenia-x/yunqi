#!/usr/bin/env python3
"""
音高检测参数优化测试
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midi_generate_enhanced import EnhancedMIDIGenerator, CREPE_AVAILABLE
from midi_feedback_enhanced import EnhancedParameterSet
from midi_evaluator import ScoreEvaluator

def test_pitch_parameters():
    print("音高检测参数优化测试")
    print("=" * 60)

    # 加载文件
    audio_path = Path("../data/test2.mp3")
    midi_path = Path("../data/test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("文件未找到")
        return

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # 基于最佳参数创建基础配置
    with open("../results/best_parameters.json", 'r') as f:
        best = json.load(f)

    base_params = best['parameters']

    # 定义多个音高检测参数组合
    param_sets = {
        # 组合1: 增加PYIN精度
        "pyin_high_precision": EnhancedParameterSet(
            sr=base_params['sr'],
            hop_length=base_params['hop_length'],
            frame_length=base_params['frame_length'],
            min_freq=200,  # 降低以包含C4
            max_freq=1300,  # 提高以更好检测高音
            pitch_method='pyin',
            n_thresholds=300,  # 增加阈值数量
            resolution=0.05,   # 提高分辨率
            voicing_threshold=0.4,  # 降低阈值以检测更多音符
            confidence_weight=0.7,
            use_harmonic_enhancement=True,
            harmonic_margin=4,  # 增加谐波margin
            median_filter_size=base_params['median_filter_size'],
            smoothing_window=base_params['smoothing_window'],
            interpolate_missing=True,
            onset_threshold=0.5,
            min_note_duration=0.08,  # 进一步降低以检测短音符
            max_gap=0.03,
            min_velocity=base_params['min_velocity'],
            max_velocity=base_params['max_velocity'],
            max_polyphony=base_params['max_polyphony'],
            polyphony_threshold=base_params['polyphony_threshold']
        ),
        # 组合2: 使用CREPE（如果可用）
        "crepe_based": EnhancedParameterSet(
            sr=base_params['sr'],
            hop_length=256,
            frame_length=base_params['frame_length'],
            min_freq=200,
            max_freq=1300,
            pitch_method='crepe' if CREPE_AVAILABLE else 'hybrid',
            n_thresholds=base_params['n_thresholds'],
            resolution=base_params['resolution'],
            voicing_threshold=0.35,  # CREPE置信度阈值可以更低
            confidence_weight=0.8,
            use_harmonic_enhancement=False,  # CREPE不需要谐波增强
            harmonic_margin=base_params['harmonic_margin'],
            median_filter_size=3,  # 更小的中值滤波
            smoothing_window=5,
            interpolate_missing=True,
            onset_threshold=0.45,
            min_note_duration=0.07,
            max_gap=0.025,
            min_velocity=base_params['min_velocity'],
            max_velocity=base_params['max_velocity'],
            max_polyphony=base_params['max_polyphony'],
            polyphony_threshold=base_params['polyphony_threshold']
        ),
        # 组合3: 优化谐波增强
        "harmonic_enhanced": EnhancedParameterSet(
            sr=base_params['sr'],
            hop_length=192,  # 更小的hop_length以提高时间分辨率
            frame_length=1024,
            min_freq=180,
            max_freq=1400,
            pitch_method='hybrid',
            n_thresholds=250,
            resolution=0.08,
            voicing_threshold=0.42,
            confidence_weight=0.75,
            use_harmonic_enhancement=True,
            harmonic_margin=5,  # 更大的谐波margin
            median_filter_size=base_params['median_filter_size'],
            smoothing_window=base_params['smoothing_window'],
            interpolate_missing=True,
            onset_threshold=0.48,
            min_note_duration=0.06,  # 更短的音符
            max_gap=0.02,
            min_velocity=base_params['min_velocity'],
            max_velocity=base_params['max_velocity'],
            max_polyphony=base_params['max_polyphony'],
            polyphony_threshold=base_params['polyphony_threshold']
        ),
        # 组合4: 原始最佳参数（作为基准）
        "original_best": EnhancedParameterSet(**base_params)
    }

    results = {}
    best_f1 = 0
    best_name = None

    for name, params in param_sets.items():
        print(f"\n测试组合: {name}")
        print("-" * 40)

        # 跳过CREPE如果不可用
        if 'crepe' in name and not CREPE_AVAILABLE:
            print("  CREPE不可用，跳过此组合")
            continue

        result = test_single_params(params, audio_bytes, midi_path, name)
        results[name] = result

        if 'f1' in result and result['f1'] > best_f1:
            best_f1 = result['f1']
            best_name = name

    # 输出结果
    print("\n" + "=" * 60)
    print("音高优化测试结果汇总")
    print("=" * 60)

    for name, result in results.items():
        if 'f1' in result:
            print(f"{name:25s} | 音符: {result.get('note_count', 0):3d} | "
                  f"F1: {result['f1']:.3f} | 精确率: {result.get('precision', 0):.3f} | "
                  f"召回率: {result.get('recall', 0):.3f}")

    if best_name:
        print(f"\n最佳组合: {best_name} (F1: {best_f1:.3f})")
        best_result = results[best_name]

        # 保存最佳参数
        with open("../results/best_pitch_parameters.json", 'w') as f:
            json.dump({
                'name': best_name,
                'parameters': param_sets[best_name].to_dict(),
                'results': best_result
            }, f, indent=2)
        print(f"最佳音高参数已保存到 ../results/best_pitch_parameters.json")

    # 保存所有结果
    with open("../results/pitch_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)

def test_single_params(params, audio_bytes, ref_midi_path, label):
    """测试单个参数集"""
    try:
        generator = EnhancedMIDIGenerator(params.to_generator_config())
        midi = generator.process_audio(audio_bytes)

        if midi and midi.instruments and midi.instruments[0].notes:
            note_count = len(midi.instruments[0].notes)
            output_path = Path(f"../results/test_pitch_{label}.mid")
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
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

if __name__ == "__main__":
    test_pitch_parameters()