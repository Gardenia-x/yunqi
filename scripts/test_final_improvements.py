#!/usr/bin/env python3
"""
测试所有改进后的最终算法性能
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_feedback_enhanced import EnhancedParameterSet
from midi_evaluator import ScoreEvaluator

def test_improvements():
    print("最终改进测试")
    print("=" * 60)

    # 加载文件
    audio_path = Path("../data/test2.mp3")
    midi_path = Path("../data/test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("文件未找到")
        return

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # 加载最佳参数
    with open("../results/best_parameters.json", 'r') as f:
        best = json.load(f)

    base_params = best['parameters']
    original_f1 = best['results']['f1']

    # 使用相同的参数，但应用我们的代码改进
    params = EnhancedParameterSet(**base_params)

    print(f"原始最佳F1: {original_f1:.3f}")
    print(f"测试参数: {params.to_dict()['pitch_method']}, hop_length={params.to_dict()['hop_length']}, "
          f"min_note_duration={params.to_dict()['min_note_duration']}")

    # 测试改进后的算法
    try:
        generator = EnhancedMIDIGenerator(params.to_generator_config())
        midi = generator.process_audio(audio_bytes)

        if midi and midi.instruments and midi.instruments[0].notes:
            note_count = len(midi.instruments[0].notes)
            output_path = Path("../results/test_final_improved.mid")
            midi.write(str(output_path))

            evaluator = ScoreEvaluator(midi_path, output_path)
            accuracy = evaluator.note_accuracy()

            print(f"\n结果:")
            print(f"  生成音符数: {note_count}")
            print(f"  F1分数: {accuracy['f1']:.3f}")
            print(f"  精确率: {accuracy['precision']:.3f}")
            print(f"  召回率: {accuracy['recall']:.3f}")

            improvement = accuracy['f1'] - original_f1
            print(f"\n改进对比:")
            print(f"  原始F1: {original_f1:.3f}")
            print(f"  改进后F1: {accuracy['f1']:.3f}")
            print(f"  提升: {improvement:+.3f} ({improvement/original_f1*100:.1f}%)")

            # 分析短音符检测改进
            import pretty_midi
            ref_midi = pretty_midi.PrettyMIDI(str(midi_path))
            gen_midi = pretty_midi.PrettyMIDI(str(output_path))

            ref_notes = ref_midi.instruments[0].notes if ref_midi.instruments else []
            gen_notes = gen_midi.instruments[0].notes if gen_midi.instruments else []

            ref_durations = [n.end - n.start for n in ref_notes]
            gen_durations = [n.end - n.start for n in gen_notes]

            short_threshold = 0.3
            ref_short = len([d for d in ref_durations if d < short_threshold])
            gen_short = len([d for d in gen_durations if d < short_threshold])

            print(f"\n短音符检测 (<{short_threshold}s):")
            print(f"  参考短音符: {ref_short}/{len(ref_notes)} ({ref_short/len(ref_notes)*100:.1f}%)")
            print(f"  生成短音符: {gen_short}/{len(gen_notes)} ({gen_short/len(gen_notes)*100:.1f}%)")
            if ref_short > 0:
                print(f"  短音符召回率: {gen_short/ref_short*100:.1f}%")

            # 保存结果
            result = {
                'original_f1': original_f1,
                'improved_f1': accuracy['f1'],
                'improvement': improvement,
                'improvement_percent': improvement/original_f1*100,
                'precision': accuracy['precision'],
                'recall': accuracy['recall'],
                'note_count': note_count,
                'short_note_recall': gen_short/ref_short*100 if ref_short > 0 else 0,
                'parameters': params.to_dict()
            }

            with open("../results/final_improvement_results.json", 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n结果已保存到 ../results/final_improvement_results.json")

            return accuracy['f1']
        else:
            print("错误：未生成音符")
            return None

    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_improvements()