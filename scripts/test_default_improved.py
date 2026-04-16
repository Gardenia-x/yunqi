#!/usr/bin/env python3
"""
测试改进后的默认参数
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_evaluator import ScoreEvaluator

def test():
    print("测试改进后的默认参数")
    print("=" * 50)

    audio_path = Path("../data/test2.mp3")
    midi_path = Path("../data/test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("文件未找到")
        return

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # 使用改进后的默认生成器
    generator = EnhancedMIDIGenerator()
    print(f"使用参数:")
    print(f"  min_note_duration: {generator.config['min_note_duration']}")
    print(f"  max_gap: {generator.config['max_gap']}")
    print(f"  voicing_threshold: {generator.config['voicing_threshold']}")
    print(f"  min_freq: {generator.config['min_freq']}")
    print(f"  max_freq: {generator.config['max_freq']}")
    print()

    try:
        midi = generator.process_audio(audio_bytes)
        if midi and midi.instruments and midi.instruments[0].notes:
            note_count = len(midi.instruments[0].notes)
            output_path = Path("../results/test_improved_default.mid")
            midi.write(str(output_path))

            evaluator = ScoreEvaluator(midi_path, output_path)
            accuracy = evaluator.note_accuracy()

            print(f"结果:")
            print(f"  生成音符数: {note_count} (参考: 138)")
            print(f"  F1分数: {accuracy['f1']:.3f}")
            print(f"  精确率: {accuracy['precision']:.3f}")
            print(f"  召回率: {accuracy['recall']:.3f}")

            # 与之前的结果比较（之前F1≈0.235）
            previous_f1 = 0.235
            improvement = accuracy['f1'] - previous_f1
            print(f"\n改进对比:")
            print(f"  之前F1: {previous_f1:.3f}")
            print(f"  当前F1: {accuracy['f1']:.3f}")
            print(f"  改进: {improvement:+.3f} ({improvement/previous_f1*100:.1f}%)")
        else:
            print("错误：未生成音符")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()