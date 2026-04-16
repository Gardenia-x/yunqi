#!/usr/bin/env python3
"""
生成v2.0_basic_optimized.mid
基于参数优化的版本，使用听觉相似度作为优化目标
"""
import sys
import time
from pathlib import Path
import json

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_auditory_optimizer import optimize_midi_generation
import pretty_midi


def main():
    print("生成v2.0_basic_optimized.mid")
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

    # 加载v1.0作为基准
    v1_path = output_dir / "v1.0_original.mid"
    if v1_path.exists():
        v1_midi = pretty_midi.PrettyMIDI(str(v1_path))
        v1_notes = len(v1_midi.instruments[0].notes) if v1_midi.instruments else 0
        print(f"v1.0基准: {v1_notes} 音符")
    else:
        print("警告: v1.0_original.mid 不存在")
        v1_notes = 0

    print("\n开始参数优化...")
    print("策略: hybrid (网格+随机搜索)")
    print("最大迭代: 30")
    print("超时: 180秒")

    start_time = time.time()

    try:
        # 执行优化
        result = optimize_midi_generation(
            audio_path=str(audio_path),
            reference_midi_path=str(ref_path),
            output_dir=str(output_dir),
            strategy='hybrid'
        )

        elapsed = time.time() - start_time

        print(f"\n优化完成! 耗时: {elapsed:.1f} 秒")
        print(f"最佳综合评分: {result['scores']['overall_score']:.3f}")
        print(f"最佳参数:")
        for key, value in result['params'].items():
            print(f"  {key}: {value}")

        # 重命名优化结果为v2.0
        optimized_midi_path = Path(result['midi_file'])
        v2_path = output_dir / "v2.0_basic_optimized.mid"

        if optimized_midi_path.exists():
            # 复制文件
            import shutil
            shutil.copy2(optimized_midi_path, v2_path)
            print(f"\n已保存v2.0版本: {v2_path}")

            # 加载并分析v2.0
            v2_midi = pretty_midi.PrettyMIDI(str(v2_path))
            v2_notes = len(v2_midi.instruments[0].notes) if v2_midi.instruments else 0

            print(f"v2.0信息:")
            print(f"  音符数: {v2_notes}")
            if v1_notes > 0:
                diff = (v2_notes - v1_notes) / v1_notes * 100
                print(f"  相对于v1.0变化: {diff:+.1f}%")

            # 生成比较报告
            from auditory_similarity_evaluator import evaluate_auditory_similarity

            print(f"\n听觉相似度对比:")
            v1_scores = evaluate_auditory_similarity(str(ref_path), str(v1_path)) if v1_path.exists() else None
            v2_scores = evaluate_auditory_similarity(str(ref_path), str(v2_path))

            if v1_scores:
                v1_overall = v1_scores['scores']['overall_score']
                v2_overall = v2_scores['scores']['overall_score']
                improvement = (v2_overall - v1_overall) / v1_overall * 100
                print(f"  v1.0综合评分: {v1_overall:.3f}")
                print(f"  v2.0综合评分: {v2_overall:.3f}")
                print(f"  改进: {improvement:+.1f}%")

            # 保存优化结果元数据
            metadata = {
                'version': 'v2.0_basic_optimized',
                'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'optimization_time_seconds': elapsed,
                'optimization_strategy': 'hybrid',
                'best_params': result['params'],
                'best_scores': result['scores'],
                'note_count': v2_notes,
                'v1_note_count': v1_notes,
                'midi_file': str(v2_path),
            }

            metadata_path = output_dir / "v2.0_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n元数据已保存: {metadata_path}")

            # 打印关键改进
            print(f"\n预期听觉改进:")
            print("  1. 基于听觉相似度优化的参数")
            print("  2. 平衡旋律保持和节奏流畅度")
            print("  3. 改进短音符检测和音符合并")
            print(f"\n请试听: {v2_path}")
            print("对比v1.0_original.mid，关注:")
            print("  - 整体相似度是否提高")
            print("  - 旋律线是否更清晰")
            print("  - 节奏是否更流畅")

        else:
            print(f"错误：优化MIDI文件不存在: {optimized_midi_path}")

    except Exception as e:
        print(f"优化过程失败: {e}")
        import traceback
        traceback.print_exc()

        # 回退方案：使用默认参数的变体
        print("\n尝试回退方案：使用改进的默认参数...")
        try:
            from simple_midi_generator import SimpleMIDIGenerator

            # 改进的默认参数（基于经验）
            improved_params = {
                'min_duration': 0.08,      # 稍微降低以检测更多短音符
                'max_gap': 0.04,          # 稍微降低以减少间隙
                'voicing_threshold': 0.5,  # 降低阈值以检测更多音符
                'harmonic_margin': 4,      # 增强谐波检测
                'hop_length': 384,         # 提高时间分辨率
            }

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            generator = SimpleMIDIGenerator(improved_params)
            midi = generator.process_audio(audio_bytes)

            v2_path = output_dir / "v2.0_basic_optimized.mid"
            midi.write(str(v2_path))

            note_count = len(midi.instruments[0].notes) if midi.instruments else 0
            print(f"回退方案生成v2.0: {note_count} 音符")
            print(f"参数: {improved_params}")
            print(f"保存到: {v2_path}")

        except Exception as e2:
            print(f"回退方案也失败: {e2}")


if __name__ == "__main__":
    main()