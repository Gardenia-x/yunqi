#!/usr/bin/env python3
"""
生成v3.0_algorithm_improved.mid
基于算法改进的版本，专注提高听觉流畅度
"""
import sys
import time
from pathlib import Path
import json

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_midi_generator import EnhancedMIDIGenerator
from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi


def main():
    print("生成v3.0_algorithm_improved.mid")
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

    # 加载v1.0和v2.0作为对比
    v1_path = output_dir / "v1.0_original.mid"
    v2_path = output_dir / "v2.0_basic_optimized.mid"

    # 加载v2.0优化参数
    v2_params = None
    v2_metadata_path = output_dir / "v2.0_metadata.json"
    if v2_metadata_path.exists():
        with open(v2_metadata_path, 'r') as f:
            v2_metadata = json.load(f)
            v2_params = v2_metadata['best_params']
            print(f"使用v2.0优化参数:")
            for key, value in v2_params.items():
                print(f"  {key}: {value}")
    else:
        print("警告: v2.0_metadata.json 不存在，使用增强算法默认参数")
        v2_params = {}

    # 增强算法配置（基于v2.0参数 + 增强参数）
    enhanced_config = {
        # 基于v2.0的优化参数
        'min_duration': 0.05,
        'max_gap': 0.03,
        'voicing_threshold': 0.4,
        'harmonic_margin': 5,
        'hop_length': 256,
        'min_freq': 80,
        'max_freq': 1200,

        # 增强参数（针对流畅度优化）
        'pitch_smooth_window': 7,       # 音高平滑窗口
        'note_merge_threshold': 2,      # 音符合并阈值（半音）
        'dynamic_onset_detection': True, # 动态onset检测
        'energy_threshold_ratio': 0.3,  # 能量阈值
        'min_silence_duration': 0.02,   # 最小静音时长
    }

    # 如果v2_params存在，用它覆盖部分配置
    if v2_params:
        for key in ['min_duration', 'max_gap', 'voicing_threshold',
                   'harmonic_margin', 'hop_length', 'min_freq', 'max_freq']:
            if key in v2_params:
                enhanced_config[key] = v2_params[key]

    print(f"\n增强算法配置:")
    for key, value in enhanced_config.items():
        print(f"  {key}: {value}")

    print(f"\n开始生成v3.0版本...")
    print("算法改进要点:")
    print("  1. 音高平滑处理，减少音高跳变")
    print("  2. 智能音符合并，相似音高的相邻音符合并")
    print("  3. 动态onset检测，基于能量变化确定音符开始")
    print("  4. 能量边界调整，使音符边界更自然")

    start_time = time.time()

    try:
        # 加载音频
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        # 使用增强算法生成MIDI
        generator = EnhancedMIDIGenerator(enhanced_config)
        midi = generator.process_audio(audio_bytes)

        elapsed = time.time() - start_time

        # 保存v3.0版本
        v3_path = output_dir / "v3.0_algorithm_improved.mid"
        midi.write(str(v3_path))

        # 分析v3.0
        v3_midi = pretty_midi.PrettyMIDI(str(v3_path))
        v3_notes = len(v3_midi.instruments[0].notes) if v3_midi.instruments else 0

        print(f"\n生成完成! 耗时: {elapsed:.1f} 秒")
        print(f"v3.0信息:")
        print(f"  音符数: {v3_notes}")

        # 对比之前的版本
        versions_to_compare = []
        if v1_path.exists():
            v1_midi = pretty_midi.PrettyMIDI(str(v1_path))
            v1_notes = len(v1_midi.instruments[0].notes) if v1_midi.instruments else 0
            versions_to_compare.append(('v1.0', v1_path, v1_notes))

        if v2_path.exists():
            v2_midi = pretty_midi.PrettyMIDI(str(v2_path))
            v2_notes = len(v2_midi.instruments[0].notes) if v2_midi.instruments else 0
            versions_to_compare.append(('v2.0', v2_path, v2_notes))

        versions_to_compare.append(('v3.0', v3_path, v3_notes))

        print(f"\n版本对比 (音符数):")
        for name, _, note_count in versions_to_compare:
            print(f"  {name}: {note_count} 音符")

        # 听觉相似度评估
        print(f"\n听觉相似度对比:")
        scores = {}
        for name, path, _ in versions_to_compare:
            if path.exists():
                try:
                    result = evaluate_auditory_similarity(str(ref_path), str(path))
                    scores[name] = result['scores']['overall_score']
                    print(f"  {name}综合评分: {scores[name]:.3f}")
                except Exception as e:
                    print(f"  {name}评估失败: {e}")
                    scores[name] = 0.0

        # 计算改进百分比
        if 'v1.0' in scores and 'v3.0' in scores:
            improvement_v1 = (scores['v3.0'] - scores['v1.0']) / scores['v1.0'] * 100
            print(f"  v3.0相对于v1.0改进: {improvement_v1:+.1f}%")

        if 'v2.0' in scores and 'v3.0' in scores:
            improvement_v2 = (scores['v3.0'] - scores['v2.0']) / scores['v2.0'] * 100
            print(f"  v3.0相对于v2.0改进: {improvement_v2:+.1f}%")

        # 保存v3.0元数据
        metadata = {
            'version': 'v3.0_algorithm_improved',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'generation_time_seconds': elapsed,
            'config': enhanced_config,
            'note_count': v3_notes,
            'auditory_scores': scores,
            'midi_file': str(v3_path),
            'algorithm_improvements': [
                '音高平滑处理 (pitch_smooth_window=7)',
                '智能音符合并 (note_merge_threshold=2半音)',
                '动态onset检测 (基于能量变化)',
                '能量边界调整 (使边界更自然)'
            ]
        }

        metadata_path = output_dir / "v3.0_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n元数据已保存: {metadata_path}")

        # 打印算法改进预期效果
        print(f"\n算法改进预期效果:")
        print("  1. 提高听觉流畅度，减少'一卡一卡'的感觉")
        print("  2. 改善旋律连续性，减少音高跳变")
        print("  3. 更自然的音符边界，避免生硬分割")
        print("  4. 更好的短音符检测和连接")
        print(f"\n请试听: {v3_path}")
        print("对比v1.0_original.mid和v2.0_basic_optimized.mid，关注:")
        print("  - 整体流畅度是否提高")
        print("  - 旋律线是否更连贯")
        print("  - 音符边界是否更自然")
        print("  - 整体听觉体验是否更好")

    except Exception as e:
        print(f"生成过程失败: {e}")
        import traceback
        traceback.print_exc()

        # 回退方案：使用简单生成器
        print("\n尝试回退方案：使用简单生成器...")
        try:
            from simple_midi_generator import SimpleMIDIGenerator

            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            # 使用v2.0参数
            fallback_params = v2_params if v2_params else {
                'min_duration': 0.05,
                'max_gap': 0.03,
                'voicing_threshold': 0.4,
                'harmonic_margin': 5,
                'hop_length': 256,
            }

            generator = SimpleMIDIGenerator(fallback_params)
            midi = generator.process_audio(audio_bytes)

            v3_path = output_dir / "v3.0_algorithm_improved.mid"
            midi.write(str(v3_path))

            note_count = len(midi.instruments[0].notes) if midi.instruments else 0
            print(f"回退方案生成v3.0: {note_count} 音符")
            print(f"保存到: {v3_path}")

        except Exception as e2:
            print(f"回退方案也失败: {e2}")


if __name__ == "__main__":
    main()