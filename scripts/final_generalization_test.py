#!/usr/bin/env python3
"""
最终泛化测试：在test3.mp3和test5.mp3上测试v1.0, v2.0, v3.0版本
"""
import sys
import time
import json
from pathlib import Path
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_midi_generator import SimpleMIDIGenerator
from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi
import librosa

# 导入v3.0算法
sys.path.insert(0, str(Path(__file__).parent))
try:
    from generate_v3_short_notes import multi_scale_pitch_detection, create_midi_with_short_note_support
    V3_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入v3.0算法: {e}")
    V3_AVAILABLE = False

# 版本配置
VERSION_CONFIGS = {
    'v1.0_original': {
        'sr': 44100,
        'hop_length': 512,
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.1,
        'max_gap': 0.05,
        'voicing_threshold': 0.6,
        'harmonic_margin': 3,
    },
    'v2.0_basic_optimized': {
        'sr': 44100,
        'hop_length': 256,
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.05,
        'max_gap': 0.03,
        'voicing_threshold': 0.4,
        'harmonic_margin': 5,
    },
    'v3.0_short_note_improved': {
        'sr': 44100,
        'hop_length': 256,
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.05,
        'max_gap': 0.03,
        'voicing_threshold': 0.4,
        'harmonic_margin': 6,
        'short_note_threshold': 0.1,
        'target_frequencies': [440, 659],
    }
}

# 测试文件列表（排除test2.mp3，因为已经测试过）
TEST_FILES = [
    ("test3.mp3", "test3.mid"),
    ("test5.mp3", "test5.mid"),
]

def generate_v1_v2(audio_path, config):
    """生成v1.0或v2.0 MIDI"""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    generator = SimpleMIDIGenerator(config)
    return generator.process_audio(audio_bytes)

def generate_v3(audio_path, config):
    """生成v3.0 MIDI"""
    y, sr = librosa.load(audio_path, sr=config['sr'], mono=True)
    y = librosa.effects.preemphasis(y, coef=0.97)
    y = librosa.util.normalize(y, axis=0)
    f0, voiced_flag = multi_scale_pitch_detection(y, sr, config)
    return create_midi_with_short_note_support(f0, voiced_flag, sr, config)

def evaluate(ref_path, gen_path):
    """评估听觉相似度"""
    try:
        result = evaluate_auditory_similarity(str(ref_path), str(gen_path))
        return result['scores']
    except Exception as e:
        print(f"评估失败: {e}")
        return {'overall_score': 0.0}

def main():
    print("最终泛化测试")
    print("=" * 60)

    data_dir = Path("data")
    output_dir = Path("results/generalization_final")
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    total_start = time.time()

    for audio_name, ref_name in TEST_FILES:
        audio_path = data_dir / audio_name
        ref_path = data_dir / ref_name

        if not audio_path.exists() or not ref_path.exists():
            print(f"跳过 {audio_name}: 文件不存在")
            continue

        print(f"\n处理 {audio_name}...")
        file_results = {}

        for version, config in VERSION_CONFIGS.items():
            if version == 'v3.0_short_note_improved' and not V3_AVAILABLE:
                print(f"  跳过 {version} (不可用)")
                continue

            print(f"  生成 {version}...")
            start_time = time.time()

            try:
                # 生成MIDI
                if version == 'v3.0_short_note_improved':
                    midi = generate_v3(audio_path, config)
                else:
                    midi = generate_v1_v2(audio_path, config)

                gen_time = time.time() - start_time

                # 保存MIDI
                midi_filename = f"{audio_path.stem}_{version.replace('.', '_')}.mid"
                midi_path = output_dir / midi_filename
                midi.write(str(midi_path))

                # 评估
                scores = evaluate(ref_path, midi_path)
                note_count = len(midi.instruments[0].notes) if midi.instruments else 0

                file_results[version] = {
                    'success': True,
                    'midi_file': str(midi_path),
                    'generation_time_seconds': gen_time,
                    'note_count': note_count,
                    'overall_score': scores['overall_score'],
                    'melody_similarity': scores.get('melody_similarity', 0),
                    'rhythm_similarity': scores.get('rhythm_similarity', 0),
                }

                print(f"    成功！音符数: {note_count}, 评分: {scores['overall_score']:.3f}, 耗时: {gen_time:.1f}s")

            except Exception as e:
                print(f"    失败: {e}")
                file_results[version] = {
                    'success': False,
                    'error': str(e),
                    'overall_score': 0.0,
                }

        results[audio_name] = file_results

    total_time = time.time() - total_start

    # 汇总分析
    print(f"\n{'='*60}")
    print("汇总结果:")

    summary = {}
    for version in VERSION_CONFIGS:
        scores = []
        notes = []
        successes = 0

        for audio_name in results:
            if version in results[audio_name] and results[audio_name][version].get('success'):
                scores.append(results[audio_name][version]['overall_score'])
                notes.append(results[audio_name][version]['note_count'])
                successes += 1

        if successes > 0:
            summary[version] = {
                'success_rate': successes / len(results),
                'avg_score': np.mean(scores),
                'std_score': np.std(scores),
                'avg_notes': np.mean(notes),
                'total_tests': len(results),
                'successful_tests': successes,
            }

            print(f"\n  {version}:")
            print(f"    成功率: {summary[version]['success_rate']:.1%} ({successes}/{len(results)})")
            print(f"    平均评分: {summary[version]['avg_score']:.3f} (±{summary[version]['std_score']:.3f})")
            print(f"    平均音符数: {summary[version]['avg_notes']:.1f}")
        else:
            summary[version] = {'success_rate': 0}
            print(f"\n  {version}: 无成功测试")

    # 保存结果
    output_data = {
        'test_files': TEST_FILES,
        'versions': list(VERSION_CONFIGS.keys()),
        'results': results,
        'summary': summary,
        'total_time_seconds': total_time,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_path = output_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"测试完成！总耗时: {total_time:.1f}秒")
    print(f"结果保存至: {results_path}")

    # 推荐最佳版本
    valid_versions = {v: s for v, s in summary.items() if s.get('success_rate', 0) > 0}
    if valid_versions:
        best_version = max(valid_versions.items(), key=lambda x: x[1].get('avg_score', 0))
        print(f"\n推荐版本: {best_version[0]}")
        print(f"  平均评分: {best_version[1]['avg_score']:.3f}")
        print(f"  成功率: {best_version[1]['success_rate']:.1%}")
    else:
        print("\n警告：无成功版本")

if __name__ == "__main__":
    main()