#!/usr/bin/env python3
"""
测试各版本在不同音频文件上的泛化能力
测试文件：test2.mp3, test3.mp3, test5.mp3
版本：v1.0_original, v2.0_basic_optimized, v3.0_short_note_improved
"""
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simple_midi_generator import SimpleMIDIGenerator
from auditory_similarity_evaluator import evaluate_auditory_similarity
import pretty_midi
import librosa

# 导入v3.0短音符改进算法
sys.path.insert(0, str(Path(__file__).parent))
try:
    from generate_v3_short_notes import multi_scale_pitch_detection, create_midi_with_short_note_support
    V3_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入v3.0算法，跳过v3.0测试: {e}")
    V3_AVAILABLE = False


def get_version_configs() -> Dict[str, Dict[str, Any]]:
    """获取各版本的配置参数"""
    configs = {
        # v1.0: 原始配置（基于test1_copy.py的硬编码参数）
        'v1.0_original': {
            'sr': 44100,
            'hop_length': 512,
            'min_freq': 80,
            'max_freq': 1200,
            'min_duration': 0.1,
            'max_gap': 0.05,
            'voicing_threshold': 0.6,
            'harmonic_margin': 3,
            'preemphasis_coef': 0.97,
            'n_thresholds': 200,
            'resolution': 0.1,
        },
        # v2.0: 基础优化配置（基于优化器结果）
        'v2.0_basic_optimized': {
            'sr': 44100,
            'hop_length': 256,
            'min_freq': 80,
            'max_freq': 1200,
            'min_duration': 0.05,
            'max_gap': 0.03,
            'voicing_threshold': 0.4,
            'harmonic_margin': 5,
            'preemphasis_coef': 0.97,
            'n_thresholds': 200,
            'resolution': 0.1,
        },
        # v3.0: 短音符改进配置（基于v2.0优化，专注短音符和A4/E5检测）
        'v3.0_short_note_improved': {
            'sr': 44100,
            'hop_length': 256,
            'min_freq': 80,
            'max_freq': 1200,
            'min_duration': 0.05,
            'max_gap': 0.03,
            'voicing_threshold': 0.4,
            'harmonic_margin': 6,
            'preemphasis_coef': 0.97,
            'short_note_threshold': 0.1,
            'target_frequencies': [440, 659],
        }
    }
    return configs


def generate_v1_v2_midi(audio_path: Path, config: Dict[str, Any]) -> pretty_midi.PrettyMIDI:
    """生成v1.0或v2.0 MIDI（使用SimpleMIDIGenerator）"""
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    generator = SimpleMIDIGenerator(config)
    return generator.process_audio(audio_bytes)


def generate_v3_midi(audio_path: Path, config: Dict[str, Any]) -> pretty_midi.PrettyMIDI:
    """生成v3.0 MIDI（使用多尺度音高检测和短音符支持）"""
    if not V3_AVAILABLE:
        raise RuntimeError("v3.0算法不可用")

    # 加载音频
    y, sr = librosa.load(audio_path, sr=config['sr'], mono=True)

    # 音频预处理（与原始方法相同）
    y = librosa.effects.preemphasis(y, coef=config.get('preemphasis_coef', 0.97))
    y = librosa.util.normalize(y, axis=0)

    # 多尺度音高检测
    f0, voiced_flag = multi_scale_pitch_detection(y, sr, config)

    # 生成MIDI（支持短音符）
    midi = create_midi_with_short_note_support(f0, voiced_flag, sr, config)
    return midi


def evaluate_midi(ref_midi_path: Path, gen_midi_path: Path) -> Dict[str, float]:
    """评估MIDI的听觉相似度"""
    try:
        result = evaluate_auditory_similarity(str(ref_midi_path), str(gen_midi_path))
        return result['scores']
    except Exception as e:
        print(f"评估失败 {ref_midi_path} vs {gen_midi_path}: {e}")
        return {'overall_score': 0.0}


def find_test_files(data_dir: Path) -> List[Tuple[Path, Path]]:
    """查找测试音频和对应的参考MIDI文件"""
    test_files = []

    # 查找所有.mp3文件
    for mp3_path in data_dir.glob("*.mp3"):
        # 对应的MIDI文件（同名）
        mid_path = mp3_path.with_suffix('.mid')
        if mid_path.exists():
            test_files.append((mp3_path, mid_path))
        else:
            print(f"警告：找不到参考MIDI文件 {mid_path}，跳过 {mp3_path}")

    # 按文件名排序
    test_files.sort(key=lambda x: x[0].name)
    return test_files


def main():
    print("测试各版本在不同音频文件上的泛化能力")
    print("=" * 70)

    data_dir = Path("data")
    output_dir = Path("results/generalization_test")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 查找测试文件
    test_files = find_test_files(data_dir)
    if not test_files:
        print("错误：未找到测试文件（需.mp3和对应的.mid文件）")
        return

    print(f"找到 {len(test_files)} 个测试文件:")
    for audio_path, ref_path in test_files:
        print(f"  - {audio_path.name} (参考: {ref_path.name})")

    # 获取版本配置
    version_configs = get_version_configs()
    if not V3_AVAILABLE:
        print("警告：v3.0算法不可用，跳过v3.0测试")
        version_configs.pop('v3.0_short_note_improved', None)

    versions = list(version_configs.keys())
    print(f"\n测试版本: {', '.join(versions)}")

    # 存储所有结果
    all_results = {
        'test_files': [str(audio_path.name) for audio_path, _ in test_files],
        'versions': versions,
        'results': {},  # 按文件名和版本组织
        'summary': {},  # 汇总统计
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    total_tests = len(test_files) * len(versions)
    test_count = 0

    for audio_path, ref_midi_path in test_files:
        file_name = audio_path.stem
        print(f"\n{'='*60}")
        print(f"测试文件: {file_name}")
        print(f"音频: {audio_path.name}")
        print(f"参考MIDI: {ref_midi_path.name}")

        file_results = {}

        for version in versions:
            test_count += 1
            config = version_configs[version]
            print(f"\n  [{test_count}/{total_tests}] 生成 {version}...")

            try:
                start_time = time.time()

                # 根据版本选择生成方法
                if version == 'v3.0_short_note_improved':
                    midi = generate_v3_midi(audio_path, config)
                else:
                    midi = generate_v1_v2_midi(audio_path, config)

                generation_time = time.time() - start_time

                # 保存生成的MIDI
                midi_filename = f"{file_name}_{version.replace('.', '_')}.mid"
                midi_path = output_dir / midi_filename
                midi.write(str(midi_path))

                # 评估听觉相似度
                print(f"      评估听觉相似度...")
                scores = evaluate_midi(ref_midi_path, midi_path)

                # 记录结果
                note_count = len(midi.instruments[0].notes) if midi.instruments else 0
                file_results[version] = {
                    'midi_file': str(midi_path),
                    'generation_time_seconds': generation_time,
                    'note_count': note_count,
                    'auditory_scores': scores,
                    'config': config,
                }

                print(f"      完成！音符数: {note_count}, 综合评分: {scores['overall_score']:.3f}")

            except Exception as e:
                print(f"      生成{version}失败: {e}")
                import traceback
                traceback.print_exc()
                file_results[version] = {
                    'error': str(e),
                    'midi_file': None,
                    'generation_time_seconds': 0,
                    'note_count': 0,
                    'auditory_scores': {'overall_score': 0.0},
                }

        all_results['results'][file_name] = file_results

    # 计算汇总统计
    print(f"\n{'='*70}")
    print("汇总统计:")

    summary = {}
    for version in versions:
        version_scores = []
        version_notes = []

        for file_name in all_results['results']:
            result = all_results['results'][file_name].get(version)
            if result and 'error' not in result:
                version_scores.append(result['auditory_scores']['overall_score'])
                version_notes.append(result['note_count'])

        if version_scores:
            summary[version] = {
                'avg_score': np.mean(version_scores),
                'std_score': np.std(version_scores),
                'min_score': np.min(version_scores),
                'max_score': np.max(version_scores),
                'avg_notes': np.mean(version_notes),
                'num_successful': len(version_scores),
            }

            print(f"\n  {version}:")
            print(f"    成功测试: {len(version_scores)}/{len(test_files)}")
            print(f"    平均综合评分: {summary[version]['avg_score']:.3f} (±{summary[version]['std_score']:.3f})")
            print(f"    评分范围: {summary[version]['min_score']:.3f} - {summary[version]['max_score']:.3f}")
            print(f"    平均音符数: {summary[version]['avg_notes']:.1f}")
        else:
            summary[version] = {'num_successful': 0}
            print(f"\n  {version}: 无成功测试")

    all_results['summary'] = summary

    # 保存详细结果
    results_path = output_dir / "generalization_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"测试完成!")
    print(f"详细结果已保存: {results_path}")
    print(f"生成的MIDI文件在: {output_dir}")

    # 打印最佳版本推荐
    successful_versions = {v: s for v, s in summary.items() if s.get('num_successful', 0) > 0}
    if successful_versions:
        best_version = max(successful_versions.items(),
                          key=lambda x: x[1].get('avg_score', 0))
        print(f"\n推荐版本: {best_version[0]} (平均评分: {best_version[1]['avg_score']:.3f})")
    else:
        print("\n警告：所有版本均无成功测试")


if __name__ == "__main__":
    main()