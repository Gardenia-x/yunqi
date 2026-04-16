#!/usr/bin/env python3
"""
轻量级听觉优化器
使用网格搜索和随机搜索优化MIDI生成参数，以听觉相似度为目标
"""
import sys
import random
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import time

from simple_midi_generator import SimpleMIDIGenerator
from auditory_similarity_evaluator import AuditorySimilarityEvaluator
import pretty_midi


@dataclass
class OptimizationResult:
    """优化结果"""
    params: Dict[str, Any]
    scores: Dict[str, float]
    midi_data: pretty_midi.PrettyMIDI
    generation_time: float


class SimpleAuditoryOptimizer:
    """
    轻量级听觉优化器
    优化MIDI生成参数以最大化听觉相似度
    """

    def __init__(self, reference_midi: pretty_midi.PrettyMIDI):
        """
        初始化优化器

        Args:
            reference_midi: 参考MIDI文件
        """
        self.reference_midi = reference_midi
        self.evaluator = AuditorySimilarityEvaluator(reference_midi)

        # 默认参数搜索范围（基于原始方法）
        self.param_ranges = self._get_default_param_ranges()

    def _get_default_param_ranges(self) -> Dict[str, List[Any]]:
        """获取默认参数搜索范围"""
        return {
            # 关键参数：影响听觉体验最大
            'min_duration': [0.05, 0.08, 0.1, 0.12, 0.15],  # 短音符检测
            'max_gap': [0.03, 0.05, 0.07, 0.1],            # 音符合并阈值
            'voicing_threshold': [0.4, 0.5, 0.6, 0.7],     # 发声检测灵敏度

            # 次要参数：微调
            'harmonic_margin': [2, 3, 4, 5],               # 谐波增强强度
            'hop_length': [256, 384, 512, 640],           # 时间分辨率

            # 频率范围（通常保持原始范围）
            'min_freq': [80],                             # 保持原始值
            'max_freq': [1200],                           # 保持原始值
        }

    def analyze_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        分析音频特性，自适应调整参数范围

        Args:
            audio_bytes: 音频数据

        Returns:
            音频特性分析结果
        """
        # 简单实现：返回默认范围
        # TODO: 实现音频特性分析（音高范围、音符密度等）
        return {
            'note_density': 'medium',  # low, medium, high
            'pitch_range': 'mid',      # low, mid, high
            'dynamic_range': 'medium', # low, medium, high
        }

    def optimize(self, audio_bytes: bytes, strategy: str = 'grid',
                 max_iterations: int = 50, timeout: float = 300.0) -> OptimizationResult:
        """
        优化参数以最大化听觉相似度

        Args:
            audio_bytes: 音频数据
            strategy: 优化策略 ('grid', 'random', 'hybrid')
            max_iterations: 最大迭代次数
            timeout: 超时时间（秒）

        Returns:
            最佳优化结果
        """
        start_time = time.time()
        best_result = None
        best_score = -1.0

        # 分析音频特性（未来版本）
        audio_features = self.analyze_audio(audio_bytes)

        # 根据策略生成参数组合
        param_combinations = self._generate_param_combinations(strategy, max_iterations)

        print(f"开始优化，策略: {strategy}, 参数组合数: {len(param_combinations)}")
        print(f"音频特性: {audio_features}")

        for i, params in enumerate(param_combinations):
            if time.time() - start_time > timeout:
                print(f"超时，已运行 {time.time() - start_time:.1f} 秒")
                break

            try:
                # 生成MIDI
                gen_start = time.time()
                generator = SimpleMIDIGenerator(params)
                midi = generator.process_audio(audio_bytes)
                gen_time = time.time() - gen_start

                # 评估听觉相似度
                scores = self.evaluator.evaluate(midi)
                overall_score = scores['overall_score']

                # 记录结果
                result = OptimizationResult(
                    params=params.copy(),
                    scores=scores,
                    midi_data=midi,
                    generation_time=gen_time
                )

                # 更新最佳结果
                if overall_score > best_score:
                    best_score = overall_score
                    best_result = result
                    print(f"  迭代 {i+1}: 新最佳分数 {overall_score:.3f}")

                # 进度报告
                if (i + 1) % 10 == 0:
                    print(f"  进度: {i+1}/{len(param_combinations)}, 当前最佳: {best_score:.3f}")

            except Exception as e:
                print(f"  迭代 {i+1} 失败: {e}")
                continue

        if best_result is None:
            raise RuntimeError("优化失败，未找到有效参数组合")

        total_time = time.time() - start_time
        print(f"\n优化完成!")
        print(f"  总时间: {total_time:.1f} 秒")
        print(f"  最佳分数: {best_score:.3f}")
        print(f"  最佳参数: {self._format_params(best_result.params)}")

        return best_result

    def _generate_param_combinations(self, strategy: str, max_iterations: int) -> List[Dict[str, Any]]:
        """生成参数组合"""
        combinations = []

        if strategy == 'grid':
            # 网格搜索：所有参数组合
            combinations = self._grid_search_combinations()
            # 限制数量
            if len(combinations) > max_iterations:
                # 随机抽样
                indices = np.random.choice(len(combinations), max_iterations, replace=False)
                combinations = [combinations[i] for i in indices]

        elif strategy == 'random':
            # 随机搜索：随机生成参数组合
            combinations = []
            for _ in range(max_iterations):
                params = {}
                for param_name, values in self.param_ranges.items():
                    params[param_name] = random.choice(values)
                combinations.append(params)

        elif strategy == 'hybrid':
            # 混合策略：先网格后随机
            grid_combs = self._grid_search_combinations()
            if len(grid_combs) > max_iterations // 2:
                indices = np.random.choice(len(grid_combs), max_iterations // 2, replace=False)
                grid_combs = [grid_combs[i] for i in indices]

            combinations = grid_combs
            # 补充随机组合
            remaining = max_iterations - len(combinations)
            for _ in range(remaining):
                params = {}
                for param_name, values in self.param_ranges.items():
                    params[param_name] = random.choice(values)
                combinations.append(params)

        else:
            raise ValueError(f"未知策略: {strategy}")

        return combinations

    def _grid_search_combinations(self) -> List[Dict[str, Any]]:
        """生成网格搜索的所有参数组合"""
        from itertools import product

        # 将参数范围转换为列表的列表
        param_lists = []
        param_names = []
        for name, values in self.param_ranges.items():
            param_names.append(name)
            param_lists.append(values)

        # 生成所有组合
        combinations = []
        for values in product(*param_lists):
            params = dict(zip(param_names, values))
            combinations.append(params)

        return combinations

    def _format_params(self, params: Dict[str, Any]) -> str:
        """格式化参数字典为可读字符串"""
        important_params = ['min_duration', 'max_gap', 'voicing_threshold', 'harmonic_margin', 'hop_length']
        parts = []
        for key in important_params:
            if key in params:
                parts.append(f"{key}: {params[key]}")
        return "{" + ", ".join(parts) + "}"

    def optimize_with_feedback(self, audio_bytes: bytes, initial_params: Optional[Dict[str, Any]] = None,
                               iterations: int = 3) -> OptimizationResult:
        """
        带反馈的渐进优化（多轮优化）

        Args:
            audio_bytes: 音频数据
            initial_params: 初始参数（可选）
            iterations: 优化轮数

        Returns:
            最佳优化结果
        """
        if initial_params is None:
            # 使用原始方法参数
            initial_params = {
                'min_duration': 0.1,
                'max_gap': 0.05,
                'voicing_threshold': 0.6,
                'harmonic_margin': 3,
                'hop_length': 512,
            }

        current_params = initial_params.copy()
        best_result = None
        best_score = -1.0

        for iteration in range(iterations):
            print(f"\n=== 第 {iteration + 1}/{iterations} 轮优化 ===")
            print(f"当前参数: {self._format_params(current_params)}")

            # 在当前参数附近搜索
            local_ranges = self._create_local_ranges(current_params)
            self.param_ranges = local_ranges

            # 执行一轮优化
            result = self.optimize(audio_bytes, strategy='random', max_iterations=20, timeout=60.0)

            # 更新最佳结果
            if result.scores['overall_score'] > best_score:
                best_score = result.scores['overall_score']
                best_result = result
                current_params = result.params.copy()

            print(f"本轮最佳分数: {result.scores['overall_score']:.3f}")

        if best_result is None:
            raise RuntimeError("渐进优化失败")

        return best_result

    def _create_local_ranges(self, center_params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """创建局部搜索范围（在中心参数附近）"""
        local_ranges = {}

        # 定义每个参数的搜索步长
        step_sizes = {
            'min_duration': 0.02,
            'max_gap': 0.01,
            'voicing_threshold': 0.05,
            'harmonic_margin': 1,
            'hop_length': 64,
        }

        for param_name, values in self.param_ranges.items():
            if param_name in center_params and param_name in step_sizes:
                center = center_params[param_name]
                step = step_sizes[param_name]

                # 生成局部值
                local_values = []
                for offset in [-2, -1, 0, 1, 2]:
                    if param_name == 'hop_length':
                        # hop_length需要是整数
                        value = int(center + offset * step)
                        if value > 0:
                            local_values.append(value)
                    else:
                        value = center + offset * step
                        if value > 0:
                            local_values.append(round(value, 3))

                # 去重并排序
                local_values = sorted(set(local_values))
                local_ranges[param_name] = local_values
            else:
                # 保持原始范围
                local_ranges[param_name] = values

        return local_ranges


# 便捷函数
def optimize_midi_generation(audio_path: str, reference_midi_path: str,
                             output_dir: str = 'results',
                             strategy: str = 'hybrid') -> Dict[str, Any]:
    """
    便捷函数：优化MIDI生成参数

    Args:
        audio_path: 音频文件路径
        reference_midi_path: 参考MIDI文件路径
        output_dir: 输出目录
        strategy: 优化策略

    Returns:
        优化结果字典
    """
    # 加载数据
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    reference_midi = pretty_midi.PrettyMIDI(reference_midi_path)

    # 创建优化器
    optimizer = SimpleAuditoryOptimizer(reference_midi)

    # 执行优化
    result = optimizer.optimize(audio_bytes, strategy=strategy, max_iterations=30, timeout=180.0)

    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 保存最佳MIDI
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    midi_filename = f"optimized_{timestamp}.mid"
    midi_path = output_dir / midi_filename
    result.midi_data.write(str(midi_path))

    # 保存参数和评分
    result_dict = {
        'params': result.params,
        'scores': result.scores,
        'generation_time': result.generation_time,
        'midi_file': str(midi_path),
        'timestamp': timestamp,
        'strategy': strategy,
    }

    json_path = output_dir / f"optimization_result_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2, default=str)

    print(f"\n结果已保存:")
    print(f"  MIDI文件: {midi_path}")
    print(f"  参数文件: {json_path}")

    return result_dict


if __name__ == "__main__":
    # 简单测试
    import sys
    from pathlib import Path

    if len(sys.argv) >= 3:
        audio_path = Path(sys.argv[1])
        ref_path = Path(sys.argv[2])

        if audio_path.exists() and ref_path.exists():
            result = optimize_midi_generation(str(audio_path), str(ref_path))
            print(f"\n优化完成!")
            print(f"综合评分: {result['scores']['overall_score']:.3f}")
        else:
            print(f"文件不存在: {audio_path if not audio_path.exists() else ref_path}")
    else:
        print("使用方法: python simple_auditory_optimizer.py <音频文件> <参考MIDI>")
        print("示例: python simple_auditory_optimizer.py data/test2.mp3 data/test2.mid")
        print("\n可选参数:")
        print("  策略: grid, random, hybrid (默认: hybrid)")