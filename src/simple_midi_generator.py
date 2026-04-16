#!/usr/bin/env python3
"""
基于原始方法(test1_copy.py)的简单MIDI生成器
将硬编码参数改为可配置，保持原始算法逻辑
"""
import sys
import os
import tempfile
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

# 必须导入
import scipy
import scipy.signal
import librosa
import pretty_midi
import soundfile as sf


class SimpleMIDIGenerator:
    """
    基于原始方法的简单MIDI生成器
    保持test1_copy.py的核心算法，但参数可配置
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化生成器

        Args:
            config: 配置参数字典，可选
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置参数（基于test1_copy.py的原始值）"""
        return {
            # 音频处理参数
            'sr': 44100,                    # 采样率
            'hop_length': 512,              # 帧移长度（原始值）

            # 频率范围
            'min_freq': 80,                 # 最低频率（Hz）
            'max_freq': 1200,               # 最高频率（Hz）

            # 音高检测参数
            'n_thresholds': 200,            # PYIN阈值数量
            'resolution': 0.1,              # PYIN分辨率
            'harmonic_margin': 3,           # 谐波增强边界

            # 音符分割参数
            'min_duration': 0.1,            # 最小音符时长（秒）- 原始硬编码值
            'max_gap': 0.05,                # 最大间隙（秒）- 原始硬编码值
            'voicing_threshold': 0.6,       # 发声检测阈值

            # 音频预处理
            'preemphasis_coef': 0.97,       # 预加重系数
            'normalize_axis': 0,            # 归一化轴
        }

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        """
        处理音频并生成MIDI - 基于test1_copy.py的process_audio函数

        Args:
            audio_bytes: 原始音频字节

        Returns:
            pretty_midi.PrettyMIDI对象
        """
        tmp_path = None
        try:
            # 创建临时文件（确保跨平台兼容）
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            # 加载并预处理音频
            y, sr = librosa.load(tmp_path, sr=self.config['sr'], mono=True)
            y = librosa.effects.preemphasis(y, coef=self.config['preemphasis_coef'])
            y = librosa.util.normalize(y, axis=self.config['normalize_axis'])

            # 智能长度调整（兼容各librosa版本）
            target_length = ((len(y) // self.config['hop_length']) + 1) * self.config['hop_length']
            try:
                # 新版本处理方式（优先尝试）
                y = librosa.util.fix_length(y, size=target_length, mode='wrap')
            except TypeError as e:
                if "unexpected keyword argument 'mode'" in str(e):
                    # 旧版本回退方案
                    y = librosa.util.fix_length(y, target_length)
                    if len(y) < target_length:
                        repeats = (target_length // len(y)) + 1
                        y = np.tile(y, repeats)[:target_length]
                    else:
                        y = y[:target_length]
                else:
                    raise

            # 音高检测
            f0, voiced_flag = self._enhanced_pitch_detection(y, sr)

            # 后处理优化（使用高斯滤波）- 原始方法中的处理
            voiced_flag = np.convolve(voiced_flag, [0.25, 0.5, 0.25], mode='same') > 0.6

            # MIDI生成
            midi = self._create_midi(f0, voiced_flag, sr)
            if not midi.instruments[0].notes:
                raise ValueError("生成的MIDI文件无有效音符")

            return midi

        except Exception as e:
            raise RuntimeError(f"音频处理流程异常: {str(e)}") from e
        finally:
            # 确保清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as clean_error:
                    print(f"临时文件清理警告: {clean_error}")

    def _enhanced_pitch_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        稳健版音高检测算法 - 基于test1_copy.py的enhanced_pitch_detection

        Args:
            y: 音频信号
            sr: 采样率

        Returns:
            f0: 基频序列
            voiced_flag: 发声标志序列
        """
        hop_length = self.config['hop_length']
        min_freq = self.config['min_freq']
        max_freq = self.config['max_freq']

        # 参数标准化
        frame_length = 2048
        if hop_length > frame_length // 4:
            frame_length = hop_length * 4  # 自动调整帧长保持分析窗口

        # 主检测（增加异常处理）
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=max(20, min_freq),  # 确保不低于物理下限
                fmax=min(20000, max_freq),
                sr=sr,
                hop_length=hop_length,
                frame_length=frame_length,
                n_thresholds=self.config['n_thresholds'],
                resolution=self.config['resolution'],
                fill_na=np.nan
            )
        except Exception as e:
            raise RuntimeError(f"PYIN音高检测失败: {str(e)}") from e

        # 谐波增强检测
        try:
            harmonic = librosa.effects.harmonic(y, margin=self.config['harmonic_margin'])
            f0_harmonic, _, _ = librosa.pyin(
                harmonic,
                fmin=min_freq,
                fmax=max_freq,
                sr=sr,
                hop_length=hop_length,
                frame_length=frame_length,
                fill_na=np.nan
            )
        except Exception as e:
            f0_harmonic = np.full_like(f0, np.nan)  # 谐波检测失败时降级处理

        # 智能融合策略优化
        fused_f0 = np.where(np.isnan(f0), f0_harmonic, f0)

        # 置信度计算修正
        S, phase = librosa.magphase(librosa.stft(y, hop_length=hop_length))
        confidence = librosa.feature.spectral_flatness(S=S)
        confidence = librosa.util.normalize(confidence.squeeze(), axis=0)

        # 基于置信度的加权融合
        alpha = np.clip(confidence, 0.2, 0.8)  # 限制权重范围
        fused_f0 = alpha * f0 + (1 - alpha) * np.nan_to_num(f0_harmonic, nan=0)

        # 后处理流程优化
        valid_mask = ~np.isnan(fused_f0)
        if np.any(valid_mask):
            x = np.arange(len(fused_f0))
            fused_f0 = np.interp(x, x[valid_mask], fused_f0[valid_mask])
            fused_f0 = scipy.signal.medfilt(fused_f0, kernel_size=5)
        else:
            fused_f0[:] = 0

        # 生成最终有效标志（增加频率范围校验）
        voiced_flag = (fused_f0 >= min_freq) & (fused_f0 <= max_freq)
        return fused_f0, voiced_flag

    def _create_midi(self, f0: np.ndarray, voiced_flag: np.ndarray, sr: int) -> pretty_midi.PrettyMIDI:
        """
        稳健版MIDI生成算法 - 基于test1_copy.py的create_midi

        Args:
            f0: 基频序列
            voiced_flag: 发声标志序列
            sr: 采样率

        Returns:
            pretty_midi.PrettyMIDI对象
        """
        hop_length = self.config['hop_length']
        min_duration = self.config['min_duration']
        max_gap = self.config['max_gap']

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_note = None
        last_end = 0.0

        for i in range(len(f0)):
            time = i * hop_length / sr
            is_voiced = voiced_flag[i]
            pitch = f0[i] if is_voiced else None

            if pitch and not np.isnan(pitch):
                note_num = int(round(librosa.hz_to_midi(pitch)))

                # 音高有效时处理
                if current_note is None:
                    # 开始新音符
                    current_note = {
                        'pitch': note_num,
                        'start': max(last_end, time - max_gap),
                        'end': time
                    }
                elif note_num == current_note['pitch']:
                    # 延续当前音符
                    current_note['end'] = time
                else:
                    # 音高变化时结束当前音符
                    if current_note['end'] - current_note['start'] >= min_duration:
                        instrument.notes.append(pretty_midi.Note(
                            velocity=100,
                            pitch=current_note['pitch'],
                            start=current_note['start'],
                            end=current_note['end']
                        ))
                    last_end = current_note['end']
                    current_note = {
                        'pitch': note_num,
                        'start': time,
                        'end': time
                    }
            else:
                # 静音段处理
                if current_note is not None:
                    if current_note['end'] - current_note['start'] >= min_duration:
                        instrument.notes.append(pretty_midi.Note(
                            velocity=100,
                            pitch=current_note['pitch'],
                            start=current_note['start'],
                            end=current_note['end']
                        ))
                    last_end = current_note['end']
                    current_note = None

        # 处理最后一个音符
        if current_note and (current_note['end'] - current_note['start']) >= min_duration:
            instrument.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=current_note['pitch'],
                start=current_note['start'],
                end=current_note['end']
            ))

        if not instrument.notes:
            raise ValueError("无有效音符生成，请检查输入音频")

        midi.instruments.append(instrument)
        return midi

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def update_config(self, config_updates: Dict[str, Any]):
        """更新配置参数"""
        self.config.update(config_updates)


# 便捷函数，保持与原始方法类似的接口
def process_audio_simple(audio_bytes: bytes, **kwargs) -> pretty_midi.PrettyMIDI:
    """
    便捷函数：使用简单MIDI生成器处理音频

    Args:
        audio_bytes: 原始音频字节
        **kwargs: 配置参数，覆盖默认值

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    generator = SimpleMIDIGenerator(kwargs)
    return generator.process_audio(audio_bytes)


if __name__ == "__main__":
    # 简单测试
    import sys
    from pathlib import Path

    # 测试配置
    config = {
        'sr': 44100,
        'hop_length': 512,
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.1,
        'max_gap': 0.05,
    }

    # 如果有音频文件路径参数
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if audio_path.exists():
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            generator = SimpleMIDIGenerator(config)
            midi = generator.process_audio(audio_bytes)

            output_path = audio_path.with_suffix('.mid')
            midi.write(str(output_path))
            print(f"生成MIDI: {output_path}, 音符数: {len(midi.instruments[0].notes)}")
        else:
            print(f"文件不存在: {audio_path}")
    else:
        print("使用方法: python simple_midi_generator.py <音频文件路径>")
        print("当前配置:", config)