#!/usr/bin/env python3
"""
增强版MIDI生成器
基于原始方法改进，专注于提高听觉流畅度
"""
import sys
import os
import tempfile
import traceback
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

# 必须导入
import scipy
import scipy.signal
import librosa
import pretty_midi
import soundfile as sf


class EnhancedMIDIGenerator:
    """
    增强版MIDI生成器
    在原始方法基础上改进音高检测和音符分割，提高听觉流畅度
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
        """获取增强版默认配置"""
        return {
            # 音频处理参数
            'sr': 44100,                    # 采样率
            'hop_length': 256,              # 提高时间分辨率（基于v2.0优化结果）

            # 频率范围
            'min_freq': 80,                 # 最低频率（Hz）
            'max_freq': 1200,               # 最高频率（Hz）

            # 音高检测参数
            'n_thresholds': 200,            # PYIN阈值数量
            'resolution': 0.1,              # PYIN分辨率
            'harmonic_margin': 5,           # 谐波增强边界（基于v2.0优化结果）

            # 音符分割参数（基于v2.0优化结果）
            'min_duration': 0.05,           # 最小音符时长（秒）
            'max_gap': 0.03,                # 最大间隙（秒）
            'voicing_threshold': 0.4,       # 发声检测阈值

            # 增强参数：流畅度优化
            'pitch_smooth_window': 7,       # 音高平滑窗口大小（帧数）
            'note_merge_threshold': 2,      # 音符合并阈值（半音）
            'dynamic_onset_detection': True, # 动态onset检测
            'energy_threshold_ratio': 0.3,  # 能量阈值比例
            'min_silence_duration': 0.02,   # 最小静音时长（秒）

            # 音频预处理
            'preemphasis_coef': 0.97,       # 预加重系数
            'normalize_axis': 0,            # 归一化轴
        }

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        """
        处理音频并生成MIDI - 增强版算法

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
            hop_length = self.config['hop_length']
            target_length = ((len(y) // hop_length) + 1) * hop_length
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

            # 增强版音高检测
            f0, voiced_flag = self._enhanced_pitch_detection(y, sr)

            # 音高平滑（提高流畅度）
            if self.config['pitch_smooth_window'] > 1:
                f0 = self._smooth_pitch(f0, self.config['pitch_smooth_window'])

            # 动态onset检测（可选）
            if self.config['dynamic_onset_detection']:
                onsets = self._detect_onsets(y, sr)
                # 结合onset信息优化发声标志
                voiced_flag = self._enhance_with_onsets(voiced_flag, onsets, sr)

            # 后处理优化
            voiced_flag = np.convolve(voiced_flag, [0.25, 0.5, 0.25], mode='same') > 0.6

            # 增强版MIDI生成
            midi = self._create_enhanced_midi(f0, voiced_flag, sr, y)
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
        增强版音高检测算法
        在原始方法基础上增加多重检测和融合策略

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

        # 主检测
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
            f0_harmonic = np.full_like(f0, np.nan)

        # 置信度计算修正
        S, phase = librosa.magphase(librosa.stft(y, hop_length=hop_length))
        confidence = librosa.feature.spectral_flatness(S=S)
        confidence = librosa.util.normalize(confidence.squeeze(), axis=0)

        # 智能融合策略：基于置信度的加权融合
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

        # 生成最终有效标志
        voiced_flag = (fused_f0 >= min_freq) & (fused_f0 <= max_freq)
        return fused_f0, voiced_flag

    def _smooth_pitch(self, f0: np.ndarray, window_size: int) -> np.ndarray:
        """
        音高平滑处理，减少跳变

        Args:
            f0: 原始基频序列
            window_size: 平滑窗口大小

        Returns:
            平滑后的基频序列
        """
        if window_size < 2:
            return f0

        # 使用移动平均平滑，但保留NaN值
        smoothed = f0.copy()
        valid_mask = ~np.isnan(f0)

        if np.any(valid_mask):
            # 只对有效值进行平滑
            valid_f0 = f0[valid_mask]
            smoothed_valid = np.convolve(valid_f0, np.ones(window_size)/window_size, mode='same')
            smoothed[valid_mask] = smoothed_valid

        return smoothed

    def _detect_onsets(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        动态onset检测，基于音频能量变化

        Args:
            y: 音频信号
            sr: 采样率

        Returns:
            onset时间点（秒）
        """
        hop_length = self.config['hop_length']

        # 计算能量包络
        energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # 归一化能量
        energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

        # 检测onset（能量显著上升点）
        onsets = []
        threshold = self.config['energy_threshold_ratio']

        for i in range(1, len(energy_norm)):
            if energy_norm[i] - energy_norm[i-1] > threshold:
                time = i * hop_length / sr
                onsets.append(time)

        return np.array(onsets)

    def _enhance_with_onsets(self, voiced_flag: np.ndarray, onsets: np.ndarray, sr: int) -> np.ndarray:
        """
        使用onset信息增强发声标志

        Args:
            voiced_flag: 原始发声标志
            onsets: onset时间点
            sr: 采样率

        Returns:
            增强后的发声标志
        """
        hop_length = self.config['hop_length']
        enhanced = voiced_flag.copy()

        # 将onset时间转换为帧索引
        for onset_time in onsets:
            frame_idx = int(onset_time * sr / hop_length)
            if frame_idx < len(enhanced):
                # 在onset点强制设置为发声
                enhanced[frame_idx] = True
                # 向前后扩展几帧
                start = max(0, frame_idx - 2)
                end = min(len(enhanced), frame_idx + 3)
                enhanced[start:end] = True

        return enhanced

    def _create_enhanced_midi(self, f0: np.ndarray, voiced_flag: np.ndarray, sr: int,
                            audio_signal: np.ndarray) -> pretty_midi.PrettyMIDI:
        """
        增强版MIDI生成算法
        改进音符分割，提高听觉流畅度

        Args:
            f0: 基频序列
            voiced_flag: 发声标志序列
            sr: 采样率
            audio_signal: 原始音频信号（用于能量分析）

        Returns:
            pretty_midi.PrettyMIDI对象
        """
        hop_length = self.config['hop_length']
        min_duration = self.config['min_duration']
        max_gap = self.config['max_gap']
        merge_threshold = self.config['note_merge_threshold']

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # 第一步：生成原始音符片段
        raw_notes = self._extract_raw_notes(f0, voiced_flag, sr)

        # 第二步：音符合并优化
        merged_notes = self._merge_similar_notes(raw_notes, merge_threshold, min_duration)

        # 第三步：基于能量的音符边界调整
        final_notes = self._adjust_note_boundaries(merged_notes, audio_signal, sr)

        # 添加到乐器
        for note in final_notes:
            if note['end'] - note['start'] >= min_duration:
                instrument.notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=note['pitch'],
                    start=note['start'],
                    end=note['end']
                ))

        if not instrument.notes:
            raise ValueError("无有效音符生成，请检查输入音频")

        midi.instruments.append(instrument)
        return midi

    def _extract_raw_notes(self, f0: np.ndarray, voiced_flag: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        提取原始音符片段（基于简单分割）

        Args:
            f0: 基频序列
            voiced_flag: 发声标志序列
            sr: 采样率

        Returns:
            原始音符片段列表
        """
        hop_length = self.config['hop_length']
        max_gap = self.config['max_gap']

        notes = []
        current_note = None
        last_end = 0.0

        for i in range(len(f0)):
            time = i * hop_length / sr
            is_voiced = voiced_flag[i]
            pitch = f0[i] if is_voiced else None

            if pitch and not np.isnan(pitch):
                note_num = int(round(librosa.hz_to_midi(pitch)))

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
                    notes.append(current_note.copy())
                    last_end = current_note['end']
                    current_note = {
                        'pitch': note_num,
                        'start': time,
                        'end': time
                    }
            else:
                # 静音段处理
                if current_note is not None:
                    notes.append(current_note.copy())
                    last_end = current_note['end']
                    current_note = None

        # 处理最后一个音符
        if current_note is not None:
            notes.append(current_note)

        return notes

    def _merge_similar_notes(self, notes: List[Dict[str, Any]],
                           merge_threshold: int, min_duration: float) -> List[Dict[str, Any]]:
        """
        合并相似音高的相邻音符，提高流畅度

        Args:
            notes: 原始音符列表
            merge_threshold: 音符合并阈值（半音）
            min_duration: 最小音符时长

        Returns:
            合并后的音符列表
        """
        if not notes:
            return []

        merged = []
        current = notes[0].copy()

        for i in range(1, len(notes)):
            next_note = notes[i]

            # 检查是否应该合并
            pitch_diff = abs(next_note['pitch'] - current['pitch'])
            time_gap = next_note['start'] - current['end']
            max_gap = self.config['max_gap']

            if (pitch_diff <= merge_threshold and
                time_gap <= max_gap and
                time_gap >= -max_gap):  # 允许轻微重叠
                # 合并音符
                current['end'] = next_note['end']
                # 音高取平均（加权平均）
                current_duration = current['end'] - current['start']
                next_duration = next_note['end'] - next_note['start']
                total_duration = current_duration + next_duration

                if total_duration > 0:
                    current['pitch'] = int(round(
                        (current['pitch'] * current_duration +
                         next_note['pitch'] * next_duration) / total_duration
                    ))
            else:
                # 不合并，保存当前音符
                if current['end'] - current['start'] >= min_duration:
                    merged.append(current.copy())
                current = next_note.copy()

        # 处理最后一个音符
        if current['end'] - current['start'] >= min_duration:
            merged.append(current)

        return merged

    def _adjust_note_boundaries(self, notes: List[Dict[str, Any]],
                              audio_signal: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """
        基于音频能量调整音符边界，使边界更自然

        Args:
            notes: 音符列表
            audio_signal: 音频信号
            sr: 采样率

        Returns:
            调整边界后的音符列表
        """
        adjusted_notes = []

        for note in notes:
            # 计算音符区域内的能量
            start_sample = int(note['start'] * sr)
            end_sample = int(note['end'] * sr)

            # 确保索引在范围内
            start_sample = max(0, min(start_sample, len(audio_signal)-1))
            end_sample = max(0, min(end_sample, len(audio_signal)-1))

            if start_sample >= end_sample:
                # 无效音符，跳过
                continue

            note_signal = audio_signal[start_sample:end_sample]
            if len(note_signal) == 0:
                continue

            # 计算能量包络
            hop_length = 256  # 用于能量分析的hop_length
            energy = librosa.feature.rms(y=note_signal, hop_length=hop_length)[0]

            if len(energy) < 3:
                # 能量分析数据太少，保持原样
                adjusted_notes.append(note.copy())
                continue

            # 归一化能量
            energy_norm = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-10)

            # 查找能量上升和下降点
            onset_threshold = 0.3
            offset_threshold = 0.2

            # 调整开始时间（向前找能量上升点）
            adjusted_start = note['start']
            for i in range(len(energy_norm)):
                if i > 0 and energy_norm[i] - energy_norm[i-1] > onset_threshold:
                    # 找到能量显著上升点
                    adjusted_start = note['start'] + (i * hop_length / sr)
                    break

            # 调整结束时间（向后找能量下降点）
            adjusted_end = note['end']
            for i in range(len(energy_norm)-1, -1, -1):
                if i > 0 and energy_norm[i-1] - energy_norm[i] > offset_threshold:
                    # 找到能量显著下降点
                    adjusted_end = note['start'] + (i * hop_length / sr)
                    break

            # 确保调整后的时长合理
            min_silence = self.config['min_silence_duration']
            if adjusted_end - adjusted_start >= min_silence:
                adjusted_note = note.copy()
                adjusted_note['start'] = adjusted_start
                adjusted_note['end'] = adjusted_end
                adjusted_notes.append(adjusted_note)
            else:
                # 调整后时长太短，保持原样
                adjusted_notes.append(note.copy())

        return adjusted_notes

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def update_config(self, config_updates: Dict[str, Any]):
        """更新配置参数"""
        self.config.update(config_updates)


# 便捷函数
def process_audio_enhanced(audio_bytes: bytes, **kwargs) -> pretty_midi.PrettyMIDI:
    """
    便捷函数：使用增强版MIDI生成器处理音频

    Args:
        audio_bytes: 原始音频字节
        **kwargs: 配置参数，覆盖默认值

    Returns:
        pretty_midi.PrettyMIDI对象
    """
    generator = EnhancedMIDIGenerator(kwargs)
    return generator.process_audio(audio_bytes)


if __name__ == "__main__":
    # 简单测试
    import sys
    from pathlib import Path

    # 测试配置
    config = {
        'sr': 44100,
        'hop_length': 256,
        'min_freq': 80,
        'max_freq': 1200,
        'min_duration': 0.05,
        'max_gap': 0.03,
    }

    # 如果有音频文件路径参数
    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
        if audio_path.exists():
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()

            generator = EnhancedMIDIGenerator(config)
            midi = generator.process_audio(audio_bytes)

            output_path = audio_path.with_suffix('.enhanced.mid')
            midi.write(str(output_path))
            print(f"生成增强版MIDI: {output_path}, 音符数: {len(midi.instruments[0].notes)}")
        else:
            print(f"文件不存在: {audio_path}")
    else:
        print("使用方法: python enhanced_midi_generator.py <音频文件路径>")
        print("当前配置:", config)