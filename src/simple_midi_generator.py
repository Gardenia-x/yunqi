#!/usr/bin/env python3
"""
简单MIDI生成器 — 从音频提取旋律生成MIDI
基于 PYIN 音高检测 + 改进的音符分割算法
"""
import sys
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np

import scipy.signal
import librosa
import pretty_midi
import soundfile as sf


class SimpleMIDIGenerator:
    """基于PYIN的MIDI生成器，参数可配置"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'sr': 44100,
            'hop_length': 512,
            'min_freq': 65,          # C2 — 覆盖更多人声和乐器范围
            'max_freq': 2000,        # B6 — 捕获高音旋律
            'n_thresholds': 200,
            'resolution': 0.1,
            'harmonic_margin': 4,
            'min_duration': 0.06,    # 60ms，保留短装饰音
            'max_gap': 0.06,
            'semitone_tolerance': 1, # 相邻帧差≤1半音视为同一音符（减少碎片化）
            'pitch_smooth_kernel': 7,  # 中值滤波核大小
            'confidence_threshold': 0.3,
            'voicing_threshold': 0.5,
            'preemphasis_coef': 0.97,
            'normalize_axis': 0,
        }

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            y, sr = librosa.load(tmp_path, sr=self.config['sr'], mono=True)
            y = librosa.effects.preemphasis(y, coef=self.config['preemphasis_coef'])
            y = librosa.util.normalize(y, axis=self.config['normalize_axis'])

            # 长度对齐
            hop = self.config['hop_length']
            target_length = ((len(y) // hop) + 1) * hop
            try:
                y = librosa.util.fix_length(y, size=target_length, mode='wrap')
            except TypeError:
                y = librosa.util.fix_length(y, target_length)
                if len(y) < target_length:
                    y = np.tile(y, np.ceil(target_length / len(y)).astype(int))[:target_length]
                else:
                    y = y[:target_length]

            # 音高检测
            f0, voiced_flag, amplitude = self._pitch_detection(y, sr)

            # MIDI生成（传入振幅用于力度控制）
            midi = self._create_midi(f0, voiced_flag, amplitude, sr)
            if not midi.instruments[0].notes:
                raise ValueError("生成的MIDI文件无有效音符")

            return midi

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _pitch_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        改进的音高检测：单次 PYIN + 更好的后处理

        Returns:
            f0: 平滑后的基频序列 (Hz)
            voiced_flag: 发声标志
            amplitude: 每帧能量（用于力度控制）
        """
        hop = self.config['hop_length']
        fmin = self.config['min_freq']
        fmax = self.config['max_freq']

        # 帧长自适应：确保至少包含2个周期的最低频率波长
        frame_length = max(2048, int(sr / fmin * 2))
        frame_length = 2 ** int(np.ceil(np.log2(frame_length)))  # 向上取整到2的幂

        # ---- PYIN主检测 ----
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop,
            frame_length=frame_length,
            n_thresholds=self.config['n_thresholds'],
            resolution=self.config['resolution'],
            fill_na=np.nan,
        )

        # ---- 计算每帧RMS能量 ----
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        amplitude = np.clip((rms_db + 60) / 60, 0.05, 1.0)  # 归一化到 0.05-1.0

        # ---- 谐波增强检测（作为补充） ----
        try:
            harmonic = librosa.effects.harmonic(y, margin=self.config['harmonic_margin'])
            f0_h, vo_flag_h, _ = librosa.pyin(
                harmonic,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                hop_length=hop,
                frame_length=frame_length,
                fill_na=np.nan,
            )
        except Exception:
            f0_h = np.full_like(f0, np.nan)
            vo_flag_h = np.zeros_like(voiced_flag)

        # ---- 融合：优先取有发声的、置信度高的 ----
        conf_thresh = self.config['confidence_threshold']
        use_harmonic = (voiced_prob < conf_thresh) & vo_flag_h
        fused_f0 = np.where(use_harmonic, f0_h, f0)
        fused_voiced = voiced_flag | (vo_flag_h & (voiced_prob >= conf_thresh * 0.5))

        # ---- 后处理：插值 + 中值滤波 ----
        valid = ~np.isnan(fused_f0)
        if np.any(valid):
            x = np.arange(len(fused_f0))
            fused_f0 = np.interp(x, x[valid], fused_f0[valid])
            # 中值滤波去除毛刺
            kernel = self.config['pitch_smooth_kernel']
            fused_f0 = scipy.signal.medfilt(fused_f0, kernel_size=kernel)

        # ---- 基于能量的二次发声判断 ----
        energy_voiced = rms > np.median(rms) * 0.3
        final_voiced = fused_voiced & energy_voiced & (fused_f0 >= fmin) & (fused_f0 <= fmax)

        return fused_f0, final_voiced, amplitude

    def _create_midi(
        self, f0: np.ndarray, voiced_flag: np.ndarray,
        amplitude: np.ndarray, sr: int
    ) -> pretty_midi.PrettyMIDI:
        """
        改进的MIDI生成：
        - 半音容差避免音符碎片化
        - 基于能量的力度控制
        - 最小静音段处理避免卡顿
        """
        hop = self.config['hop_length']
        min_dur = self.config['min_duration']
        max_gap = self.config['max_gap']
        tolerance = self.config['semitone_tolerance']

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_note = None
        pending_notes = []

        def finalize_note(note_dict):
            """结束一个音符，检查最小时长"""
            dur = note_dict['end'] - note_dict['start']
            if dur >= min_dur:
                avg_amp = np.mean(note_dict['amplitudes'])
                velocity = int(np.clip(avg_amp * 100, 30, 120))
                pending_notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_dict['pitch'],
                    start=note_dict['start'],
                    end=note_dict['end'],
                ))

        for i in range(len(f0)):
            time = i * hop / sr
            is_voiced = voiced_flag[i]
            pitch_hz = f0[i]

            if is_voiced and not np.isnan(pitch_hz) and pitch_hz > 0:
                note_num = int(round(librosa.hz_to_midi(pitch_hz)))
                amp = amplitude[i]

                if current_note is None:
                    current_note = {
                        'pitch': note_num,
                        'start': time,
                        'end': time,
                        'amplitudes': [amp],
                    }
                elif abs(note_num - current_note['pitch']) <= tolerance:
                    # 音高在容差范围内 → 延续当前音符
                    current_note['end'] = time
                    current_note['amplitudes'].append(amp)
                else:
                    # 音高变化 → 结束旧音符，开始新音符
                    finalize_note(current_note)
                    current_note = {
                        'pitch': note_num,
                        'start': max(current_note['end'], time - max_gap),
                        'end': time,
                        'amplitudes': [amp],
                    }
            else:
                # 静音/非发声帧
                if current_note is not None:
                    gap = time - current_note['end']
                    if gap > max_gap * 2:
                        # 间隙过大 → 确认音符结束
                        finalize_note(current_note)
                        current_note = None
                    else:
                        # 微小间隙 → 可能是PYIN的瞬时丢失，延长当前音符
                        current_note['end'] = time

        # 处理最后一个音符
        if current_note is not None:
            finalize_note(current_note)

        # ---- 后处理：重叠音符去重 ----
        if pending_notes:
            pending_notes.sort(key=lambda n: n.start)
            merged = [pending_notes[0]]
            for note in pending_notes[1:]:
                prev = merged[-1]
                # 相邻音符：相同音高且间隙极小 → 合并
                if note.pitch == prev.pitch and (note.start - prev.end) < max_gap * 0.5:
                    prev.end = max(prev.end, note.end)
                else:
                    merged.append(note)
            instrument.notes = merged

        if not instrument.notes:
            raise ValueError("无有效音符生成，请检查输入音频")

        midi.instruments.append(instrument)
        return midi

    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()

    def update_config(self, config_updates: Dict[str, Any]):
        self.config.update(config_updates)
