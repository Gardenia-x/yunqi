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
    """MIDI生成器，支持 PYIN 和 Basic Pitch (Spotify) 两种方法"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'sr': 44100,
            'hop_length': 512,
            'min_freq': 65,
            'max_freq': 2000,
            'n_thresholds': 200,
            'resolution': 0.1,
            'harmonic_margin': 3,
            'min_duration': 0.10,
            'max_gap': 0.05,
            'semitone_tolerance': 0,
            'pitch_smooth_kernel': 5,
            'confidence_threshold': 0.3,
            'voicing_threshold': 0.5,
            'preemphasis_coef': 0.97,
            'normalize_axis': 0,
            # 检测方法: 'basic_pitch' (复音，推荐) / 'pyin' (单声部)
            'detection_method': 'basic_pitch',
        }

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        method = self.config.get('detection_method', 'basic_pitch')
        if method == 'basic_pitch':
            return self._process_with_basic_pitch(audio_bytes)
        else:
            return self._process_with_pyin(audio_bytes)

    def _process_with_basic_pitch(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        """使用 Spotify Basic Pitch 进行复音转录"""
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            model_path = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"
            _, midi_data, _ = predict(tmp_path, model_path)

            if not midi_data.instruments or not midi_data.instruments[0].notes:
                raise ValueError("Basic Pitch 未生成有效音符")

            # 后处理：过滤力度过低的音符
            notes = midi_data.instruments[0].notes
            conf = self.config.get('confidence_threshold', 0.3)
            notes = [n for n in notes if n.velocity >= conf * 100]
            midi_data.instruments[0].notes = notes

            return midi_data

        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _process_with_pyin(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
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

            # 音高检测（根据配置选择方法）
            if self.config.get('detection_method') == 'melody':
                f0, voiced_flag, amplitude = self._melody_tracking_detection(y, sr)
            else:
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
        改进的音高检测：PYIN + 谐波增强 + 能量过滤

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
        frame_length = 2 ** int(np.ceil(np.log2(frame_length)))

        # ---- PYIN主检测 ----
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr,
            hop_length=hop, frame_length=frame_length,
            n_thresholds=self.config['n_thresholds'],
            resolution=self.config['resolution'],
            fill_na=np.nan,
        )

        # ---- 计算每帧RMS能量 ----
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        amplitude = np.clip((rms_db + 60) / 60, 0.05, 1.0)

        # ---- 谐波增强检测（作为补充） ----
        try:
            harmonic = librosa.effects.harmonic(y, margin=self.config['harmonic_margin'])
            f0_h, vo_flag_h, _ = librosa.pyin(
                harmonic, fmin=fmin, fmax=fmax, sr=sr,
                hop_length=hop, frame_length=frame_length, fill_na=np.nan,
            )
        except Exception:
            f0_h = np.full_like(f0, np.nan)
            vo_flag_h = np.zeros_like(voiced_flag)

        # ---- 融合：PYIN 优先，低置信度时用谐波检测补充 ----
        conf_thresh = self.config['confidence_threshold']
        use_harmonic = (voiced_prob < conf_thresh) & vo_flag_h
        fused_f0 = np.where(use_harmonic, f0_h, f0)

        # 关键：以实际检测到音高的帧为准
        has_pitch = ~np.isnan(fused_f0)

        # ---- 后处理：插值 + 中值滤波 ----
        if np.any(has_pitch):
            x = np.arange(len(fused_f0))
            fused_f0 = np.interp(x, x[has_pitch], fused_f0[has_pitch])
            kernel = self.config['pitch_smooth_kernel']
            fused_f0 = scipy.signal.medfilt(fused_f0, kernel_size=kernel)

        # ---- 能量辅助：滤除噪声帧 ----
        energy_voiced = rms > np.median(rms) * 0.15
        final_voiced = has_pitch & energy_voiced & (fused_f0 >= fmin) & (fused_f0 <= fmax)

        return fused_f0, final_voiced, amplitude

    def _melody_tracking_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        多声部旋律追踪：piptrack 多候选 + 连续性追踪
        适用于有伴奏的合唱/合奏音频，自动跟踪最连续的旋律线
        """
        hop = self.config['hop_length']
        fmin = self.config['min_freq']
        fmax = self.config['max_freq']

        frame_length = max(2048, int(sr / fmin * 2))
        frame_length = 2 ** int(np.ceil(np.log2(frame_length)))

        # ---- piptrack：每帧返回多个音高候选 ----
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, fmin=fmin, fmax=fmax,
            hop_length=hop, n_fft=frame_length,
        )

        n_candidates, n_frames = pitches.shape
        n_top = min(5, n_candidates)

        # ---- 每帧取 top-N 候选 ----
        top_pitches = np.zeros((n_top, n_frames))
        top_mags = np.zeros((n_top, n_frames))
        for i in range(n_frames):
            col_mag = magnitudes[:, i]
            order = np.argsort(col_mag)[::-1][:n_top]
            top_pitches[:, i] = pitches[order, i]
            top_mags[:, i] = col_mag[order]

        # ---- 计算每帧 RMS 能量 ----
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        amplitude = np.clip((rms_db + 60) / 60, 0.05, 1.0)

        # ---- 旋律追踪：贪婪地跟踪最连续的轮廓 ----
        best_f0 = np.full(n_frames, np.nan)

        # 从能量最强的帧开始（最有可能是旋律音）
        start_frame = int(np.argmax(rms))
        if top_mags[0, start_frame] > 0:
            best_f0[start_frame] = top_pitches[0, start_frame]

        # 向后追踪
        prev_pitch = best_f0[start_frame]
        for i in range(start_frame - 1, -1, -1):
            if top_mags[0, i] <= 0:
                continue
            # 在候选者中找最接近前一音高且幅度较高的
            best_score = np.inf
            best_p = np.nan
            for k in range(n_top):
                p = top_pitches[k, i]
                m = top_mags[k, i]
                if p <= 0:
                    continue
                # 分数 = 音高变化（半音）+ 幅度惩罚
                cents_change = abs(1200 * np.log2(max(p, prev_pitch) / min(p, prev_pitch))) if prev_pitch > 0 else 0
                score = cents_change - 50 * (m / (top_mags[0, i] + 1e-9))
                if score < best_score:
                    best_score = score
                    best_p = p
            if best_score < 200:  # 最大允许 200 音分跳变（2半音）
                best_f0[i] = best_p
                prev_pitch = best_p
            else:
                best_f0[i] = np.nan

        # 向前追踪
        prev_pitch = best_f0[start_frame]
        for i in range(start_frame + 1, n_frames):
            if top_mags[0, i] <= 0:
                continue
            best_score = np.inf
            best_p = np.nan
            for k in range(n_top):
                p = top_pitches[k, i]
                m = top_mags[k, i]
                if p <= 0:
                    continue
                cents_change = abs(1200 * np.log2(max(p, prev_pitch) / min(p, prev_pitch))) if prev_pitch > 0 else 0
                score = cents_change - 50 * (m / (top_mags[0, i] + 1e-9))
                if score < best_score:
                    best_score = score
                    best_p = p
            if best_score < 200:
                best_f0[i] = best_p
                prev_pitch = best_p
            else:
                best_f0[i] = np.nan

        # ---- 后处理：插值 + 中值滤波 ----
        valid = ~np.isnan(best_f0)
        if np.sum(valid) > 0:
            x = np.arange(n_frames)
            best_f0 = np.interp(x, x[valid], best_f0[valid])
            kernel = self.config['pitch_smooth_kernel']
            best_f0 = scipy.signal.medfilt(best_f0, kernel_size=kernel)

        # ---- 发声判断 ----
        energy_voiced = rms > np.median(rms) * 0.15
        has_pitch = ~np.isnan(best_f0)
        final_voiced = has_pitch & energy_voiced & (best_f0 >= fmin) & (best_f0 <= fmax)

        return best_f0, final_voiced, amplitude

    def _create_midi(
        self, f0: np.ndarray, voiced_flag: np.ndarray,
        amplitude: np.ndarray, sr: int
    ) -> pretty_midi.PrettyMIDI:
        """
        MIDI生成：半音容差 + 能量力度 + 间隙容错
        """
        hop = self.config['hop_length']
        min_dur = self.config['min_duration']
        max_gap = self.config['max_gap']
        tolerance = self.config.get('semitone_tolerance', 1)

        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_note = None
        pending_notes = []

        def finalize_note(note_dict):
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
                        'pitch': note_num, 'start': time,
                        'end': time, 'amplitudes': [amp],
                    }
                elif abs(note_num - current_note['pitch']) <= tolerance:
                    current_note['end'] = time
                    current_note['amplitudes'].append(amp)
                else:
                    finalize_note(current_note)
                    current_note = {
                        'pitch': note_num,
                        'start': max(current_note['end'], time - max_gap),
                        'end': time, 'amplitudes': [amp],
                    }
            else:
                if current_note is not None:
                    gap = time - current_note['end']
                    if gap > max_gap * 2:
                        finalize_note(current_note)
                        current_note = None
                    else:
                        current_note['end'] = time

        if current_note is not None:
            finalize_note(current_note)

        if pending_notes:
            pending_notes.sort(key=lambda n: n.start)
            merged = [pending_notes[0]]
            for note in pending_notes[1:]:
                prev = merged[-1]
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
