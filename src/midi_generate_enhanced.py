"""
Enhanced Audio-to-MIDI Conversion System
Improves accuracy with better pitch detection, note segmentation, and configurable parameters
"""

import sys
import os
import tempfile
import traceback
from io import BytesIO
import io
import re
import uuid
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import subprocess
import numpy as np

# Required imports
import scipy
import scipy.signal
import librosa
import pretty_midi
import soundfile as sf
from music21 import environment

# Windows event loop policy
if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    try:
        from asyncio import WindowsSelectorEventLoopPolicy
        import asyncio
        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
    except ImportError:
        pass

# Global configuration
MSCORE_PATH = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"  # MuseScore executable path
DEFAULT_SR = 44100  # Default sample rate
TEMP_DIR_NAME = "Audio2MIDI_TEMP"  # Temporary directory name

# Initialize music21 environment
env = environment.Environment()
env['musescoreDirectPNGPath'] = MSCORE_PATH

# Optional: CREPE for ML-based pitch detection
try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    crepe = None
    CREPE_AVAILABLE = False


class EnhancedMIDIGenerator:
    """
    Enhanced audio-to-MIDI converter with improved accuracy and configurable parameters
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize generator with configuration

        Args:
            config: Dictionary of configuration parameters
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)

        # Validate MuseScore installation (optional, only needed for sheet music)
        try:
            self._validate_musescore()
            self.musescore_available = True
        except Exception:
            self.musescore_available = False
            # MuseScore is optional - silent failure since sheet music generation is not required

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters"""
        return {
            # Audio processing
            'sr': 44100,
            'hop_length': 256,
            'frame_length': 2048,

            # Frequency range
            'min_freq': 200,
            'max_freq': 1500,

            # Pitch detection
            'pitch_method': 'hybrid',  # 'pyin', 'crepe', or 'hybrid'
            'n_thresholds': 200,
            'resolution': 0.1,
            'fill_na': np.nan,

            # Voicing detection
            'voicing_threshold': 0.3,  # Lower threshold to detect more notes, especially higher pitches
            'confidence_weight': 0.8,  # Slightly higher confidence weighting
            'use_harmonic_enhancement': True,
            'harmonic_margin': 6,

            # Post-processing
            'median_filter_size': 5,
            'smoothing_window': 3,
            'interpolate_missing': True,

            # Note segmentation
            'onset_threshold': 0.3,  # Higher threshold to reduce false onsets
            'min_note_duration': 0.04,  # Reduced to detect more short notes (47% of reference notes are <0.3s)
            'max_gap': 0.02,  # Reduced to prevent merging of short notes
            'min_velocity': 40,
            'max_velocity': 127,

            # Polyphony (experimental)
            'max_polyphony': 1,  # 1 for monophonic, >1 for polyphonic
            'polyphony_threshold': 0.3,
        }

    def _validate_musescore(self):
        """Validate MuseScore installation and version"""
        if not Path(MSCORE_PATH).exists():
            raise FileNotFoundError(f"MuseScore not found: {MSCORE_PATH}")

        try:
            result = subprocess.run(
                [MSCORE_PATH, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
                encoding='gbk',
                errors='replace'
            )
            output = result.stdout

            # Version matching
            match = re.search(r'MuseScore[ _]?4[^\d]*(\d+\.\d+\.\d+)', output)
            if not match or int(match.group(1).split('.')[0]) != 4:
                raise ValueError(f"Requires MuseScore 4.x, detected version: {match.group(1) if match else 'unknown'}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MuseScore execution failed:\nExit code: {e.returncode}\nOutput: {e.output}")

    def process_audio(self, audio_bytes: bytes) -> pretty_midi.PrettyMIDI:
        """
        Process audio and generate MIDI with enhanced algorithms

        Args:
            audio_bytes: Raw audio file bytes

        Returns:
            PrettyMIDI object
        """
        tmp_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            # Load audio
            y, sr = librosa.load(tmp_path, sr=self.config['sr'], mono=True)

            # Preprocessing
            y = self._preprocess_audio(y)

            # Pitch detection
            f0, voiced_flag = self._enhanced_pitch_detection(y, sr)

            # Post-processing
            f0, voiced_flag = self._post_process_pitch(f0, voiced_flag)

            # Note segmentation with onset detection
            midi = self._create_enhanced_midi(f0, voiced_flag, sr, y)

            # Validate MIDI
            if not midi.instruments or not midi.instruments[0].notes:
                raise ValueError("Generated MIDI contains no valid notes")

            return midi

        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}") from e
        finally:
            # Cleanup temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """Audio preprocessing chain"""
        # Pre-emphasis
        y = librosa.effects.preemphasis(y, coef=0.97)

        # Normalize
        y = librosa.util.normalize(y, axis=0)

        # Ensure proper length for analysis
        target_length = ((len(y) // self.config['hop_length']) + 1) * self.config['hop_length']
        y = librosa.util.fix_length(y, size=target_length, mode='wrap')

        return y

    def _enhanced_pitch_detection(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced pitch detection with multiple methods and fusion
        """
        method = self.config['pitch_method']

        if method == 'crepe' and CREPE_AVAILABLE:
            f0, confidence = self._pitch_crepe(y, sr)
            voiced_flag = confidence > self.config['voicing_threshold']

        elif method == 'hybrid':
            f0_pyin, voiced_pyin = self._pitch_pyin(y, sr)
            if CREPE_AVAILABLE:
                f0_crepe, confidence_crepe = self._pitch_crepe(y, sr)
                # Fuse results
                alpha = self.config['confidence_weight']
                f0 = alpha * f0_pyin + (1 - alpha) * f0_crepe
                voiced_flag = (voiced_pyin & (confidence_crepe > self.config['voicing_threshold']))
            else:
                f0, voiced_flag = f0_pyin, voiced_pyin

        else:  # 'pyin' as default
            f0, voiced_flag = self._pitch_pyin(y, sr)

        return f0, voiced_flag

    def _pitch_pyin(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """PYIN pitch detection with harmonic enhancement"""
        # Main PYIN detection
        fmin = max(20, self.config['min_freq'])
        fmax = min(20000, self.config['max_freq'])

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=self.config['hop_length'],
            frame_length=self.config['frame_length'],
            n_thresholds=self.config['n_thresholds'],
            resolution=self.config['resolution'],
            fill_na=self.config['fill_na']
        )

        # Harmonic enhancement if enabled
        if self.config['use_harmonic_enhancement']:
            try:
                harmonic = librosa.effects.harmonic(y, margin=self.config['harmonic_margin'])
                f0_harmonic, _, _ = librosa.pyin(
                    harmonic,
                    fmin=fmin,
                    fmax=fmax,
                    sr=sr,
                    hop_length=self.config['hop_length'],
                    frame_length=self.config['frame_length'],
                    fill_na=np.nan
                )

                # Confidence-based fusion
                S, _ = librosa.magphase(librosa.stft(y, hop_length=self.config['hop_length']))
                confidence = librosa.feature.spectral_flatness(S=S).squeeze()
                confidence = librosa.util.normalize(confidence, axis=0)

                alpha = np.clip(confidence, 0.2, 0.8)
                f0 = alpha * np.nan_to_num(f0, nan=0) + (1 - alpha) * np.nan_to_num(f0_harmonic, nan=0)
            except Exception:
                pass  # Fall back to original f0 if harmonic processing fails

        # Convert NaN to 0 for unvoiced frames
        f0 = np.nan_to_num(f0, nan=0)
        voiced_flag = voiced_flag.astype(bool)

        return f0, voiced_flag

    def _pitch_crepe(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """CREPE neural network pitch detection"""
        if not CREPE_AVAILABLE:
            raise ImportError("CREPE not available. Install with: pip install crepe")

        # Resample to 16kHz for CREPE if needed
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Run CREPE
        time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True)

        # Resample back to original hop length
        target_times = np.arange(0, len(y)/sr, self.config['hop_length']/sr)
        f0 = np.interp(target_times, time, frequency, left=0, right=0)
        conf = np.interp(target_times, time, confidence, left=0, right=0)

        # Convert to Hz (CREPE returns frequency in Hz already)
        return f0, conf

    def _post_process_pitch(self, f0: np.ndarray, voiced_flag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process pitch contour"""
        # Median filtering
        if self.config['median_filter_size'] > 1:
            f0 = scipy.signal.medfilt(f0, kernel_size=self.config['median_filter_size'])

        # Smoothing
        if self.config['smoothing_window'] > 1:
            window = np.ones(self.config['smoothing_window']) / self.config['smoothing_window']
            f0 = np.convolve(f0, window, mode='same')

        # Interpolate missing values
        if self.config['interpolate_missing']:
            x = np.arange(len(f0))
            valid_mask = (f0 > 0) & (f0 >= self.config['min_freq']) & (f0 <= self.config['max_freq'])
            if np.any(valid_mask):
                f0 = np.interp(x, x[valid_mask], f0[valid_mask])
            else:
                f0[:] = 0

        # Update voicing based on processed pitch
        voiced_flag = (f0 > 0) & (f0 >= self.config['min_freq']) & (f0 <= self.config['max_freq'])

        return f0, voiced_flag

    def _create_enhanced_midi(self, f0: np.ndarray, voiced_flag: np.ndarray,
                             sr: int, audio: np.ndarray) -> pretty_midi.PrettyMIDI:
        """Create MIDI with enhanced note segmentation and velocity estimation"""
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        # Detect onsets for better note segmentation
        onsets = self._detect_onsets(audio, sr)

        # Estimate velocity from audio amplitude
        velocity_profile = self._estimate_velocity(audio, sr)

        # Segment notes using combined information
        notes = self._segment_notes(f0, voiced_flag, sr, onsets, velocity_profile)

        # Add notes to instrument
        for note_info in notes:
            midi_note = pretty_midi.Note(
                velocity=int(note_info['velocity']),
                pitch=note_info['pitch'],
                start=note_info['start'],
                end=note_info['end']
            )
            instrument.notes.append(midi_note)

        midi.instruments.append(instrument)
        return midi

    def _detect_onsets(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Detect note onsets in audio with enhanced sensitivity for short notes"""
        try:
            # 使用复合onset检测：结合能量和频谱特征
            onset_strength = librosa.onset.onset_strength(
                y=audio,
                sr=sr,
                hop_length=self.config['hop_length'],
                aggregate=np.median
            )

            # 确保onset强度非空且有效
            if len(onset_strength) == 0 or np.all(np.isnan(onset_strength)):
                raise ValueError("Onset strength computation failed")

            # 自适应阈值：使用动态阈值提高灵敏度
            threshold_value = np.percentile(onset_strength[~np.isnan(onset_strength)], 60)
            threshold_value = max(threshold_value * 0.8, onset_strength.mean() * 0.3)
            threshold_value = min(threshold_value, self.config['onset_threshold'] * 2)

            # 检测onset帧，使用更敏感的参数
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_strength,
                sr=sr,
                hop_length=self.config['hop_length'],
                backtrack=True,
                threshold=threshold_value,
                pre_max=2,
                post_max=2,
                pre_avg=2,
                post_avg=2,
                delta=0.03,  # 更小的变化阈值
                wait=1       # 允许更密集的onset
            )

            # 转换为时间
            onset_times = librosa.frames_to_time(
                onset_frames,
                sr=sr,
                hop_length=self.config['hop_length']
            )

            # 过滤掉过于接近的onset（10ms内）
            if len(onset_times) > 1:
                filtered_times = [onset_times[0]]
                for t in onset_times[1:]:
                    if t - filtered_times[-1] >= 0.01:  # 10ms最小间隔
                        filtered_times.append(t)
                onset_times = np.array(filtered_times)

            return onset_times
        except Exception as e:
            # 回退：使用基本的onset检测
            try:
                onset_frames = librosa.onset.onset_detect(
                    y=audio,
                    sr=sr,
                    hop_length=self.config['hop_length'],
                    backtrack=True,
                    threshold=self.config['onset_threshold'] * 0.7  # 更低的阈值
                )
                onset_times = librosa.frames_to_time(
                    onset_frames,
                    sr=sr,
                    hop_length=self.config['hop_length']
                )
                return onset_times
            except Exception:
                # 最终回退：空数组
                return np.array([])

    def _estimate_velocity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Estimate MIDI velocity from audio amplitude"""
        # Compute RMS energy per frame
        frame_length = self.config['frame_length']
        hop_length = self.config['hop_length']

        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        ).squeeze()

        # Normalize and map to MIDI velocity range
        if len(rms) > 0:
            rms_min, rms_max = rms.min(), rms.max()
            if rms_max > rms_min:
                normalized = (rms - rms_min) / (rms_max - rms_min)
            else:
                normalized = np.ones_like(rms)
        else:
            normalized = np.array([0.5])

        # Map to MIDI velocity range
        velocity = self.config['min_velocity'] + normalized * (
            self.config['max_velocity'] - self.config['min_velocity']
        )

        return velocity

    def _segment_notes(self, f0: np.ndarray, voiced_flag: np.ndarray, sr: int,
                      onsets: np.ndarray, velocity_profile: np.ndarray) -> List[Dict]:
        """
        Enhanced note segmentation with energy-based detection and improved short note handling
        """
        notes = []
        hop_length = self.config['hop_length']
        min_duration = self.config['min_note_duration']
        max_gap = self.config['max_gap']

        current_note = None
        frame_times = np.arange(len(f0)) * hop_length / sr

        # 计算能量变化率（用于检测音符结束）
        energy = velocity_profile
        energy_diff = np.zeros_like(energy)
        if len(energy) > 1:
            energy_diff[1:] = energy[1:] - energy[:-1]

        # 滑动窗口用于音高稳定性检测
        pitch_history = []
        max_history_len = max(3, int(0.05 * sr / hop_length))  # 最多0.05秒的历史

        for i in range(len(f0)):
            time = frame_times[i]
            is_voiced = voiced_flag[i]
            pitch_hz = f0[i] if is_voiced else 0

            # Convert to MIDI pitch
            if pitch_hz > 0:
                pitch_midi = int(round(librosa.hz_to_midi(pitch_hz)))
            else:
                pitch_midi = None

            # Check if this frame aligns with an onset
            is_onset = False
            if len(onsets) > 0:
                is_onset = np.any(np.abs(onsets - time) < (hop_length / sr))

            # 能量下降检测：如果能量显著下降，可能表示音符结束
            energy_drop = False
            if i > 0 and energy_diff[i] < -0.15 * energy[i-1] if energy[i-1] > 0 else False:
                # 能量下降超过15%
                energy_drop = True

            # 处理音符分割
            if pitch_midi is not None:
                if current_note is None:
                    # 开始新音符
                    current_note = {
                        'pitch': pitch_midi,
                        'start': time,
                        'end': time,
                        'velocity_frames': [],
                        'pitch_history': [pitch_midi]
                    }
                else:
                    # 检查音高变化和能量变化
                    pitch_diff = abs(pitch_midi - current_note['pitch'])

                    # 音高稳定性检测：检查最近音高历史
                    current_note['pitch_history'].append(pitch_midi)
                    if len(current_note['pitch_history']) > max_history_len:
                        current_note['pitch_history'].pop(0)

                    # 计算音高标准差（如果历史足够）
                    if len(current_note['pitch_history']) >= 3:
                        pitch_std = np.std(current_note['pitch_history'])
                        pitch_stable = pitch_std < 2.0  # 标准差小于2.0半音视为稳定
                    else:
                        pitch_stable = True

                    # 判断是否继续当前音符
                    continue_note = False
                    if pitch_stable and pitch_diff <= 2 and not is_onset:  # 放宽到±2半音
                        # 音高稳定且变化小，继续音符
                        continue_note = True

                    # 即使音高变化小，但能量显著下降，也结束音符
                    if energy_drop and not is_onset:
                        # 能量下降但无onset，可能是音符自然衰减
                        if pitch_diff <= 2:  # 音高变化小
                            continue_note = True  # 仍继续音符
                        else:
                            continue_note = False  # 音高变化大，结束音符

                    if continue_note:
                        current_note['end'] = time
                    else:
                        # 音高变化超出容差、检测到onset或能量下降
                        if current_note['end'] - current_note['start'] >= min_duration:
                            # 完成前一个音符
                            note_data = self._finalize_note(current_note, velocity_profile, frame_times)
                            notes.append(note_data)
                        elif current_note['end'] - current_note['start'] >= min_duration * 0.5:
                            # 短音符但长度超过min_duration的一半
                            # 尝试与上一个音符合并（如果音高相同且时间接近）
                            if notes and notes[-1]['pitch'] == current_note['pitch']:
                                last_note = notes[-1]
                                gap = current_note['start'] - last_note['end']
                                if gap < max_gap * 0.5:  # 间隙很小
                                    # 合并音符
                                    last_note['end'] = current_note['end']
                                    # 更新velocity（重新计算）
                                    start_idx = np.searchsorted(frame_times, last_note['start'])
                                    end_idx = np.searchsorted(frame_times, last_note['end'])
                                    if end_idx > start_idx and start_idx < len(velocity_profile):
                                        note_velocities = velocity_profile[start_idx:min(end_idx, len(velocity_profile))]
                                        last_note['velocity'] = int(np.mean(note_velocities))
                                    continue

                        # 开始新音符
                        current_note = {
                            'pitch': pitch_midi,
                            'start': time,
                            'end': time,
                            'velocity_frames': [],
                            'pitch_history': [pitch_midi]
                        }
            else:
                # 非浊音区域
                if current_note is not None:
                    gap = time - current_note['end']
                    if gap > max_gap:
                        # 音符结束
                        if current_note['end'] - current_note['start'] >= min_duration:
                            note_data = self._finalize_note(current_note, velocity_profile, frame_times)
                            notes.append(note_data)
                        current_note = None

        # 处理最后一个音符
        if current_note is not None and current_note['end'] - current_note['start'] >= min_duration:
            note_data = self._finalize_note(current_note, velocity_profile, frame_times)
            notes.append(note_data)

        # 后处理：合并相同音高且间隙很小的相邻音符
        notes = self._merge_adjacent_notes(notes, max_gap * 0.3)

        return notes

    def _merge_adjacent_notes(self, notes: List[Dict], max_gap: float) -> List[Dict]:
        """Merge adjacent notes with same pitch and small gap"""
        if len(notes) <= 1:
            return notes

        merged = []
        current = notes[0].copy()

        for i in range(1, len(notes)):
            next_note = notes[i]

            # 检查是否可合并：相同音高且间隙小
            if (current['pitch'] == next_note['pitch'] and
                next_note['start'] - current['end'] < max_gap):
                # 合并音符：扩展结束时间，重新计算velocity平均值
                current['end'] = next_note['end']
                # 合并后的velocity取平均值
                current['velocity'] = (current['velocity'] + next_note['velocity']) // 2
            else:
                merged.append(current)
                current = next_note.copy()

        merged.append(current)
        return merged

    def _finalize_note(self, note: Dict, velocity_profile: np.ndarray,
                      frame_times: np.ndarray) -> Dict:
        """Finalize note with computed velocity"""
        # Find frames corresponding to this note
        start_idx = np.searchsorted(frame_times, note['start'])
        end_idx = np.searchsorted(frame_times, note['end'])

        # Average velocity over note duration
        if end_idx > start_idx and start_idx < len(velocity_profile):
            note_velocities = velocity_profile[start_idx:min(end_idx, len(velocity_profile))]
            velocity = int(np.mean(note_velocities))
        else:
            velocity = 100  # Default

        return {
            'pitch': note['pitch'],
            'start': note['start'],
            'end': note['end'],
            'velocity': velocity
        }

    def generate_sheet_music(self, midi_data: pretty_midi.PrettyMIDI,
                            format_type: str = 'png') -> bytes:
        """
        Generate sheet music from MIDI using MuseScore

        Args:
            midi_data: PrettyMIDI object
            format_type: 'png' or 'pdf'

        Returns:
            Sheet music image bytes
        """
        temp_dir = Path(os.environ.get('MUSIC21_TEMPDIR', tempfile.gettempdir()))
        uid = uuid.uuid4().hex[:8]

        midi_path = temp_dir / f"input_{uid}.mid"
        output_path = temp_dir / f"output_{uid}.{format_type}"

        try:
            # Write MIDI file
            midi_data.write(str(midi_path))

            # Build conversion command
            cmd = [
                MSCORE_PATH,
                "--force",
                midi_path.resolve().as_posix(),
                "--export-to",
                output_path.resolve().as_posix()
            ]

            if format_type == 'png':
                cmd.extend(["-T", "1"])  # 300 DPI

            # Execute conversion
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=True,
                encoding='utf-8',
                errors='replace'
            )

            # Verify output
            if not output_path.exists():
                raise FileNotFoundError(f"Output file not generated: {output_path}")

            return output_path.read_bytes()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"MuseScore conversion failed: {e.output}") from e
        except Exception as e:
            raise RuntimeError(f"Sheet music generation failed: {str(e)}") from e
        finally:
            # Cleanup
            for f in [midi_path, output_path]:
                if f and f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass


# Utility functions for backward compatibility
def enhanced_process_audio(audio_bytes: bytes, **kwargs) -> pretty_midi.PrettyMIDI:
    """
    Convenience function for enhanced audio processing

    Args:
        audio_bytes: Raw audio file bytes
        **kwargs: Configuration parameters

    Returns:
        PrettyMIDI object
    """
    generator = EnhancedMIDIGenerator(kwargs)
    return generator.process_audio(audio_bytes)


def main():
    """Test the enhanced MIDI generator"""
    print("Enhanced MIDI Generator Test")
    print("=" * 50)

    # Example configuration
    config = {
        'sr': 44100,
        'hop_length': 512,
        'min_freq': 80,
        'max_freq': 1200,
        'pitch_method': 'pyin',
        'use_harmonic_enhancement': True,
    }

    generator = EnhancedMIDIGenerator(config)
    print(f"Generator initialized with config: {config}")
    print(f"CREPE available: {CREPE_AVAILABLE}")

    # Note: Need an audio file to test
    print("\nReady to process audio files.")
    print("Usage:")
    print("  generator = EnhancedMIDIGenerator(config)")
    print("  midi = generator.process_audio(audio_bytes)")
    print("  sheet_music = generator.generate_sheet_music(midi, 'png')")


if __name__ == "__main__":
    main()