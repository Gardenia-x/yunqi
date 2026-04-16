"""
MIDI Score Evaluation System v4.5
English Visualization Edition
Time-Pitch Analysis Enhanced
"""

import numpy as np
import bisect
import functools
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Optional, Union, Any
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tabulate import tabulate
import matplotlib.pyplot as plt
from music21 import converter, note, chord, key, stream, voiceLeading

# Optional PyTorch import for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Type definitions for MIDI elements
class NoteDict(TypedDict, total=False):
    pitch: int
    start: float
    end: float
    velocity: int
    is_rest: bool
    is_chord: bool
    chord_position: int
    track: int

# Configure visualization settings
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150
})


class ScoreEvaluator:
    def __init__(self, reference_midi: Path, test_midi: Path):
        self._validate_file(reference_midi, "Reference")
        self._validate_file(test_midi, "Test")

        try:
            print(f"\n[Parsing] {reference_midi.name}...")
            self.ref_notes = self._parse_midi(reference_midi)
            self.ref_onsets = self._get_valid_onsets(self.ref_notes)
            print(f"Valid notes: {len(self.ref_onsets)}")

            print(f"\n[Parsing] {test_midi.name}...")
            self.test_notes = self._parse_midi(test_midi)
            self.test_onsets = self._get_valid_onsets(self.test_notes)
            print(f"Valid notes: {len(self.test_onsets)}")

            self._check_data_validity()
            self.test_midi = test_midi

            # Initialize caches
            self._accuracy_cache: Dict[Tuple[float, int], Dict[str, Any]] = {}
            self._rhythm_cache: Optional[Dict[str, Any]] = None
            self._ref_notes_without_rest: Optional[List[NoteDict]] = None
            self._test_notes_without_rest: Optional[List[NoteDict]] = None

            # Pre-computed numpy arrays for faster access
            self._ref_starts_np: Optional[np.ndarray] = None
            self._ref_pitches_np: Optional[np.ndarray] = None
            self._test_starts_np: Optional[np.ndarray] = None
            self._test_pitches_np: Optional[np.ndarray] = None

            # Method selection: 0=original, 1=optimized
            self._use_optimized_method = True

        except Exception as e:
            print(f"\033[31mInitialization failed: {str(e)}\033[0m")
            raise

    def _validate_file(self, path: Path, file_type: str):
        """Enhanced file validation"""
        checks = [
            (not path.exists(), f"{file_type} file missing: {path}"),
            (path.suffix.lower() != '.mid', f"Warning: Non-standard MIDI format ({path.suffix})"),
            (path.stat().st_size < 1024, f"Warning: Small file size ({path.stat().st_size} bytes)")
        ]
        for condition, message in checks:
            if condition:
                print(f"\033[33m{message}\033[0m")

    def _parse_midi(self, midi_path: Path) -> List[NoteDict]:
        """MIDI parser with error handling"""
        try:
            midi = converter.parse(str(midi_path))
            elements = []
            rest_count = 0

            for elem in midi.flatten().notesAndRests:
                if isinstance(elem, note.Rest):
                    elements.append(self._create_rest_dict(elem))
                    rest_count += 1
                elif isinstance(elem, note.Note):
                    elements.append(self._create_note_dict(elem))
                elif isinstance(elem, chord.Chord):
                    elements.extend(self._process_chord(elem))
                else:
                    print(f"Skipping unprocessed element: {elem.classes[0]}")

            print(f"Parsed: {len(elements)} elements (Rests: {rest_count})")
            return sorted(elements, key=lambda x: x['start'])

        except Exception as e:
            print(f"\033[31mParsing error {midi_path.name}: {str(e)}\033[0m")
            return []

    def _process_chord(self, element) -> List[NoteDict]:
        """Chord processing method"""
        chord_notes = []
        duration = float(element.duration.quarterLength)
        base_dict = {
            'start': float(element.offset),
            'end': float(element.offset) + duration,
            'velocity': element.volume.velocity if hasattr(element.volume, 'velocity') else 64,
            'is_rest': False,
            'is_chord': True,
            'track': element.activeSite.id if hasattr(element.activeSite, 'id') else 0
        }

        for idx, p in enumerate(sorted(element.pitches, key=lambda x: x.midi)):
            note_dict = base_dict.copy()
            note_dict['pitch'] = p.midi
            note_dict['chord_position'] = idx + 1
            chord_notes.append(note_dict)
        return chord_notes

    def _create_note_dict(self, element) -> NoteDict:
        return {
            'pitch': element.pitch.midi,
            'start': float(element.offset),
            'end': float(element.offset + element.duration.quarterLength),
            'velocity': element.volume.velocity if hasattr(element.volume, 'velocity') else 64,
            'is_rest': False,
            'is_chord': False,
            'track': element.activeSite.id if hasattr(element.activeSite, 'id') else 0
        }

    def _create_rest_dict(self, element) -> NoteDict:
        return {
            'start': float(element.offset),
            'end': float(element.offset + element.duration.quarterLength),
            'is_rest': True,
            'is_chord': False,
            'track': element.activeSite.id if hasattr(element.activeSite, 'id') else 0
        }

    def _check_data_validity(self):
        """Data validation"""
        errors = []
        if not self.ref_notes:
            errors.append("Empty reference MIDI data")
        if not self.test_notes:
            errors.append("Empty test MIDI data")
        if len(self.ref_onsets) < 5:
            errors.append("Insufficient reference notes (<5)")
        if len(self.test_onsets) < 5:
            errors.append("Insufficient test notes (<5)")

        if errors:
            raise ValueError("\n".join([f"\033[31m{err}\033[0m" for err in errors]))

    def note_accuracy(self, onset_tol: float = 0., pitch_tol: int = 3) -> Dict[str, Any]:
        """Note matching algorithm with caching - automatically uses optimized version if enabled"""
        # Use optimized method if enabled
        if self._use_optimized_method:
            return self.note_accuracy_optimized(onset_tol, pitch_tol)

        # Check cache first
        cache_key = (onset_tol, pitch_tol)
        if cache_key in self._accuracy_cache:
            return self._accuracy_cache[cache_key].copy()  # Return copy to avoid mutation

        # Cache non-rest notes if not already cached
        if self._ref_notes_without_rest is None:
            self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
        if self._test_notes_without_rest is None:
            self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]

        ref_notes = self._ref_notes_without_rest
        test_notes = self._test_notes_without_rest

        TP, FP, FN = 0, 0, 0
        ref_matched = [False] * len(ref_notes)
        test_matched = [False] * len(test_notes)

        # Forward matching
        for ti, t_note in enumerate(test_notes):
            candidates = []
            left = bisect.bisect_left(self.ref_onsets, t_note['start'] - onset_tol)
            right = bisect.bisect_right(self.ref_onsets, t_note['start'] + onset_tol)

            for ri in range(left, min(right + 1, len(ref_notes))):
                if not ref_matched[ri]:
                    r_note = ref_notes[ri]
                    time_diff = abs(t_note['start'] - r_note['start'])
                    pitch_diff = abs(t_note['pitch'] - r_note['pitch'])
                    if time_diff <= onset_tol and pitch_diff <= pitch_tol:
                        candidates.append((ri, time_diff + pitch_diff * 0.1))

            if candidates:
                best_match = min(candidates, key=lambda x: x[1])
                ref_matched[best_match[0]] = True
                test_matched[ti] = True
                TP += 1

        # Backward matching
        for ri, r_note in enumerate(ref_notes):
            if not ref_matched[ri]:
                left = bisect.bisect_left(self.test_onsets, r_note['start'] - onset_tol)
                right = bisect.bisect_right(self.test_onsets, r_note['start'] + onset_tol)

                for ti in range(left, min(right + 1, len(test_notes))):
                    if not test_matched[ti]:
                        t_note = test_notes[ti]
                        time_diff = abs(r_note['start'] - t_note['start'])
                        pitch_diff = abs(r_note['pitch'] - t_note['pitch'])
                        if time_diff <= onset_tol and pitch_diff <= pitch_tol:
                            test_matched[ti] = True
                            ref_matched[ri] = True
                            TP += 1
                            break

        FP = sum([1 for m in test_matched if not m])
        FN = sum([1 for m in ref_matched if not m])

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        result = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'onset_tol': onset_tol,
            'pitch_tol': pitch_tol
        }

        # Cache result
        self._accuracy_cache[cache_key] = result.copy()
        return result

    def _get_numpy_arrays(self):
        """Get or create numpy arrays for fast vectorized operations"""
        if self._ref_starts_np is None or self._ref_pitches_np is None:
            if self._ref_notes_without_rest is None:
                self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
            self._ref_starts_np = np.array([n['start'] for n in self._ref_notes_without_rest])
            self._ref_pitches_np = np.array([n['pitch'] for n in self._ref_notes_without_rest])

        if self._test_starts_np is None or self._test_pitches_np is None:
            if self._test_notes_without_rest is None:
                self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]
            self._test_starts_np = np.array([n['start'] for n in self._test_notes_without_rest])
            self._test_pitches_np = np.array([n['pitch'] for n in self._test_notes_without_rest])

    def note_accuracy_optimized(self, onset_tol: float = 0., pitch_tol: int = 3) -> Dict[str, Any]:
        """Optimized note matching algorithm using vectorized operations and sliding window"""
        # Check cache first (use same cache as original method)
        cache_key = (onset_tol, pitch_tol)
        if cache_key in self._accuracy_cache:
            return self._accuracy_cache[cache_key].copy()

        # Cache non-rest notes if not already cached
        if self._ref_notes_without_rest is None:
            self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
        if self._test_notes_without_rest is None:
            self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]

        ref_notes = self._ref_notes_without_rest
        test_notes = self._test_notes_without_rest

        n_ref = len(ref_notes)
        n_test = len(test_notes)

        # Extract data to numpy arrays
        ref_starts = np.array([n['start'] for n in ref_notes])
        ref_pitches = np.array([n['pitch'] for n in ref_notes])
        test_starts = np.array([n['start'] for n in test_notes])
        test_pitches = np.array([n['pitch'] for n in test_notes])

        # Initialize matching arrays
        ref_matched = np.zeros(n_ref, dtype=bool)
        test_matched = np.zeros(n_test, dtype=bool)
        TP = 0

        # Use sliding window approach with two pointers
        # Since both onset arrays are sorted, we can use a more efficient algorithm

        # First pass: match test notes to reference notes
        ref_idx = 0
        test_idx = 0

        while test_idx < n_test and ref_idx < n_ref:
            if test_matched[test_idx]:
                test_idx += 1
                continue

            t_start = test_starts[test_idx]

            # Advance ref_idx to the start of the window
            while ref_idx < n_ref and ref_starts[ref_idx] < t_start - onset_tol:
                ref_idx += 1

            # Check potential matches in the window
            window_start = ref_idx
            best_match = -1
            best_score = float('inf')

            # Check reference notes within the time window
            j = window_start
            while j < n_ref and ref_starts[j] <= t_start + onset_tol:
                if not ref_matched[j]:
                    time_diff = abs(t_start - ref_starts[j])
                    pitch_diff = abs(test_pitches[test_idx] - ref_pitches[j])

                    if time_diff <= onset_tol and pitch_diff <= pitch_tol:
                        score = time_diff + pitch_diff * 0.01  # Small weight for pitch
                        if score < best_score:
                            best_score = score
                            best_match = j
                j += 1

            if best_match != -1:
                # Found a match
                ref_matched[best_match] = True
                test_matched[test_idx] = True
                TP += 1

            test_idx += 1

        # Second pass: match remaining reference notes to test notes
        ref_idx = 0
        test_idx = 0

        while ref_idx < n_ref and test_idx < n_test:
            if ref_matched[ref_idx]:
                ref_idx += 1
                continue

            r_start = ref_starts[ref_idx]

            # Advance test_idx to the start of the window
            while test_idx < n_test and test_starts[test_idx] < r_start - onset_tol:
                test_idx += 1

            # Check potential matches in the window
            window_start = test_idx
            found_match = False

            # Check test notes within the time window
            j = window_start
            while j < n_test and test_starts[j] <= r_start + onset_tol:
                if not test_matched[j]:
                    time_diff = abs(r_start - test_starts[j])
                    pitch_diff = abs(ref_pitches[ref_idx] - test_pitches[j])

                    if time_diff <= onset_tol and pitch_diff <= pitch_tol:
                        test_matched[j] = True
                        ref_matched[ref_idx] = True
                        TP += 1
                        found_match = True
                        break
                j += 1

            ref_idx += 1

        # Calculate metrics
        FP = np.sum(~test_matched)
        FN = np.sum(~ref_matched)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        result = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'onset_tol': onset_tol,
            'pitch_tol': pitch_tol
        }

        # Cache result
        self._accuracy_cache[cache_key] = result.copy()
        return result

    def rhythm_analysis(self) -> Dict[str, Any]:
        """Rhythm analysis using DTW with caching"""
        if self._rhythm_cache is not None:
            return self._rhythm_cache.copy()

        # Use cached non-rest notes if available
        if self._ref_notes_without_rest is None:
            self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
        if self._test_notes_without_rest is None:
            self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]

        # Convert to numpy arrays for vectorized operations
        ref_arr = np.array([[n['start'], n['end'] - n['start']] for n in self._ref_notes_without_rest])
        test_arr = np.array([[n['start'], n['end'] - n['start']] for n in self._test_notes_without_rest])

        # Choose distance function based on GPU availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Transfer data to GPU once
            device = torch.device('cuda')
            ref_tensor = torch.tensor(ref_arr, device=device, dtype=torch.float32)
            test_tensor = torch.tensor(test_arr, device=device, dtype=torch.float32)

            def torch_dist(x, y):
                # x and y are 1D numpy arrays from fastdtw
                # Convert to torch tensors
                x_t = torch.tensor(x, device=device, dtype=torch.float32)
                y_t = torch.tensor(y, device=device, dtype=torch.float32)
                return torch.cdist(x_t.unsqueeze(0), y_t.unsqueeze(0)).item()

            distance, path = fastdtw(ref_arr, test_arr, dist=torch_dist)
            print("[DTW] Using GPU acceleration")
        else:
            distance, path = fastdtw(ref_arr, test_arr, dist=euclidean)
            if TORCH_AVAILABLE:
                print("[DTW] GPU available but not used (CUDA not detected)")

        alignment = np.array(path)

        # Vectorized timing deviations calculation
        if len(path) > 0:
            ref_indices = np.array([i for i, j in path])
            test_indices = np.array([j for i, j in path])
            time_diffs = np.abs(ref_arr[ref_indices, 0] - test_arr[test_indices, 0])
        else:
            time_diffs = np.array([])

        # Handle empty time_diffs case
        if len(time_diffs) == 0:
            avg_dev = 0.0
            max_dev = 0.0
        else:
            avg_dev = round(float(np.mean(time_diffs)), 3)
            max_dev = round(float(np.max(time_diffs)), 3)

        result = {
            'dtw_distance': distance,
            'avg_time_deviation': avg_dev,
            'max_deviation': max_dev,
            'alignment_path': alignment
        }

        self._rhythm_cache = result.copy()
        return result

    def generate_text_report(self) -> str:
        """Generate text report"""
        accuracy = self.note_accuracy()
        rhythm = self.rhythm_analysis()

        report = [
            ["Note Accuracy", f"Precision: {accuracy['precision']}", f"Recall: {accuracy['recall']}",
             f"F1: {accuracy['f1']}"],
            ["Rhythm Analysis",
             f"DTW Distance: {rhythm['dtw_distance']:.1f}",
             f"Avg Deviation: {rhythm['avg_time_deviation']} beats",
             f"Max Deviation: {rhythm['max_deviation']} beats"],
            ["Tolerance Settings",
             f"Time Tolerance: ±{accuracy['onset_tol']} beats",
             f"Pitch Tolerance: ±{accuracy['pitch_tol']} semitones"]
        ]

        return tabulate(report,
                        headers=["Metric", "Param1", "Param2", "Param3"],
                        tablefmt="grid")

    def generate_visual_report(self, dpi: int = 200, output_dir: Optional[Union[str, Path]] = None):
        """Generate visual analysis report with optimized memory usage"""
        plt.figure(figsize=(12, 8))

        try:
            # Rhythm alignment visualization
            plt.subplot(2, 2, 1)
            self._plot_rhythm_alignment()

            # Pitch vs Time comparison
            plt.subplot(2, 2, 2)
            self._plot_pitch_time_comparison()

            # Velocity analysis
            plt.subplot(2, 2, 3)
            self._plot_velocity_analysis()

            # Save figure with optimized settings
            plt.tight_layout(pad=2.0)
            filename = f"{self.test_midi.stem}_analysis_en.png"

            # 如果指定了输出目录，则使用完整路径
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                filepath = output_dir / filename
            else:
                filepath = Path(filename)

            plt.savefig(str(filepath), bbox_inches='tight', dpi=dpi, optimize=True,
                       metadata={'Creator': 'MIDI Score Evaluator'})
            print(f"\nVisual report saved: {filepath} (DPI: {dpi})")
            return str(filepath)
        except Exception as e:
            print(f"\033[31mVisual report generation failed: {str(e)}\033[0m")
            raise
        finally:
            plt.close('all')  # Close all figures to free memory

    def _plot_rhythm_alignment(self):
        """Plot rhythm alignment path"""
        rhythm_data = self.rhythm_analysis()
        plt.plot(rhythm_data['alignment_path'][:, 0],
                 rhythm_data['alignment_path'][:, 1],
                 'b-', alpha=0.5)
        plt.title("Rhythm Alignment (DTW)")
        plt.xlabel("Reference Note Index")
        plt.ylabel("Test Note Index")
        plt.grid(True)

    def _plot_pitch_time_comparison(self):
        """Plot pitch vs time comparison"""
        # Use cached non-rest notes
        if self._ref_notes_without_rest is None:
            self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
        if self._test_notes_without_rest is None:
            self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]

        ref_times = [n['start'] for n in self._ref_notes_without_rest]
        ref_pitches = [n['pitch'] for n in self._ref_notes_without_rest]
        test_times = [n['start'] for n in self._test_notes_without_rest]
        test_pitches = [n['pitch'] for n in self._test_notes_without_rest]

        plt.scatter(ref_times, ref_pitches,
                    c='#1f77b4', alpha=0.6,
                    s=15, label='Reference',
                    edgecolors='w', linewidth=0.3)
        plt.scatter(test_times, test_pitches,
                    c='#ff7f0e', alpha=0.6,
                    s=15, label='Test',
                    marker='s', edgecolors='w',
                    linewidth=0.3)

        plt.xlabel("Time (beats)")
        plt.ylabel("MIDI Pitch Value")
        plt.title("Pitch vs Time Comparison")

        # Set axis range
        min_time = min(min(ref_times), min(test_times)) - 1
        max_time = max(max(ref_times), max(test_times)) + 1
        plt.xlim(min_time, max_time)

        # Add piano range annotations
        ax2 = plt.gca().twinx()
        ax2.set_ylim(plt.gca().get_ylim())
        ax2.set_yticks([21, 36, 48, 60, 72, 84, 108])
        ax2.set_yticklabels(['A0 (Low)', 'C2', 'C3', 'C4 (Middle)',
                             'C5', 'C6', 'C8 (High)'])

        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(title='Version:')

    def _plot_velocity_analysis(self):
        """Plot velocity analysis"""
        # Use cached non-rest notes
        if self._ref_notes_without_rest is None:
            self._ref_notes_without_rest = [n for n in self.ref_notes if not n['is_rest']]
        if self._test_notes_without_rest is None:
            self._test_notes_without_rest = [n for n in self.test_notes if not n['is_rest']]

        ref_vel = [n['velocity'] for n in self._ref_notes_without_rest]
        test_vel = [n['velocity'] for n in self._test_notes_without_rest]

        plt.plot(ref_vel, 'g-', label='Reference Velocity')
        plt.plot(test_vel, 'r--', label='Test Velocity')
        plt.title("Velocity Comparison")
        plt.xlabel("Note Index")
        plt.ylabel("Velocity Value")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def _get_valid_onsets(notes: List[NoteDict]) -> List[float]:
        return [n['start'] for n in notes if not n['is_rest']]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MIDI Score Evaluation System - Compare reference and test MIDI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py reference.mid test.mid
  python test.py reference.mid test.mid --onset-tol 0.5 --pitch-tol 2 --no-visual
  python test.py reference.mid test.mid --use-gpu
        """
    )
    parser.add_argument("reference", help="Path to reference MIDI file")
    parser.add_argument("test", help="Path to test MIDI file")
    parser.add_argument("--onset-tol", type=float, default=0.0,
                        help="Time tolerance for note matching (in beats, default: 0.0)")
    parser.add_argument("--pitch-tol", type=int, default=3,
                        help="Pitch tolerance for note matching (in semitones, default: 3)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Enable GPU acceleration for DTW (if available)")
    parser.add_argument("--no-visual", action="store_true",
                        help="Disable visual report generation")

    args = parser.parse_args()

    try:
        ref_path = Path(args.reference)
        test_path = Path(args.test)

        print("\n" + "=" * 50)
        print(" MIDI Score Evaluation System v4.5 ".center(50, '*'))
        print("=" * 50)
        print(f"Reference: {ref_path.name}")
        print(f"Test: {test_path.name}")
        print(f"Onset tolerance: ±{args.onset_tol} beats")
        print(f"Pitch tolerance: ±{args.pitch_tol} semitones")
        if args.use_gpu:
            print("GPU acceleration: Enabled")
        else:
            print("GPU acceleration: Disabled")

        evaluator = ScoreEvaluator(ref_path, test_path)

        print("\n" + " Analysis Results ".center(50, '-'))
        accuracy = evaluator.note_accuracy(onset_tol=args.onset_tol, pitch_tol=args.pitch_tol)
        print(f"Note Accuracy:")
        print(f"  Precision: {accuracy['precision']}")
        print(f"  Recall:    {accuracy['recall']}")
        print(f"  F1-score:  {accuracy['f1']}")

        rhythm = evaluator.rhythm_analysis()
        print(f"\nRhythm Analysis:")
        print(f"  DTW distance:      {rhythm['dtw_distance']:.1f}")
        print(f"  Avg time deviation: {rhythm['avg_time_deviation']} beats")
        print(f"  Max time deviation: {rhythm['max_deviation']} beats")

        if not args.no_visual:
            print("\nGenerating visual report...")
            evaluator.generate_visual_report()
        else:
            print("\nVisual report generation skipped (--no-visual)")

    except Exception as e:
        print(f"\n\033[31m[ERROR] {str(e)}\033[0m")
        import sys
        sys.exit(1)
    finally:
        print("\n" + " Analysis Complete ".center(50, '=') + "\n")
        