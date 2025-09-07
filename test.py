"""
MIDI Score Evaluation System v4.5
English Visualization Edition
Time-Pitch Analysis Enhanced
"""

import numpy as np
import bisect
from pathlib import Path
from typing import Dict, List, Tuple
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tabulate import tabulate
import matplotlib.pyplot as plt
from music21 import converter, note, chord, key, stream, voiceLeading

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

    def _parse_midi(self, midi_path: Path) -> List[Dict]:
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

    def _process_chord(self, element) -> List[Dict]:
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

    def _create_note_dict(self, element) -> Dict:
        return {
            'pitch': element.pitch.midi,
            'start': float(element.offset),
            'end': float(element.offset + element.duration.quarterLength),
            'velocity': element.volume.velocity if hasattr(element.volume, 'velocity') else 64,
            'is_rest': False,
            'is_chord': False,
            'track': element.activeSite.id if hasattr(element.activeSite, 'id') else 0
        }

    def _create_rest_dict(self, element) -> Dict:
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

    def note_accuracy(self, onset_tol=0., pitch_tol=3) -> Dict:
        """Note matching algorithm"""
        ref_notes = [n for n in self.ref_notes if not n['is_rest']]
        test_notes = [n for n in self.test_notes if not n['is_rest']]

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

        return {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'onset_tol': onset_tol,
            'pitch_tol': pitch_tol
        }

    def rhythm_analysis(self) -> Dict:
        """Rhythm analysis using DTW"""
        ref = [[n['start'], n['end'] - n['start']] for n in self.ref_notes if not n['is_rest']]
        test = [[n['start'], n['end'] - n['start']] for n in self.test_notes if not n['is_rest']]

        distance, path = fastdtw(ref, test, dist=euclidean)
        alignment = np.array(path)

        # Calculate timing deviations
        time_diffs = []
        for i, j in path:
            time_diffs.append(abs(ref[i][0] - test[j][0]))

        return {
            'dtw_distance': distance,
            'avg_time_deviation': round(np.mean(time_diffs), 3),
            'max_deviation': round(np.max(time_diffs), 3),
            'alignment_path': alignment
        }

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

    def generate_visual_report(self):
        """Generate visual analysis report"""
        plt.figure(figsize=(15, 10))

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

            # Save figure
            plt.tight_layout(pad=3.0)
            filename = f"{self.test_midi.stem}_analysis_en.png"
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"\nVisual report saved: {filename}")
        except Exception as e:
            print(f"\033[31mVisual report generation failed: {str(e)}\033[0m")
        finally:
            plt.close()

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
        ref_times = [n['start'] for n in self.ref_notes if not n['is_rest']]
        ref_pitches = [n['pitch'] for n in self.ref_notes if not n['is_rest']]
        test_times = [n['start'] for n in self.test_notes if not n['is_rest']]
        test_pitches = [n['pitch'] for n in self.test_notes if not n['is_rest']]

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
        ref_vel = [n['velocity'] for n in self.ref_notes if not n['is_rest']]
        test_vel = [n['velocity'] for n in self.test_notes if not n['is_rest']]

        plt.plot(ref_vel, 'g-', label='Reference Velocity')
        plt.plot(test_vel, 'r--', label='Test Velocity')
        plt.title("Velocity Comparison")
        plt.xlabel("Note Index")
        plt.ylabel("Velocity Value")
        plt.legend()
        plt.grid(True)

    @staticmethod
    def _get_valid_onsets(notes: List[Dict]) -> List[float]:
        return [n['start'] for n in notes if not n['is_rest']]


if __name__ == "__main__":
    try:

        ref_path = Path(r"C:\Users\LENOVO\Downloads\test2.mid")
        test_path = Path(r"C:\Users\LENOVO\Downloads\converted (17).mid")
        print("\n" + "=" * 40)
        print(" MIDI Score Evaluation System ".center(40, '*'))
        print("=" * 40)

        evaluator = ScoreEvaluator(ref_path, test_path)

        print("\n" + " Analysis Results ".center(40, '-'))
        print(evaluator.generate_text_report())

        evaluator.generate_visual_report()

    except Exception as e:
        print(f"\n\033[31m[ERROR] {str(e)}\033[0m")
    finally:
        print("\n" + " Analysis Complete ".center(40, '=') + "\n")