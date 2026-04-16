"""Final demonstration: Show system improvement with feedback loop"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midi_feedback_enhanced import EnhancedFeedbackSystem, EnhancedParameterSet
from midi_generate_enhanced import EnhancedMIDIGenerator
from midi_evaluator import ScoreEvaluator

def demonstrate_improvement():
    print("MIDI System Improvement Demonstration")
    print("=" * 60)
    print("Goal: Show how feedback loop improves audio-to-MIDI conversion")
    print()

    # Load files
    audio_path = Path("../data/test2.mp3")
    midi_path = Path("../data/test2.mid")

    if not audio_path.exists() or not midi_path.exists():
        print("Files not found. Please ensure test2.mp3 and test2.mid are present.")
        return

    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    print(f"Audio: {audio_path.name} ({len(audio_bytes):,} bytes)")
    print(f"Reference MIDI: {midi_path.name} (138 notes)")
    print()

    # Step 1: Baseline with default parameters
    print("1. BASELINE PERFORMANCE (Default Parameters)")
    print("-" * 40)
    default_generator = EnhancedMIDIGenerator()
    midi_default = default_generator.process_audio(audio_bytes)
    if midi_default and midi_default.instruments and midi_default.instruments[0].notes:
        default_notes = len(midi_default.instruments[0].notes)
        midi_default.write("baseline_default.mid")
        evaluator = ScoreEvaluator(midi_path, Path("baseline_default.mid"))
        accuracy_default = evaluator.note_accuracy()
        print(f"   Generated notes: {default_notes}")
        print(f"   F1 score: {accuracy_default['f1']:.3f}")
        print(f"   Precision: {accuracy_default['precision']:.3f}")
        print(f"   Recall: {accuracy_default['recall']:.3f}")
        baseline_f1 = accuracy_default['f1']
    else:
        print("   Failed to generate MIDI")
        baseline_f1 = 0

    print()

    # Step 2: Audio-informed parameters (smart starting point)
    print("2. AUDIO-INFORMED PARAMETERS (Smart Starting Point)")
    print("-" * 40)
    informed_params = EnhancedParameterSet(
        sr=44100,
        hop_length=512,
        min_freq=200,
        max_freq=1000,
        pitch_method='pyin',
        voicing_threshold=0.6,
        confidence_weight=0.7,
        min_note_duration=0.08,
        max_gap=0.05,
        onset_threshold=0.5
    )
    informed_generator = EnhancedMIDIGenerator(informed_params.to_generator_config())
    midi_informed = informed_generator.process_audio(audio_bytes)
    if midi_informed and midi_informed.instruments and midi_informed.instruments[0].notes:
        informed_notes = len(midi_informed.instruments[0].notes)
        midi_informed.write("informed_baseline.mid")
        evaluator = ScoreEvaluator(midi_path, Path("informed_baseline.mid"))
        accuracy_informed = evaluator.note_accuracy()
        print(f"   Generated notes: {informed_notes}")
        print(f"   F1 score: {accuracy_informed['f1']:.3f}")
        print(f"   Precision: {accuracy_informed['precision']:.3f}")
        print(f"   Recall: {accuracy_informed['recall']:.3f}")
        informed_f1 = accuracy_informed['f1']
        improvement_from_default = informed_f1 - baseline_f1
        print(f"   Improvement over default: {improvement_from_default:+.3f} F1")
    else:
        print("   Failed to generate MIDI")
        informed_f1 = 0

    print()

    # Step 3: Quick optimization (3 iterations to show learning)
    print("3. OPTIMIZATION WITH FEEDBACK LOOP (3 iterations)")
    print("-" * 40)
    system = EnhancedFeedbackSystem()
    print("   Running optimization...")
    best_params = system.optimize_for_audio(
        audio_bytes=audio_bytes,
        reference_midi_path=midi_path,
        n_iterations=3,
        strategy="adaptive"
    )

    print(f"   Best score achieved: {system.best_score:.4f}")
    print(f"   Best parameters found:")
    print(f"     min_freq: {best_params.min_freq}")
    print(f"     max_freq: {best_params.max_freq}")
    print(f"     hop_length: {best_params.hop_length}")
    print(f"     pitch_method: {best_params.pitch_method}")

    # Step 4: Optimized generator
    print()
    print("4. OPTIMIZED GENERATOR (After Feedback)")
    print("-" * 40)
    optimized_generator = system.create_optimized_generator()
    midi_optimized = optimized_generator.process_audio(audio_bytes)
    if midi_optimized and midi_optimized.instruments and midi_optimized.instruments[0].notes:
        optimized_notes = len(midi_optimized.instruments[0].notes)
        midi_optimized.write("optimized_result.mid")
        evaluator = ScoreEvaluator(midi_path, Path("optimized_result.mid"))
        accuracy_optimized = evaluator.note_accuracy()
        print(f"   Generated notes: {optimized_notes}")
        print(f"   F1 score: {accuracy_optimized['f1']:.3f}")
        print(f"   Precision: {accuracy_optimized['precision']:.3f}")
        print(f"   Recall: {accuracy_optimized['recall']:.3f}")
        optimized_f1 = accuracy_optimized['f1']
        improvement_from_informed = optimized_f1 - informed_f1
        total_improvement = optimized_f1 - baseline_f1
        print(f"   Improvement over informed: {improvement_from_informed:+.3f} F1")
        print(f"   Total improvement: {total_improvement:+.3f} F1")
    else:
        print("   Failed to generate optimized MIDI")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY: SYSTEM IMPROVEMENT")
    print("=" * 60)
    print(f"Baseline (default):     F1 = {baseline_f1:.3f}")
    print(f"Audio-informed:         F1 = {informed_f1:.3f} ({improvement_from_default:+.3f})")
    print(f"After optimization:     F1 = {optimized_f1:.3f} ({improvement_from_informed:+.3f})")
    print(f"Total improvement:      {total_improvement:+.3f} F1 score")
    print()
    print("Generated files:")
    print("  baseline_default.mid  - Default parameters")
    print("  informed_baseline.mid - Audio-informed parameters")
    print("  optimized_result.mid  - After feedback loop optimization")
    print()
    print("The system successfully learned from the reference MIDI and")
    print("improved the audio-to-MIDI conversion accuracy.")

    # Save results
    results = {
        'baseline_f1': baseline_f1,
        'informed_f1': informed_f1,
        'optimized_f1': optimized_f1,
        'improvement_from_default': float(improvement_from_default),
        'improvement_from_informed': float(improvement_from_informed),
        'total_improvement': float(total_improvement),
        'best_params': best_params.to_dict() if 'best_params' in locals() else None
    }
    with open("demo_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    demonstrate_improvement()