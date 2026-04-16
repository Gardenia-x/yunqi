"""Diagnose audio and MIDI to understand optimization issues"""
import librosa
import numpy as np
import pretty_midi
from pathlib import Path

def diagnose():
    audio_path = Path("test2.mp3")
    midi_path = Path("test2.mid")

    print("Audio and MIDI Diagnosis")
    print("=" * 60)

    # Audio analysis
    print("\n1. AUDIO ANALYSIS")
    y, sr = librosa.load(audio_path, sr=None)
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(y)/sr:.2f} seconds")
    print(f"  Samples: {len(y):,}")
    print(f"  Max amplitude: {np.max(np.abs(y)):.4f}")
    print(f"  Mean amplitude: {np.mean(np.abs(y)):.4f}")

    # Spectral analysis
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    print(f"  Spectral centroid: {np.mean(spectral_centroid):.1f} Hz (avg)")
    print(f"  Spectral bandwidth: {np.mean(spectral_bandwidth):.1f} Hz (avg)")

    # Pitch range estimation
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        print(f"  Pitch range: {np.min(voiced_f0):.1f} - {np.max(voiced_f0):.1f} Hz")
        print(f"  Voiced frames: {len(voiced_f0)} / {len(f0)} ({len(voiced_f0)/len(f0)*100:.1f}%)")
    else:
        print(f"  No voiced pitch detected with PYIN")

    # MIDI analysis
    print("\n2. REFERENCE MIDI ANALYSIS")
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    print(f"  Instruments: {len(midi.instruments)}")
    for i, inst in enumerate(midi.instruments):
        name_safe = inst.name.encode('ascii', 'ignore').decode('ascii')
        print(f"    Instrument {i}: {name_safe}, {len(inst.notes)} notes")
        if inst.notes:
            pitches = [note.pitch for note in inst.notes]
            velocities = [note.velocity for note in inst.notes]
            durations = [note.end - note.start for note in inst.notes]
            print(f"      Pitch range: {min(pitches)}-{max(pitches)} "
                  f"(MIDI notes: {librosa.midi_to_note(min(pitches))}-{librosa.midi_to_note(max(pitches))})")
            print(f"      Pitch mean: {np.mean(pitches):.1f}")
            print(f"      Velocity: {np.mean(velocities):.1f} avg, {np.std(velocities):.1f} std")
            print(f"      Duration: {np.mean(durations):.3f}s avg, {np.std(durations):.3f}s std")

    # Compare with generated MIDI from debug run
    print("\n3. GENERATED MIDI (from debug run)")
    debug_midi_path = Path("debug_generated.mid")
    if debug_midi_path.exists():
        gen_midi = pretty_midi.PrettyMIDI(str(debug_midi_path))
        print(f"  Instruments: {len(gen_midi.instruments)}")
        for i, inst in enumerate(gen_midi.instruments):
            print(f"    Instrument {i}: {inst.name}, {len(inst.notes)} notes")
            if inst.notes:
                pitches = [note.pitch for note in inst.notes]
                print(f"      Pitch range: {min(pitches)}-{max(pitches)}")
                print(f"      Pitch mean: {np.mean(pitches):.1f}")

    # Parameter suggestions
    print("\n4. PARAMETER SUGGESTIONS")
    if len(voiced_f0) > 0:
        min_hz = np.min(voiced_f0)
        max_hz = np.max(voiced_f0)
        print(f"  Suggested min_freq: {max(20, min_hz * 0.8):.0f} Hz")
        print(f"  Suggested max_freq: {min(4000, max_hz * 1.2):.0f} Hz")
    else:
        print(f"  Default ranges: min_freq=80, max_freq=2000")

    print(f"  Suggested hop_length: 512 (balance of time/freq resolution)")
    print(f"  For {sr} Hz audio, frame duration: {512/sr*1000:.1f} ms")

if __name__ == "__main__":
    diagnose()