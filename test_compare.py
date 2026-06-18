"""
三方法对比：Basic Pitch vs SwiftF0 vs PYIN
"""
import time, numpy as np, librosa
from pathlib import Path
from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from swift_f0 import SwiftF0, segment_notes, export_to_midi as sf_export

TEST_FILES = [
    Path("D:/python/Yunqi/data/test2.mp3"),
    Path("D:/python/Yunqi/data/test3.mp3"),
    Path("D:/python/Yunqi/data/test5.mp3"),
]
OUT_DIR = Path("D:/python/Yunqi/test_output")
OUT_DIR.mkdir(exist_ok=True)

BP_MODEL = Path(ICASSP_2022_MODEL_PATH).parent / "nmp.onnx"


def test_basic_pitch(fpath):
    start = time.time()
    _, midi_data, note_events = bp_predict(str(fpath), BP_MODEL)
    elapsed = time.time() - start
    notes = midi_data.instruments[0].notes if midi_data.instruments else []
    midi_data.write(str(OUT_DIR / f"{fpath.stem}_basicpitch.mid"))
    return notes, elapsed, f"Basic Pitch"


def test_swiftf0(fpath):
    detector = SwiftF0(fmin=65, fmax=2000, confidence_threshold=0.4)
    start = time.time()
    result = detector.detect_from_file(str(fpath))
    notes = segment_notes(result, split_semitone_threshold=0.8, min_note_duration=0.06)
    elapsed = time.time() - start
    sf_export(notes, str(OUT_DIR / f"{fpath.stem}_swiftf0.mid"))
    # Convert to pretty_midi notes for analysis
    import pretty_midi
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        inst.notes.append(pretty_midi.Note(
            velocity=80, pitch=n.pitch_midi,
            start=n.start, end=n.end,
        ))
    midi.instruments.append(inst)
    return inst.notes, elapsed, f"SwiftF0"


def test_pyin(fpath):
    import sys
    sys.path.insert(0, str(Path("D:/python/Yunqi/src")))
    from simple_midi_generator import SimpleMIDIGenerator

    config = {
        "sr": 44100, "hop_length": 512,
        "min_freq": 65, "max_freq": 2000,
        "min_duration": 0.06, "max_gap": 0.06,
        "semitone_tolerance": 1, "pitch_smooth_kernel": 7,
        "confidence_threshold": 0.3, "voicing_threshold": 0.5,
        "detection_method": "pyin",
    }
    generator = SimpleMIDIGenerator(config)
    start = time.time()
    with open(fpath, "rb") as f:
        midi = generator.process_audio(f.read())
    elapsed = time.time() - start
    notes = midi.instruments[0].notes if midi.instruments else []
    midi.write(str(OUT_DIR / f"{fpath.stem}_pyin.mid"))
    return notes, elapsed, f"PYIN"


def analyze(name, notes, elapsed):
    if not notes:
        return f"  {name}: 0 notes, {elapsed:.1f}s"

    pitches = [n.pitch for n in notes]
    durations = [n.end - n.start for n in notes]
    jumps = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]

    lines = [
        f"  {name}: {len(notes)} notes, {elapsed:.1f}s",
        f"    音高: MIDI {min(pitches)}-{max(pitches)}",
        f"    跳变: avg={sum(jumps)/len(jumps):.1f} st, max={max(jumps) if jumps else 0}, >8度={sum(1 for j in jumps if j>12)}",
        f"    时值: avg={np.mean(durations):.2f}s, median={np.median(durations):.2f}s",
    ]
    return "\n".join(lines)


def main():
    print("=" * 65)
    print("Basic Pitch (ONNX) vs SwiftF0 vs PYIN — 对比测试")
    print("=" * 65)

    for fpath in TEST_FILES:
        if not fpath.exists():
            continue

        dur = librosa.get_duration(path=str(fpath))
        print(f"\n{'─'*50}")
        print(f"  {fpath.name}  (时长 {dur:.1f}s)")
        print(f"{'─'*50}")

        for test_fn in [test_basic_pitch, test_swiftf0, test_pyin]:
            try:
                notes, elapsed, name = test_fn(fpath)
                print(analyze(name, notes, elapsed))
            except Exception as e:
                print(f"  {test_fn.__name__}: FAILED — {e}")

    print(f"\nMIDI 文件保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
