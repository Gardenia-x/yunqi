"""
对比测试：SwiftF0 vs 当前 PYIN 方法
"""
import time
import numpy as np
import librosa
import pretty_midi
from pathlib import Path
from swift_f0 import SwiftF0, segment_notes, export_to_midi

# 测试文件
TEST_FILES = [
    Path("D:/python/Yunqi/data/test2.mp3"),
    Path("D:/python/Yunqi/data/test3.mp3"),
    Path("D:/python/Yunqi/data/test5.mp3"),
]

OUT_DIR = Path("D:/python/Yunqi/test_output")
OUT_DIR.mkdir(exist_ok=True)


def test_swiftf0(audio_path: Path):
    """使用 SwiftF0 提取旋律 → MIDI"""
    detector = SwiftF0(
        fmin=65,        # C2
        fmax=2000,      # ~B6
        confidence_threshold=0.3,
    )

    start = time.time()
    result = detector.detect_from_file(str(audio_path))
    detect_time = time.time() - start

    # 音符分割
    notes = segment_notes(
        result,
        split_semitone_threshold=0.8,
        min_note_duration=0.05,
    )

    total_time = time.time() - start
    note_count = len(notes)

    # 导出 MIDI
    out_path = OUT_DIR / f"{audio_path.stem}_swiftf0.mid"
    export_to_midi(notes, str(out_path))

    print(f"  SwiftF0: {note_count} 音符, "
          f"检测 {detect_time:.1f}s, 总计 {total_time:.1f}s")

    # 统计音高跳跃
    if len(notes) >= 2:
        pitches = [n.pitch_midi for n in notes]
        jumps = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
        avg_jump = np.mean(jumps)
        large_jumps = sum(1 for j in jumps if j > 12)  # 超过一个八度的跳变
        print(f"  平均音高跳变: {avg_jump:.1f} 半音, 大跳变(>8度): {large_jumps}")

    return notes, total_time, note_count


def test_existing_pyin(audio_path: Path):
    """使用当前 PYIN 方法提取旋律 → MIDI（仅统计对比）"""
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
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    midi = generator.process_audio(audio_bytes)
    elapsed = time.time() - start

    notes = midi.instruments[0].notes if midi.instruments else []
    note_count = len(notes)

    out_path = OUT_DIR / f"{audio_path.stem}_pyin.mid"
    midi.write(str(out_path))

    print(f"  PYIN:   {note_count} 音符, 耗时 {elapsed:.1f}s")

    if len(notes) >= 2:
        pitches = [n.pitch for n in notes]
        jumps = [abs(pitches[i] - pitches[i-1]) for i in range(1, len(pitches))]
        avg_jump = np.mean(jumps)
        large_jumps = sum(1 for j in jumps if j > 12)
        print(f"  平均音高跳变: {avg_jump:.1f} 半音, 大跳变(>8度): {large_jumps}")

    return notes, elapsed, note_count


def main():
    print("=" * 60)
    print("SwiftF0 vs PYIN 对比测试")
    print("=" * 60)

    for fpath in TEST_FILES:
        if not fpath.exists():
            print(f"\n[跳过] {fpath.name} (文件不存在)")
            continue

        print(f"\n--- {fpath.name} ---")
        duration = librosa.get_duration(path=str(fpath))
        print(f"  时长: {duration:.1f}s")

        test_swiftf0(fpath)
        test_existing_pyin(fpath)

    print(f"\nMIDI 文件已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
