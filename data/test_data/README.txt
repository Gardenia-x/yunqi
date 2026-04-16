Test Data Directory Structure

Place your test files here:
- audio/: Audio files (WAV, MP3) for conversion
- midi/: Reference MIDI files (ground truth)
- output/: Generated MIDI files and results

For the feedback system to work, you need:
1. Audio file (e.g., test.wav)
2. Corresponding reference MIDI file (e.g., test.mid)

The system will:
1. Convert audio to MIDI using current parameters
2. Compare generated MIDI with reference MIDI
3. Calculate accuracy metrics
4. Use metrics to improve parameters
