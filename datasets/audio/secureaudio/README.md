---
license: apache-2.0
task_categories:
- audio-classification
tags:
- audio-forensics
- deepfake-detection
- speech
- raw-audio
size_categories:
- 1K<n<10K
---

# SecureAudio Standardized Forensic Audio Corpus

Standardized, unpadded holdout split derived from `garystafford/deepfake-audio-detection` for raw 1D acoustic deepfake research.

## Dataset Structure

- **Sample Rate**: 16,000 Hz (PCM 16-bit)
- **Real Audio**: Genuine human vocal recordings preserving natural microphone room noise and breath dynamics.
- **Fake Audio**: Synthetic generative voice tracks generated via neural text-to-speech / voice conversion vocoders.

### Split Distribution (80 / 20)

| Partition | Real Audio (Label 0) | Fake Audio (Label 1) | Total Audio Files |
| :--- | :---: | :---: | :---: |
| **Train** | 746 | 746 | 1492 |
| **Test** | 187 | 187 | 374 |
| **Total** | 933 | 933 | 1866 |