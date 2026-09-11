# LipSynthesis microphone-replay failure

- Reported: 2026-09-10
- Source: https://www.youtube.com/watch?v=OiPvhd9ltQI
- Public title: `4k DeepFake Lip Syncing Example of Boris Johnson by LipSynthesis`
- Source description: identifies the example as fully AI-generated, with cloned lip sync and British-accent speech from two minutes of source video.
- User procedure: played the synthetic part and captured it through GovVoice Shield's microphone mode.
- Observed outcome: the v2 feature model presented a human assessment.
- Audio artifact retained: no. The exact microphone capture is therefore unavailable for reproducible scoring or training.

## Interpretation

This is a credible field failure report with public generator provenance, but not yet a reproducible labelled test item because the analyzed bytes were not retained. YouTube encoding followed by loudspeaker, room, microphone, and browser encoding creates a replay domain absent from the v2 primary training protocol. The report is consistent with known replay vulnerability in audio-deepfake detectors.

## Product action

Microphone input is now treated as replay-risk. If the feature model leans human, the interface returns `uncertain` while preserving the raw synthetic estimate and records `baseLabel: real` plus `policyReason: microphone-replay-risk` in the downloaded report. Synthetic-leaning results remain synthetic. This is an uncertainty safeguard, not evidence that the classifier learned to detect this generator.

## Required regression artifact

Retain the next recording as a WAV or download the assessment together with the source clip. Record the playback device, room, microphone, browser, recording date, exact video URL, and whether the entire video is synthetic. Keep this case outside the frozen v2 metrics and use it only in development until a new evaluation protocol is locked.
