# GovVoice Shield

**Team FalconByte · School of Cyber Defense Competition**

Live demonstration: https://hadiarahmani.github.io/GovVoice-Shield/

Source repository: https://github.com/HadiaRahmani/GovVoice-Shield

## Publish with GitHub Pages

Upload the contents of this project to the repository root. In **Settings → Pages**, choose **Deploy from a branch**, select **main** and **/(root)**, then save. The root `index.html` opens the complete application in `dist`, so no build workflow is required.

Audio screening for the competition: choose a reel sample or WAV/MP3, listen, guess Human / Synthetic / Not sure, then reveal the assessment and demo source label. Uploaded recordings remain on the device. A verdict does not authenticate an official, prove an edit, or verify a statement.

The motivating case is a voice message circulating through social media, messaging applications or broadcast channels that claims to come from a government official. A forged emergency instruction, payment request or public announcement can spread fraud, panic or disinformation before the source is checked. The intended user is a government communications officer or security reviewer deciding whether to pause redistribution and escalate verification through a known official channel.

## Run

From this folder with Python 3.12+, run:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory dist
```

Open http://127.0.0.1:8765 in a modern desktop browser. Keep the terminal open. All assets, trained weights and ten WAV files are included. The local copy needs no internet for inference. Opening the HTML as a file will not support workers/modules.

Input: PCM WAV or MP3, 2–60 seconds, maximum 20 MB. Output: Likely human, Likely synthetic, or Uncertain; a study-calibrated estimate; exploratory review timestamps; JSON report. Video requires extracting audio first. Mono conversion and resampling happen locally; neither proves authenticity.

## Model and evidence

Deployed v2 uses 90 MFCC/delta/spectral statistics and an RBF SVM trained here. Nine configurations are selected using tuning data; probability calibration uses a separate partition. User uploads also receive an AASIST neural safety check. AASIST can challenge a human-leaning primary result, but it cannot clear an RBF-SVM synthetic warning. The result view exposes both detector estimates, input and analysis rates, duration, RMS, clipping, and a 26-band log-mel frequency map. The frequency map shares the feature model's bands but is supporting visualization, not an authenticity proof or model attribution. The frozen AASIST weights were not trained here, and this post-test safeguard does not alter the locked v2 headline metrics. Hugging Face hosts public training data; there is no detector API at runtime.

## Operational concept

GovVoice Shield is a triage aid for communications officers and security review teams. Its four-step response protocol is **Screen → Contain → Verify → Record**: inspect the evidence, pause redistribution when a result is synthetic/conflicting/uncertain, confirm the message through a known official channel, and retain the original file with the downloadable assessment. This connects model output to a practical government workflow without presenting audio classification as identity authentication.

The RBF-SVM is the primary analysis path for every source mode. AASIST is an independent one-way safety challenge for original-file uploads only. A human result is released only when AASIST also leans human; an RBF-SVM synthetic warning remains synthetic even when AASIST misses it. Microphone recordings are treated as replay-risk inputs because the loudspeaker, room and microphone can conceal synthetic artifacts; a human-leaning microphone result therefore abstains.

## Why this architecture now

Government reviewers may need to screen sensitive recordings quickly without sending them to a third-party detector API. The compact RBF-SVM gives every supported input a deterministic, inspectable and fully local analysis path. Original digital-file uploads also receive an AASIST neural cross-check, where preserved waveform detail can provide complementary evidence. The asymmetric rule is deliberately safety-oriented: neural evidence may stop a human clearance, but cannot erase a synthetic alert. Microphone captures follow a separate replay-risk policy because loudspeaker, room and encoding effects can erase useful synthesis cues.

## Innovation boundary

The contribution is an **input-aware selective screening workflow**, not a new classifier architecture. It combines local privacy, calibrated abstention, an upload-only neural safety veto, replay-aware microphone decisions, interpretable signal views and the operational response **Screen → Contain → Verify → Record**. This design turns imperfect detectors into a safer decision aid for fast-moving government impersonation incidents while keeping the known limits visible.

An exploratory post-test check applied the exact upload policy to the preserved 80-clip paired primary set. It accepted 75 clips (93.75% coverage) and was correct on 74/75 accepted clips (98.67%); one human-leaning clearance was challenged, while 33 primary synthetic alerts that AASIST missed were preserved. AASIST alone achieved only 56.25% accuracy and 0.506 ROC-AUC on this shifted set. Because this test was reused after failures had been inspected, these figures do not replace the locked v2 headline results.

See dist/evaluation.json for measured results, research/v2/PROTOCOL.md and hash locks for the experiment, and BUILD_RECORD.md for findings and source links. Earlier results remain under research/archive/v1 and must not be presented as v2 results.

## Verify

With Node.js 22+:

```powershell
node tests/test-core.mjs
node tests/test-v2.mjs
node tests/test-record.mjs
node tests/test-policy.mjs
node tests/test-neural.mjs
```

Tests cover feature/probability parity, bounded log-mel visualization output, WAV/MP3 metadata inspection, actual SVM and AASIST inference, ten demo files, invalid input, repeat runs and 60-second runtime bounds. Node tests do not establish browser playback, accessibility or device compatibility. Historical AASIST accuracy remains v1 evidence; its current upload use is a post-test one-way safety safeguard.

## Reproduce

Use a copy, preserving delivered locks and results. Scripts refuse to overwrite frozen experiments. Give a new experiment a new output directory/version; do not delete delivered locks to tune the final test. Manifests record original files and assignments. Speech-service output can change.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r research/requirements.txt
# Original sequence, already executed in this delivery:
.\.venv\Scripts\python research/prepare_v2.py
.\.venv\Scripts\python research/generate_v2.py
.\.venv\Scripts\python research/train_v2.py
.\.venv\Scripts\python research/evaluate_v2.py
```

Set GOVVOICE_WORK_DIR to an absolute writable cache directory when running elsewhere. Downloads/generation need network access and several GB of disk. Training uses CPU, no paid GPU or detector API key. Manifests pin actual source revisions, but a new acquisition queries the then-current revision; exact reproduction needs recorded revisions and files. Generation sends corpus book excerpts to stock Microsoft speech synthesis through edge-tts. No voice cloning. Two announcement scripts are fictional. This service is not used by the running detector.

## Competition handoff

Use the five-page PDF with the demo. Let the audience listen and guess before reveal. Show source labels even when the model fails or abstains. Do not claim production readiness, forensic proof or a guaranteed competition score. Only report measurements preserved in this repository.

Preserve attribution and licences. Dataset card declarations are not independent audits of underlying recording rights. This demo is for private research review; assess distribution permissions before a public release.

## Record voice
Choose Record voice, click Start recording and allow microphone access. Speak for 3–15 seconds, then Stop (automatic at 15). Listen, make a guess and reveal. HTTPS or localhost is required. Open the site directly in Chrome if an embedded browser blocks permission. Audio stays local; switching mode or leaving the tab cancels capture.

Microphone capture is a replay-risk input: loudspeaker, room, microphone and browser encoding can conceal synthesis artifacts. A human-leaning model score therefore returns Uncertain in this mode, while the raw probability and `baseLabel` remain in the downloadable report. Synthetic-leaning scores remain Likely synthetic. For uploads, the RBF-SVM and AASIST both run locally. AASIST may challenge an RBF-SVM human result, while an RBF-SVM synthetic warning remains synthetic. The supplied `sample 1.mp3` reproduced the original SVM failure, while AASIST produced a 99.78% synthetic estimate, so the combined interface now abstains instead of saying human. This is post-test evidence and is excluded from the frozen v2 headline metrics. Actual hardware permissions and codec support still require browser rehearsal.
