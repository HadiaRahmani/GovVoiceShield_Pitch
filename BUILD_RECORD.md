# GovVoice Shield — build record

## Post-test replay safeguard

A user-reported LipSynthesis example was presented as human after speaker-to-microphone capture. The public source identifies the example as fully AI-generated, but the exact captured audio was not retained. Because replayed audio is outside the frozen v2 primary protocol, microphone inputs now use a source-aware safeguard: a human-leaning model output is presented as uncertain with a replay warning, while the raw estimate and base model label remain in the downloadable report. This change prevents false reassurance; it does not improve or revalidate the underlying classifier. The case record is stored at `research/failures/lipsynthesis-replay-2026-09-10.md`.

A second post-test case, the user-labelled synthetic `sample 1.mp3`, reproduced the RBF-SVM human result at 0.027% synthetic. The archived AASIST neural detector independently produced a 99.78% synthetic estimate on the same decoded audio. AASIST now acts as a one-way safety challenge: it can stop a human clearance, but it cannot erase an RBF-SVM synthetic warning. Both estimates remain in the report. This is a post-test safeguard, not a retrained detector or a change to frozen v2 headline results. The case record and file hash are stored at `research/failures/sample-1-upload-2026-09-11.md`.

## Current stage
Working research prototype with a trained browser classifier, locked-test measurements and a ten-sample listening demo. It is not a production government authentication system. Browser visual/playback interaction and a rehearsal on the competition device remain unverified. No competition score is promised.

## Research scope and engineering decisions
The system uses acoustic classification and cautious supporting evidence. It does not treat electrical-network frequency, pitch, loudness, mono conversion or metadata alone as a fake/real rule. ENF analysis requires an appropriate captured signal and a trusted reference, while compressed web speech may not preserve it. Only supplied audio with independently established provenance can be evaluated as labelled evidence.

## Product and business fit
An analyst handling a purported government voice message listens, screens it locally, reviews uncertain or suspicious intervals and verifies consequential claims through a known official channel. Intended users are communications/security review teams. The prototype adds an evidence-organized triage step to manual listening; it does not replace an official verification channel. No customer study, procurement commitment, time saving or deployment outcome has been measured.

## Steps actually taken
1. Defined the audio modality, government communications scenario, deliverables and evaluation protocol from the competition brief.
2. Built a private local-processing interface: select/upload, play, guess, reveal, source label, review timestamps and JSON report.
3. Acquired a deterministic ASVspoof 2019 subset via ANC-Spoof; trained a 90-feature RBF SVM and separately calibrated it. Preserved those baseline outputs.
4. Tested a frozen public AASIST ONNX model. Neural weights were not fine-tuned. Its dated-benchmark behavior transferred poorly to the contemporary paired test, so it was not allowed to replace the primary model. It is used only as a one-way safety challenge to human-leaning uploads.
5. Acquired 1,866 contemporary-collection clips, with the publisher's real/fake labels and six synthetic platforms. Recorded repository revision, filenames, source groups and SHA-256 hashes. Source group assignment was written before feature extraction.
6. Added 120 LibriSpeech human clips and generated 120 text-matched Microsoft stock-voice counterparts. Source texts/IDs and human speakers are disjoint across development/test; stock voices are disjoint, while the Edge engine appears in training. No official's voice was cloned.
7. Sealed 1,093 training, 247 tuning, 281 calibration and 485 test clips. Hume is completely held out from model development. Synthetic numeric IDs are grouped conservatively across other public platforms; unknown shared scripts and speaker identities limit that diagnostic.
8. Trained on clean and 20 dB white-noise versions of training clips. Variants stay in their source partition. Standardization uses training only. Equal domain/class cell weights avoid overwhelming the smaller matched corpus. Compared nine SVM C/gamma candidates by mean tuning ROC-AUC across paired/public domains.
9. Fitted sigmoid calibration on separate calibration data with balanced domain/class weights. Selected conservative human/synthetic thresholds using calibration only. Wrote model and manifest hashes before first final-test features/predictions.
10. Evaluated the fixed model once on the new test. Primary: 40 human and 40 paired synthetic recordings. Secondary: 405 public-source clips. Evaluated fixed noise/MP3 stress conditions, and the old SVM on the same fresh primary test. Original announcements are separate previously inspected regression probes.
11. Exported the SVM for dependency-free JavaScript inference, checked numerical parity, all ten demo files, repeated runs, malformed input and runtime bounds. Kept the existing interface.
12. Added a result evidence layer without changing the frozen model: input/analysis format facts, RMS and clipping measurements, the two detector estimates on uploads, and a 26-band log-mel energy map.
13. Prepared source, measured results, this record and the five-page presentation. Hosting status is reported separately in the delivery message.

## Algorithms and rationale
16 kHz mono; subtract mean and normalize peak for feature extraction. 25 ms Hamming frames, 10 ms hop, 512-point FFT, 26 mel filters, 13 MFCCs. Means/standard deviations of MFCCs, first/second deltas and six spectral/time statistics produce 90 values. Standardized RBF SVM classifies the recording; logistic calibration maps its margin to a study-conditioned estimate. The feature front end is shared in mathematical form between Python and JavaScript and is numerically checked.

The visible frequency map presents normalized log energy across the same 26 mel bands. It helps a reviewer inspect bandwidth, silence and repeated structure, but it is not a feature attribution, calibrated evidence score or authenticity proof. The 2-second/1-second-hop timeline uses raw SVM margins. It is exploratory: whole-recording training does not establish window calibration or tamper localization. Quality gates abstain on heavy clipping, very short speech and extreme standardized feature distance; silence and invalid input return errors. No speech activity detector is implemented: music or non-speech can still be scored. No ENF, voice identity, cryptographic authentication, LLM, fact-checking or video detector is claimed. A simpler trained model was chosen through development measurements rather than adding algorithm count.

## Competition requirement mapping
- One modality done end to end: audio, with local upload, microphone capture and a ten-sample blind reel.
- Combined data: 1,866 public contemporary clips, 120 LibriSpeech human clips and 120 project-generated matched synthetic clips, with recorded splits and provenance.
- Detector: trained 90-feature calibrated RBF-SVM plus a frozen AASIST upload safety challenge.
- Locked validation: accuracy, precision, recall, F1, ROC-AUC, Brier score, confusion matrix, stress tests and selective coverage.
- Interface and evidence: audience guess before reveal, calibrated study estimate, source label, review timestamps, signal facts, frequency map and downloadable report.
- Error handling: input validation, quality abstention, neural safety challenge, replay-risk policy, worker timeout and repeat-run checks.

## Exploratory neural safety-gate evaluation
We evaluated the exact upload policy on the preserved 80-clip paired primary set. It accepted 75 clips (93.75% coverage) and was correct on 74/75 accepted clips (98.67%). One human-leaning clearance was challenged, while 33 RBF-SVM synthetic alerts missed by AASIST were preserved. AASIST alone achieved 56.25% forced-binary accuracy and 0.506 ROC-AUC on this shifted set. Because this check reused a test after failures had been inspected, it is post-test evidence and does not replace the locked v2 headline results. Per-file output and hashes are preserved in `research/posttest-aasist-primary-80.json`.

## Fresh primary results
Synthetic is positive. Table metrics force a 0.5 binary decision, including files the interface may mark uncertain.

| Condition | N | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| clean | 80 | 97.5% | 95.2% | 100.0% | 0.999 |
| noise_20db | 80 | 93.8% | 90.7% | 97.5% | 0.994 |
| mp3_32kbps | 80 | 92.5% | 88.6% | 97.5% | 0.979 |

Clean F1: 0.976. Brier: 0.017. Confusion matrix (actual human/synthetic rows, predicted human/synthetic columns): [[38, 2], [0, 40]]. Descriptive Wilson accuracy interval: 91.3%–99.3%; paired scripts/shared voices violate independent-clip assumptions, so it is not a deployment confidence bound.

Interface decision coverage: 95.0%; 4 uncertain, 76 accepted. Accuracy on accepted only: 98.7%. Do not report this selective accuracy without its coverage.

Public diagnostic clean accuracy: 94.8%, ROC-AUC 0.993; source group independence is narrower than speaker/text independence. Hume-only synthetic detection: 91.4% on 116 clips. This is recall on one synthetic platform, not balanced accuracy.

Original SVM on fresh primary: 47.5% accuracy. Original fictional-announcement regression probes: [{"id": "team-01", "probability": 0.9999602975920848, "accepted": true, "label": "fake"}, {"id": "team-02", "probability": 0.9999179613315262, "accepted": true, "label": "fake"}]. Their results do not count in the fresh test. Full per-file predictions and selected hyperparameters are in research/v2.

## Security and error handling actually implemented
Audio is decoded and scored in the browser; no recording is posted to a model service. The server serves static assets and can see ordinary page requests, not the chosen local file. An optional microphone mode requests access only after Start recording; captured audio stays local and tracks stop on completion or cancellation. No account, analytics, persistence or detector API key is used by the application. Content Security Policy restricts resources to this origin, denies objects/forms and allows local media blobs. User filenames/results use textContent rather than HTML injection.

Before decoding, enforce file size, recognized WAV/MP3 headers, supported WAV encoding/channels and duration bounds; truncated and malformed files produce recoverable errors. After decoding, check finite samples, duration and silence. Analysis runs in a terminable worker with a 45-second timeout; switching input cancels stale work; object URLs are revoked. Downloaded reports contain the filename, assessment and source label, so users should handle them appropriately.

This is defensive implementation and focused testing, not a penetration test. Browser codec behavior, unusual MP3 headers, accessibility and cross-device playback require rehearsal. A deliberate adversary or unfamiliar generator may evade the classifier. Authenticity remains unproved even with a high score. Study calibration assumes balanced classes and will shift with operational prevalence.

## Verification record
Automated checks: 66 passed, 0 failed (17 core, 13 v2, 10 recorder, 11 source/routing/safety-policy and 15 neural checks). The added checks validate both directions of the asymmetric neural safety policy, finite/bounded 26-band frequency-map output and MP3 rate/channel inspection. Native ONNX and browser WASM AASIST scores agree within the recorded tolerance. The actual worker paths evaluated all ten demo WAVs, repeat runs and the 60-second bound under Node. Python/JavaScript feature and probability tolerance is 1e-8. Browser rehearsal additionally confirmed that <code>sample 1.mp3</code> triggers a neural challenge and that a primary synthetic alert remains synthetic when AASIST leans human. These are not an accessibility audit or broad device-compatibility test.

## Jury demonstration
1. State the problem: a familiar voice can be synthetic; screening supports official-channel verification.
2. Choose a blind sample, play it, and ask Human / Synthetic / Not sure.
3. Reveal the assessment and source label. Show mistakes and abstentions honestly.
4. Play one highlighted interval. Explain that this is a listening aid, not proof of editing.
5. Upload a short WAV/MP3, show a recoverable invalid-input case and then a successful second run.
6. Show held-out accuracy plus confusion matrix and decision coverage; distinguish the 80-clip primary test from the 405-clip diagnostic.
7. Explain the next validation gap: independently collected government-domain, Arabic and channel-matched recordings from unseen synthesis systems.

## Resources and licences
The contemporary collection card declares CC-BY-4.0 and describes recordings collected in December 2024. It is more contemporary than ASVspoof 2019, not a comprehensive 2026 attack benchmark. The author's labels/licence declaration are not independent rights or forensic verification. LibriSpeech uses CC-BY-4.0. Source IDs/attribution are retained. Stock generated speech is subject to its provider's terms; it is not relicensed as the dataset's audio. Old ANC-Spoof packaging is marked CC-BY-NC-4.0; ASVspoof is ODC-By. Retain these restrictions for archived materials. AASIST/ONNX third-party notices remain bundled. Python dependencies are pinned in research/requirements.txt. Hosted execution uses only browser JavaScript for v2.

- Contemporary six-platform collection and dataset card: https://huggingface.co/datasets/garystafford/deepfake-audio-detection
- LibriSpeech corpus and licence: https://www.openslr.org/12
- LibriSpeech Hugging Face packaging: https://huggingface.co/datasets/openslr/librispeech_asr
- ASVspoof 2019 legacy benchmark: https://www.asvspoof.org/index2019.html
- ANC-Spoof legacy subset packaging: https://huggingface.co/datasets/abdulahh35/ANC-Spoof
- AASIST original architecture and source: https://github.com/clovaai/aasist
- Frozen AASIST ONNX checkpoint used for the upload second opinion: https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST
- scikit-learn support vector classification: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- edge-tts generation tool: https://github.com/rany2/edge-tts
- ONNX Runtime archived browser runtime: https://onnxruntime.ai/docs/get-started/with-javascript/web.html
