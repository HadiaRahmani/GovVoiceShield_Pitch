from pathlib import Path
import json,html,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT.parent
e=json.loads((ROOT/'dist/evaluation.json').read_text());t=json.loads((ROOT/'tests/v2-results.json').read_text());post=json.loads((ROOT/'research/posttest-aasist-primary-80.json').read_text());c=e['conditions']['clean'];p=lambda x:f'{100*x:.1f}%'
sources=[('Contemporary six-platform collection and dataset card','https://huggingface.co/datasets/garystafford/deepfake-audio-detection'),('LibriSpeech corpus and licence','https://www.openslr.org/12'),('LibriSpeech Hugging Face packaging','https://huggingface.co/datasets/openslr/librispeech_asr'),('ASVspoof 2019 legacy benchmark','https://www.asvspoof.org/index2019.html'),('ANC-Spoof legacy subset packaging','https://huggingface.co/datasets/abdulahh35/ANC-Spoof'),('AASIST original architecture and source','https://github.com/clovaai/aasist'),('Archived AASIST ONNX checkpoint','https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST'),('scikit-learn support vector classification','https://scikit-learn.org/stable/modules/svm.html'),('scikit-learn probability calibration','https://scikit-learn.org/stable/modules/calibration.html'),('edge-tts generation tool','https://github.com/rany2/edge-tts'),('ONNX Runtime archived browser runtime','https://onnxruntime.ai/docs/get-started/with-javascript/web.html')]
rows='\n'.join(f"| {k} | {v['n']} | {p(v['accuracy'])} | {p(v['precision'])} | {p(v['recall'])} | {v['roc_auc']:.3f} |" for k,v in e['conditions'].items())
record=f'''# GovVoice Shield — build record

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
12. Prepared source, measured results, this record and the five-page presentation. Hosting status is reported separately in the delivery message.

## Algorithms and rationale
16 kHz mono; subtract mean and normalize peak for feature extraction. 25 ms Hamming frames, 10 ms hop, 512-point FFT, 26 mel filters, 13 MFCCs. Means/standard deviations of MFCCs, first/second deltas and six spectral/time statistics produce 90 values. Standardized RBF SVM classifies the recording; logistic calibration maps its margin to a study-conditioned estimate. The feature front end is shared in mathematical form between Python and JavaScript and is numerically checked.

The 2-second/1-second-hop timeline uses raw SVM margins. It is exploratory: whole-recording training does not establish window calibration or tamper localization. Quality gates abstain on heavy clipping, very short speech and extreme standardized feature distance; silence and invalid input return errors. No speech activity detector is implemented: music or non-speech can still be scored. No ENF, voice identity, cryptographic authentication, LLM, fact-checking or video detector is claimed. A simpler trained model was chosen through development measurements rather than adding algorithm count.

## Exploratory neural safety-gate evaluation
The exact one-way upload safety policy was evaluated on the preserved 80-clip paired primary set. It accepted {post['cascade']['accepted']} clips ({100*post['cascade']['coverage']:.2f}% coverage) with {100*post['cascade']['accuracy_when_decided']:.2f}% accuracy when decided. AASIST alone achieved {100*post['aasist_forced_binary']['accuracy']:.2f}% forced-binary accuracy and {post['aasist_forced_binary']['roc_auc']:.3f} ROC-AUC. Because this check reused a test after failures had been inspected, it is post-test evidence and does not replace the locked v2 headline results. Per-file output and hashes are preserved in research/posttest-aasist-primary-80.json.

## Fresh primary results
Synthetic is positive. Table metrics force a 0.5 binary decision, including files the interface may mark uncertain.

| Condition | N | Accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
{rows}

Clean F1: {c['f1']:.3f}. Brier: {c['brier']:.3f}. Confusion matrix (actual human/synthetic rows, predicted human/synthetic columns): {c['confusion_matrix']}. Descriptive Wilson accuracy interval: {p(c['accuracy_ci95'][0])}–{p(c['accuracy_ci95'][1])}; paired scripts/shared voices violate independent-clip assumptions, so it is not a deployment confidence bound.

Interface decision coverage: {p(e['selective']['coverage'])}; {e['selective']['uncertain']} uncertain, {e['selective']['accepted']} accepted. Accuracy on accepted only: {p(e['selective']['accuracy']) if e['selective']['accuracy'] is not None else 'not available'}. Do not report this selective accuracy without its coverage.

Public diagnostic clean accuracy: {p(e['public_diagnostic']['clean']['accuracy'])}, ROC-AUC {e['public_diagnostic']['clean']['roc_auc']:.3f}; source group independence is narrower than speaker/text independence. Hume-only synthetic detection: {p(e['public_platforms']['hu']['binary_accuracy'])} on {e['public_platforms']['hu']['n']} clips. This is recall on one synthetic platform, not balanced accuracy.

Original SVM on fresh primary: {p(e['prior_svm_on_fresh_primary']['accuracy'])} accuracy. Original fictional-announcement regression probes: {json.dumps(e['regression_probes'])}. Their results do not count in the fresh test. Full per-file predictions and selected hyperparameters are in research/v2.

## Security and error handling actually implemented
Audio is decoded and scored in the browser; no recording is posted to a model service. The server serves static assets and can see ordinary page requests, not the chosen local file. An optional microphone mode requests access only after Start recording; captured audio stays local and tracks stop on completion or cancellation. No account, analytics, persistence or detector API key is used by the application. Content Security Policy restricts resources to this origin, denies objects/forms and allows local media blobs. User filenames/results use textContent rather than HTML injection.

Before decoding, enforce file size, recognized WAV/MP3 headers, supported WAV encoding/channels and duration bounds; truncated and malformed files produce recoverable errors. After decoding, check finite samples, duration and silence. Analysis runs in a terminable worker with a 45-second timeout; switching input cancels stale work; object URLs are revoked. Downloaded reports contain the filename, assessment and source label, so users should handle them appropriately.

This is defensive implementation and focused testing, not a penetration test. Browser codec behavior, unusual MP3 headers, accessibility and cross-device playback require rehearsal. A deliberate adversary or unfamiliar generator may evade the classifier. Authenticity remains unproved even with a high score. Study calibration assumes balanced classes and will shift with operational prevalence.

## Verification record
V2 checks: {t['passed']} passed, {t['failed']} failed. Actual worker handler evaluated all ten WAVs under Node; 60-second audio processed in {t['seconds60']:.2f} seconds on this machine. Python/JavaScript feature and probability tolerance 1e-8. See tests/v2-results.json for each demo outcome. Core tests also cover parser boundaries and source-level UI checks. These are not an automated browser interaction, accessibility audit or user acceptance test. The optional WebMCP hooks are progressive enhancements and were not validated in a supported browser context.

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

'''+ '\n'.join(f'- {name}: {url}' for name,url in sources)+'\n'
(ROOT/'BUILD_RECORD.md').write_text(record,encoding='utf-8')
body=html.escape(record);body='\n'.join('<h2>'+line[3:]+'</h2>' if line.startswith('## ') else '<h1>'+line[2:]+'</h1>' if line.startswith('# ') else '<p>'+line+'</p>' if line else '' for line in body.splitlines())
links=''.join(f'<li><a href="{u}">{html.escape(n)}</a></li>' for n,u in sources)
(ROOT/'dist/resources.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GovVoice build record</title><link rel="stylesheet" href="style.css"></head><body><header><a href="./">← Back to GovVoice Shield</a></header><main class="record"><section class="panel">'+body+'<h2>Open original sources</h2><ul>'+links+'</ul></section></main></body></html>',encoding='utf-8')
with (ROOT/'dist/style.css').open('a',encoding='utf-8') as f:f.write('\n.record{max-width:1050px}.record .panel{padding:36px}.record p{line-height:1.65;overflow-wrap:anywhere}.record h2{margin-top:36px}\n')
print('Build record and linked resources written')
