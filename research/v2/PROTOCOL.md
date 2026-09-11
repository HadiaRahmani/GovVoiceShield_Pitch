# Locked evaluation protocol, v2

Written before accessing final-test features or predictions. Model choice uses development data only. The final test is not reused for tuning.

Primary endpoint: accuracy, synthetic precision/recall, F1, ROC-AUC, Brier score, confusion matrix and abstention coverage on the paired LibriSpeech/Edge test recordings. Human speakers and text IDs are disjoint from development; Edge stock voices are disjoint, but its synthesis engine is represented in training. This is a small English read-speech experiment, not a government deployment benchmark.

Secondary diagnostic: all held-out clips from the contemporary public collection. Report Hume separately as an unseen synthesis platform. Source recording groups are disjoint; speaker identities and shared scripts are unavailable, so full speaker/text independence cannot be asserted. Do not pool this diagnostic into the primary headline metric.

Training: the same 90 acoustic features as v1; RBF SVM candidates C={1,10,100}, gamma={0.003,0.01,0.03}. Fit scaling on training only. Equal weights for each domain/class cell. Include clean and deterministic 20 dB noise versions of training clips; no test augmentations enter training. Select by mean ROC-AUC across the two tuning domains. Sigmoid calibration uses a separate calibration partition, with balanced domain/class weights. Choose decision thresholds from calibration only, observed precision at least 90%, at least 10 decisions per side, human threshold at most 0.2 and synthetic threshold at least 0.8. Unsupported sides abstain. These thresholds do not guarantee 90% precision outside calibration.

Freeze and hash the exported model before loading final-test features. Evaluate clean, 20 dB added white noise and 32 kbps MP3. These are fixed, narrow stress tests. Report the original two fictional announcements separately as previously inspected regression probes. Select the first four primary-test IDs per class for the eight benchmark reel slots, independent of predictions; retain the two original fictional announcements. Show every failure.

Confidence intervals use Wilson's clip-level interval and are descriptive: paired scripts and voices create dependence. Calibration estimates assume the balanced study prior and are not operational probabilities. No ENF authentication claim: absent hum, mono conversion, pitch and loudness do not establish authenticity.
