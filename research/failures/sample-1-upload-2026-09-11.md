# User-supplied synthetic upload failure

- Reported: 2026-09-11
- Supplied filename: `sample 1.mp3`
- User-provided label: synthetic, described as a prompt-generated voice
- SHA-256: `4A4283941A896C32AAE8F3046E92BE687F2C48022D424573485E470FEE0AE2FF`
- Duration after decoding: 12.771 seconds
- Frozen v2 RBF-SVM result: human, synthetic estimate 0.0002685744, raw SVM score -0.7058254
- Frozen AASIST second opinion: synthetic estimate 0.9977999, raw neural score 1.9546433
- Short-window observation: 10 of 11 two-second RBF-SVM margins were positive; a separate audit found this pattern common in known-human clips, so it was rejected as a decision rule

The source label is supplied by the user and has not been independently authenticated. The file is a post-test regression case and is excluded from all frozen v2 measurements.

## Product action

For user uploads only, the RBF-SVM and frozen AASIST neural detector now run locally. AASIST acts as a one-way safety challenge: it can stop a human clearance, but it cannot erase an RBF-SVM synthetic warning. The interface preserves both probabilities and records `policyReason: neural-safety-challenge` when the human result is challenged. The supplied file now follows this path instead of being presented as human. Neither detector is treated as proof of authenticity.

## Research implication

The useful evidence is a neural challenge to a human clearance, while the rejected window rule would have flagged 64 known-human test clips and only one missed synthetic clip. The one-way policy was subsequently evaluated as a clearly labelled post-test robustness study; a future v3 protocol still needs Arabic, replayed, government-domain and unseen-generator recordings using a new development set and a fresh untouched test set before confirmatory claims are made.
