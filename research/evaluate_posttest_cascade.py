"""Post-test robustness study for the already-deployed upload cascade.

This does not replace or reopen the locked v2 RBF-SVM experiment. It evaluates
the frozen public AASIST checkpoint and the exact one-way safety policy
on the preserved 80-clip paired primary test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_mono_16k(path: Path) -> np.ndarray:
    wave, rate = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    if rate != 16000:
        # Deterministic linear interpolation is sufficient for this preserved set;
        # the expected primary files are already 16 kHz.
        old = np.arange(len(wave), dtype=np.float64) / rate
        new = np.arange(round(len(wave) * 16000 / rate), dtype=np.float64) / 16000
        wave = np.interp(new, old, wave).astype(np.float32)
    return np.asarray(wave, dtype=np.float32)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    # Pairwise definition with half credit for ties; small N keeps this explicit.
    pos = p[y == 1]
    neg = p[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p >= 0.5
    tn = int(np.sum((y == 0) & (~pred)))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & (~pred)))
    tp = int(np.sum((y == 1) & pred))
    return {
        "accuracy": float((tn + tp) / len(y)),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "roc_auc": auc(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "confusion_matrix_actual_rows_predicted_columns": [[tn, fp], [fn, tp]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "research" / "posttest-aasist-primary-80.json")
    args = parser.parse_args()

    manifest_path = ROOT / "research" / "v2" / "completed-manifest.json"
    predictions_path = ROOT / "research" / "v2" / "final-predictions.json"
    model_path = ROOT / "dist" / "aasist.onnx"
    config_path = ROOT / "dist" / "neural-config.json"
    manifest = {row["id"]: row for row in json.loads(manifest_path.read_text(encoding="utf-8"))}
    primary = [
        row for row in json.loads(predictions_path.read_text(encoding="utf-8"))
        if row["condition"] == "clean" and row["domain"] == "paired"
    ]
    if len(primary) != 80 or sum(row["label"] for row in primary) != 40:
        raise RuntimeError("Expected the preserved balanced 80-clip primary test")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    waves = []
    paths = []
    for row in primary:
        item = manifest[row["id"]]
        path = args.audio_dir / item["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
        waves.append(load_mono_16k(path))

    n = int(cfg["samples"])
    chunks, owners = [], []
    for owner, wave in enumerate(waves):
        for start in range(0, len(wave), n):
            tail = wave[start : start + n]
            chunks.append(np.resize(tail, n))
            owners.append(owner)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    raw = np.zeros(len(waves), dtype=np.float64)
    counts = np.zeros(len(waves), dtype=np.int64)
    for start in range(0, len(chunks), 8):
        batch = np.asarray(chunks[start : start + 8], dtype=np.float32)
        logits = session.run(None, {"wav": batch})[0]
        for offset, values in enumerate(logits):
            owner = owners[start + offset]
            raw[owner] -= float(values[1])
            counts[owner] += 1
    raw /= counts

    cal = cfg["calibration"]
    linear = np.clip(cal["slope"] * raw + cal["intercept"], -50, 50)
    neural = 1.0 / (1.0 + np.exp(-linear))
    y = np.asarray([row["label"] for row in primary], dtype=np.int64)
    feature = np.asarray([row["probability"] for row in primary], dtype=np.float64)
    feature_accepted = np.asarray([row["accepted"] for row in primary], dtype=bool)
    agreement = (feature >= 0.5) == (neural >= 0.5)
    # Asymmetric safety rule: preserve every primary synthetic alert. A primary
    # human result is accepted only when AASIST is confidently human.
    accepted = feature_accepted & ((feature >= 0.5) | (neural <= cfg["thresholds"]["real"]))
    decision = feature >= 0.5
    decided_correct = (decision == y) & accepted

    records = []
    for i, row in enumerate(primary):
        records.append({
            "id": row["id"],
            "label": int(y[i]),
            "file_sha256": sha256(paths[i]),
            "rbf_probability": float(feature[i]),
            "rbf_accepted": bool(feature_accepted[i]),
            "aasist_probability": float(neural[i]),
            "aasist_raw_score": float(raw[i]),
            "detectors_agree": bool(agreement[i]),
            "cascade_accepted": bool(accepted[i]),
            "cascade_correct_if_accepted": bool(decided_correct[i]) if accepted[i] else None,
        })

    result = {
        "study": "Post-test robustness evaluation of the deployed upload cascade",
        "status": "Exploratory; performed after inspection of user-reported failures. Does not replace locked v2 headline metrics.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {"total": 80, "human": 40, "synthetic": 40, "domain": "paired English read speech"},
        "frozen_inputs": {
            "aasist_sha256": sha256(model_path),
            "neural_config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
            "v2_predictions_sha256": sha256(predictions_path),
        },
        "aasist_forced_binary": binary_metrics(y, neural),
        "cascade": {
            "policy": "Preserve every accepted RBF-SVM synthetic alert. Release a human result only when the RBF-SVM quality decision is accepted and AASIST is confidently human (p <= 0.49); otherwise escalate.",
            "coverage": float(np.mean(accepted)),
            "accepted": int(np.sum(accepted)),
            "uncertain_or_suspicious": int(np.sum(~accepted)),
            "accuracy_when_decided": float(np.sum(decided_correct) / np.sum(accepted)) if np.any(accepted) else None,
            "errors_when_decided": int(np.sum(accepted & ~(decision == y))),
            "detector_disagreements": int(np.sum(~agreement)),
            "human_results_challenged": int(np.sum(feature_accepted & (feature < 0.5) & (neural > cfg["thresholds"]["real"]))),
            "synthetic_alerts_preserved_despite_neural_human_direction": int(np.sum(feature_accepted & (feature >= 0.5) & (neural < 0.5))),
        },
        "limitations": [
            "This test was reused after model failures were inspected, so it is not a fresh confirmatory test.",
            "AASIST calibration was fitted in the archived v1 experiment and was not refitted for v2.",
            "The sample is English paired read speech and does not validate Arabic, replay, telephone, social-media or government-domain audio.",
            "Selective accuracy must always be reported with coverage.",
        ],
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"aasist": result["aasist_forced_binary"], "cascade": result["cascade"]}, indent=2))


if __name__ == "__main__":
    main()
