---
license: cc-by-nc-4.0
task_categories:
  - audio-classification
language:
  - en
tags:
  - audio-deepfake-detection
  - anti-spoofing
  - neural-codec
  - speech
pretty_name: ANC-Spoof (NCSpoof)
size_categories:
  - 1M<n<10M
configs:
  - config_name: ASVspoof2019
    data_files:
      - split: train
        path: ASVspoof2019/train-*
      - split: dev
        path: ASVspoof2019/dev-*
      - split: eval
        path: ASVspoof2019/eval-*
  - config_name: FoR
    data_files:
      - split: data
        path: FoR/data-*
  - config_name: InTheWild
    data_files:
      - split: eval
        path: InTheWild/eval-*
---

# ANC-Spoof: Audio Neural Codec Spoof Dataset

## Overview

ANC-Spoof is a large-scale dataset for studying the
robustness of audio deepfake detection (ADD) systems against distortions introduced by
neural audio codecs. It pairs original (uncompressed) audio from three established ADD
benchmarks with codec-resynthesized versions of the same utterances, produced by eight
different neural codecs (including the uncompressed `Original` version).

The dataset is built from:
- **ASVspoof 2019 LA** — used for training, validation (`dev`), and evaluation
- **Fake-or-Real (FoR)** — used for cross-dataset evaluation only
- **In-the-Wild** — used for cross-dataset evaluation only

## Dataset Structure

The dataset is distributed as three configs, each independently downloadable:

| Config | Splits | Rows |
|---|---|---|
| `ASVspoof2019` | `train`, `dev`, `eval` | 971,688 |
| `FoR` | `data` | 37,072 |
| `InTheWild` | `eval` | 254,232 |

Only ASVspoof2019 has `train`/`dev` splits — FoR and InTheWild are provided purely as
out-of-domain generalization test sets, following standard practice in ADD research
(train on ASVspoof19, evaluate cross-dataset on FoR/In-the-Wild).

Each row represents one (utterance, codec) pair:

| Column | Type | Description |
|---|---|---|
| `audio` | `Audio` | The waveform, embedded (array + sampling rate) |
| `label` | `ClassLabel` | `0` = bonafide (real), `1` = spoof (fake) |
| `codec` | `string` | Which processing was applied: `Original`, `BigCodec`, `DAC`, `HiggsAudioV2`, `Mimi`, `SNAC`, `SpeechTokenizer`, `WavTokenizer` |
| `subset` | `string` | Source benchmark: `ASVspoof2019`, `FoR`, `InTheWild` |
| `split` | `string` | `train` / `dev` / `eval` |
| `speaker` | `string` or `null` | Speaker ID, where available |
| `utt_id` | `string` | Original utterance identifier |

## Usage

```python
from datasets import load_dataset

# Load only what you need — each call fetches just that config/split
asv_train = load_dataset("abdulahh35/ANC-Spoof", "ASVspoof2019", split="train")
asv_dev   = load_dataset("abdulahh35/ANC-Spoof", "ASVspoof2019", split="dev")
asv_eval  = load_dataset("abdulahh35/ANC-Spoof", "ASVspoof2019", split="eval")

for_eval  = load_dataset("abdulahh35/ANC-Spoof", "FoR", split="data")
wild_eval = load_dataset("abdulahh35/ANC-Spoof", "InTheWild", split="eval")

# Filter to a single codec, e.g. only BigCodec-compressed utterances
bigcodec_only = asv_eval.filter(lambda x: x["codec"] == "BigCodec")

row = asv_dev[0]
waveform, sr = row["audio"]["array"], row["audio"]["sampling_rate"]
label_str = asv_dev.features["label"].int2str(row["label"])  # "bonafide" or "spoof"
```

## Label Encoding

```python
ds.features["label"].names  # ['bonafide', 'spoof']
# 0 -> bonafide (real)
# 1 -> spoof (fake)
```

## Source Datasets

Please also cite the original benchmark datasets this data was derived from:
- ASVspoof 2019 (Wang et al., 2019)
- In-the-Wild (Müller et al., 2022)
- Fake-or-Real / FoR (Reimao & Tzerpos, 2019)

## License

<!-- Set this to match the actual license terms of the underlying source datasets
     (ASVspoof2019 / FoR / In-the-Wild) and your intended use — cc-by-nc-4.0 used
     above as a placeholder; update if that's not accurate. -->

## Citation

<!-- Add your paper's BibTeX entry here once available -->