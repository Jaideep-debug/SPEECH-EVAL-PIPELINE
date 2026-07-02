# Speech Model Inference & Evaluation Pipeline

A reproducible pipeline that downloads a pretrained speech model from Hugging Face,
runs inference on a public speech dataset, and evaluates the results with WER, CER,
and latency metrics.

- **Model:** `openai/whisper-small`
- **Dataset:** `hf-internal-testing/librispeech_asr_dummy` (a small LibriSpeech "clean"
  validation slice, already resampled to 16kHz — good for quick, reproducible runs)
- **Research summary:** see [`research_summary.md`](research_summary.md) (covers the
  Whisper paper: problem, architecture, comparison to prior approaches, training data,
  limitations)

## Project structure

```
.
├── README.md
├── requirements.txt
├── research_summary.md      # Part 1: paper summary
├── run.py                   # Part 4: single entrypoint, runs the full pipeline
├── src/
│   ├── data.py               # loads dataset from Hugging Face
│   ├── inference.py          # loads model/processor, runs inference
│   └── evaluate.py           # computes WER/CER/latency, writes reports
└── results/
    ├── predictions.csv       # audio_id, ground_truth, prediction
    ├── metrics.json          # WER, CER, latency, sample count
    └── report.md             # human-readable summary
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run.py
```

This downloads `openai/whisper-small` and its processor, loads 30 samples from the
LibriSpeech dummy dataset, runs inference, and writes results to `results/`.

### Optional flags

```bash
python run.py --model openai/whisper-small --num-samples 40
python run.py --model facebook/wav2vec2-base-960h --num-samples 20
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `openai/whisper-small` | Any Whisper or Wav2Vec2/HuBERT-CTC checkpoint on Hugging Face |
| `--dataset` | `hf-internal-testing/librispeech_asr_dummy` | HF dataset id |
| `--config` | `clean` | Dataset config |
| `--split` | `validation` | Dataset split |
| `--num-samples` | `30` | Number of audio samples to evaluate (20–50 per assignment spec) |
| `--output-dir` | `results` | Where to write outputs |

## Notes on model choice

Whisper-small was chosen over the CTC models (Wav2Vec2 / HuBERT) because:
- It handles punctuation/casing and general-domain audio more robustly out of the box.
- `facebook/hubert-base-ls960` is a self-supervised checkpoint without an ASR head,
  so it isn't directly usable for transcription without fine-tuning — swapping it in
  would require a fine-tuned variant (e.g. `facebook/hubert-large-ls960-ft`).
- `facebook/wav2vec2-base-960h` works out of the box too and is supported by this
  codebase (`--model facebook/wav2vec2-base-960h`) as a faster, lighter alternative
  for comparison.

## Metrics computed

- **Word Error Rate (WER)** and **Character Error Rate (CER)** via `jiwer`, with text
  normalization (lowercase, punctuation stripped) so formatting differences don't
  inflate error counts.
- **Average / min / max inference latency** per sample, measured wall-clock around
  the forward pass + decoding.
- **Number of processed samples** and full summary statistics, written to
  `results/metrics.json` and rendered in `results/report.md`.

Thanks,
Jai
