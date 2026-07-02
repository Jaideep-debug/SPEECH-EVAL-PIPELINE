"""
Entry point for the reproducible ASR inference + evaluation pipeline.

Usage:
    python run.py
    python run.py --model openai/whisper-small --num-samples 30
    python run.py --model facebook/wav2vec2-base-960h --num-samples 20

This single command:
  1. Downloads the model + processor from Hugging Face
  2. Loads a public speech dataset (LibriSpeech dummy split by default)
  3. Runs inference on N audio samples
  4. Computes WER / CER / latency
  5. Writes results/predictions.csv, results/metrics.json, results/report.md
"""

import argparse

from src.data import load_speech_dataset
from src.inference import run_inference
from src.evaluate import compute_metrics, write_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Speech model inference + evaluation pipeline")
    parser.add_argument("--model", default="openai/whisper-small", help="HF model id")
    parser.add_argument("--dataset", default="hf-internal-testing/librispeech_asr_dummy", help="HF dataset id")
    parser.add_argument("--config", default="clean", help="Dataset config name")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of audio samples to evaluate (20-50)")
    parser.add_argument("--output-dir", default="results", help="Directory to write outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Step 1/4: Loading {args.num_samples} samples from {args.dataset} ({args.config}/{args.split})")
    samples = load_speech_dataset(
        dataset_name=args.dataset,
        config=args.config,
        split=args.split,
        num_samples=args.num_samples,
    )

    print(f"Step 2/4: Running inference with {args.model}")
    results = run_inference(args.model, samples)

    print("Step 3/4: Computing evaluation metrics")
    metrics = compute_metrics(results, model_name=args.model, dataset_name=args.dataset)
    print(metrics)

    print("Step 4/4: Writing outputs")
    write_outputs(results, metrics, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
