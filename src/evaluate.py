"""
Computes evaluation metrics from inference results and writes:
  results/predictions.csv
  results/metrics.json
  results/report.md
"""

import csv
import json
import statistics
from pathlib import Path

import jiwer

# Standard text normalization so casing/punctuation don't inflate error rates.
NORMALIZER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

CER_NORMALIZER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def compute_metrics(results: list, model_name: str, dataset_name: str):
    refs = [r["ground_truth"] for r in results]
    hyps = [r["prediction"] for r in results]
    latencies = [r["latency_seconds"] for r in results]

    wer = jiwer.wer(
        refs, hyps,
        truth_transform=NORMALIZER, hypothesis_transform=NORMALIZER,
    )
    cer = jiwer.wer(
        refs, hyps,
        truth_transform=CER_NORMALIZER, hypothesis_transform=CER_NORMALIZER,
    )

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": len(results),
        "word_error_rate": round(wer, 4),
        "character_error_rate": round(cer, 4),
        "avg_inference_latency_seconds": round(statistics.mean(latencies), 4),
        "min_latency_seconds": round(min(latencies), 4),
        "max_latency_seconds": round(max(latencies), 4),
        "total_inference_time_seconds": round(sum(latencies), 4),
    }
    return metrics


def write_outputs(results: list, metrics: dict, output_dir: str = "results"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    csv_path = out / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "ground_truth", "prediction"])
        writer.writeheader()
        for r in results:
            writer.writerow(
                {"audio_id": r["audio_id"], "ground_truth": r["ground_truth"], "prediction": r["prediction"]}
            )

    # metrics.json
    json_path = out / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # report.md
    report_path = out / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Evaluation Report\n\n")
        f.write(f"- **Model:** {metrics['model']}\n")
        f.write(f"- **Dataset:** {metrics['dataset']}\n")
        f.write(f"- **Samples processed:** {metrics['num_samples']}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Word Error Rate (WER) | {metrics['word_error_rate']} |\n")
        f.write(f"| Character Error Rate (CER) | {metrics['character_error_rate']} |\n")
        f.write(f"| Avg inference latency (s) | {metrics['avg_inference_latency_seconds']} |\n")
        f.write(f"| Min latency (s) | {metrics['min_latency_seconds']} |\n")
        f.write(f"| Max latency (s) | {metrics['max_latency_seconds']} |\n")
        f.write(f"| Total inference time (s) | {metrics['total_inference_time_seconds']} |\n\n")
        f.write("## Sample Predictions\n\n")
        f.write("| audio_id | ground_truth | prediction |\n|---|---|---|\n")
        for r in results[:10]:
            f.write(f"| {r['audio_id']} | {r['ground_truth']} | {r['prediction']} |\n")

    print(f"Wrote: {csv_path}, {json_path}, {report_path}")
    return csv_path, json_path, report_path
