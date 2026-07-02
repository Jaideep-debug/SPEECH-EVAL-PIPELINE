"""
Downloads a pretrained speech model + processor from Hugging Face and runs
inference on a list of audio samples.

Supports two model families out of the box:
  - Seq2Seq ASR models (Whisper): openai/whisper-small, openai/whisper-tiny, ...
  - CTC models (Wav2Vec2 / HuBERT): facebook/wav2vec2-base-960h, facebook/hubert-base-ls960
    NOTE: hubert-base-ls960 is a self-supervised base checkpoint without an ASR head;
    for CER/WER evaluation use an ASR-finetuned HuBERT (e.g. facebook/hubert-large-ls960-ft)
    if you pick the HuBERT route.
"""

import time
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

WHISPER_MODELS = {"openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium"}


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name in WHISPER_MODELS or "whisper" in model_name:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model_type = "whisper"
    else:
        # wav2vec2 / hubert-style CTC models
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        model_type = "ctc"

    model.eval()
    return model, processor, model_type, device


def transcribe(model, processor, model_type, device, audio, sampling_rate):
    """Runs inference on a single audio array. Returns (prediction_text, latency_seconds)."""
    start = time.perf_counter()

    if model_type == "whisper":
        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    else:
        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(predicted_ids)[0]

    latency = time.perf_counter() - start
    return prediction.strip(), latency


def run_inference(model_name: str, samples: list):
    """
    samples: list of dicts from src.data.load_speech_dataset
    Returns list of dicts: {"audio_id", "ground_truth", "prediction", "latency_seconds"}
    """
    model, processor, model_type, device = load_model(model_name)
    print(f"Loaded {model_name} ({model_type}) on {device}")

    results = []
    for sample in samples:
        prediction, latency = transcribe(
            model, processor, model_type, device, sample["audio"], sample["sampling_rate"]
        )
        results.append(
            {
                "audio_id": sample["audio_id"],
                "ground_truth": sample["reference"],
                "prediction": prediction,
                "latency_seconds": latency,
            }
        )
        print(f"  [{sample['audio_id']}] ref='{sample['reference'][:40]}...' pred='{prediction[:40]}...' ({latency:.3f}s)")

    return results
