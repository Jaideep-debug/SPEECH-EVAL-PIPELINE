"""
Loads a small public speech dataset from Hugging Face.

Default: hf-internal-testing/librispeech_asr_dummy
  - A tiny (~9MB) validation slice of LibriSpeech "clean", hosted specifically
    for quick demos/tests. Audio is already 16kHz mono, which matches what
    Whisper / Wav2Vec2 / HuBERT expect, so no resampling code is required.

You can swap in the full "librispeech_asr" (clean) or "mozilla-foundation/common_voice_16_1"
by passing --dataset / --config / --split on the command line (see run.py).
"""

from datasets import load_dataset


def load_speech_dataset(
    dataset_name: str = "hf-internal-testing/librispeech_asr_dummy",
    config: str = "clean",
    split: str = "validation",
    num_samples: int = 30,
):
    """
    Returns a list of dicts: {"audio_id": str, "audio": np.ndarray, "sampling_rate": int, "reference": str}
    """
    ds = load_dataset(dataset_name, config, split=split)

    if num_samples is not None:
        num_samples = min(num_samples, len(ds))
        ds = ds.select(range(num_samples))

    samples = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        # LibriSpeech-style datasets store the transcript under "text";
        # Common Voice uses "sentence" -- handle both so the pipeline
        # is portable across datasets.
        reference = row.get("text") or row.get("sentence") or ""
        samples.append(
            {
                "audio_id": f"{i:04d}",
                "audio": audio["array"],
                "sampling_rate": audio["sampling_rate"],
                "reference": reference.strip(),
            }
        )
    return samples
