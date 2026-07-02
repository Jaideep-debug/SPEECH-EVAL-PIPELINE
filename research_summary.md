# Research Summary: Whisper (Radford et al., 2022)

**Paper:** "Robust Speech Recognition via Large-Scale Weak Supervision" — OpenAI, 2022.

## What problem does it solve?

Prior speech recognition systems performed well on the specific benchmark they were
trained on but degraded sharply on unfamiliar accents, background noise, technical
vocabulary, or recordings from a different domain than their training data. Building a
model that generalizes across these conditions typically required either large amounts
of carefully labeled data per domain, or fine-tuning separately for each new setting.
Whisper aims to produce a single ASR (and speech translation) model that performs
robustly "out of the box," without any dataset-specific fine-tuning.

## How does the architecture work?

Whisper uses a standard Transformer encoder-decoder (sequence-to-sequence) design,
the same family of architecture used in machine translation. Audio is first converted
into a log-Mel spectrogram, which the encoder processes into a sequence of hidden
representations. The decoder then autoregressively generates text tokens, conditioned
on the encoder output and on special task tokens that tell the model which job to do:
transcribe, translate, detect language, or predict timestamps. Framing multiple tasks
(transcription, translation, language ID, voice activity) as one multitask token
sequence lets a single model handle all of them without task-specific architectures.

## Why is it better than previous approaches?

The key change is scale and diversity of weakly-labeled data rather than a novel model
component. Instead of relying on small, cleanly transcribed corpora, Whisper was trained
on 680,000 hours of audio paired with transcripts scraped from the internet, covering
many languages, accents, and recording conditions, with only lightweight automated
filtering. This scale and diversity substitutes for heavy supervised fine-tuning: the
model learns robustness directly from seeing messy, varied real-world audio during
pretraining, so it transfers to new domains in a zero-shot setting better than models
trained on smaller, cleaner, single-domain datasets.

## What datasets were used?

Training data was 680,000 hours of audio-transcript pairs collected from the web,
spanning roughly a third non-English audio and including speech translation pairs.
No single canonical benchmark dataset was used for training; instead, evaluation was
done zero-shot on many existing benchmarks (LibriSpeech, Common Voice, TED-LIUM,
switchboard-style conversational data, and others) to measure generalization rather
than in-domain accuracy.

## What are its limitations?

- Larger compute and latency cost than lightweight CTC models like base Wav2Vec2,
  since it autoregressively decodes text token by token.
- Can "hallucinate" fluent but incorrect text during silence or non-speech audio,
  a known failure mode of large sequence-to-sequence decoders.
- Performance is uneven across languages, correlating with how much training data
  existed for each language on the web.
- Because training transcripts were scraped rather than professionally verified,
  quality varies, and some biases or errors from web data can be inherited.

*(Note: This is a paraphrased summary for the assignment's research component, not
a reproduction of the original paper text.)*
