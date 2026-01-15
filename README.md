# native-pyannote-rs

Pyannote audio diarization in Rust.

This is a fork of https://github.com/thewh1teagle/pyannote-rs with Rust native crate for audio feature extraction using [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank) instead of bindings to C++ variant (knf-rs).

## Features

- Compute 1 hour of audio in less than a minute on CPU.
- Faster performance with DirectML on Windows and CoreML on macOS.
- Accurate timestamps with Pyannote segmentation.
- Identify speakers with wespeaker embeddings.

## Examples

See [examples](examples)

## Usage

```rust
use pyannote_rs::{read_wav, EmbeddingExtractor, EmbeddingManager, Segmenter};

fn main() -> eyre::Result<()> {
    let (samples, sample_rate) = read_wav("audio.wav")?;
    let mut segmenter = Segmenter::new("segmentation-3.0.onnx")?;
    let mut extractor = EmbeddingExtractor::new("wespeaker_en_voxceleb_CAM++.onnx")?;
    let mut speakers = EmbeddingManager::new(4);

    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        let segment = segment?;
        let embedding = extractor.extract(&segment.samples, sample_rate)?;
        let speaker_id = speakers.upsert(&embedding, 0.5).unwrap_or(0);

        println!("{:.2}-{:.2} => speaker {}", segment.start, segment.end, speaker_id);
    }

    Ok(())
}
```

<details>
<summary>How it works</summary>

pyannote-rs uses 2 models for speaker diarization:

1. **Segmentation**: [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) identifies when speech occurs.
2. **Speaker Identification**: [wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) identifies who is speaking.

Inference is powered by [onnxruntime](https://onnxruntime.ai/).

- The segmentation model processes up to 10s of audio, using a sliding window approach (iterating in chunks).
- The embedding model processes filter banks (audio features) extracted with [kaldi-native-fbank](https://crates.io/crates/kaldi-native-fbank).

Speaker comparison (e.g., determining if Alice spoke again) is done using cosine similarity.
</details>

## Credits

Big thanks to [pyannote-onnx](https://github.com/pengzhendong/pyannote-onnx) and [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
