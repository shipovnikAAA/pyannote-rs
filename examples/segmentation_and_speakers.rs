use anyhow::{Result, anyhow};
use hound::{WavSpec, WavWriter};
use pyannote_rs::{
    EmbeddingExtractor, EmbeddingManager, PldaModule, Segmenter, UpdateStrategy, read_wav_optimized,
};
use std::thread;
use std::{fs, path::Path};

fn main() -> anyhow::Result<()> {
    let child = thread::Builder::new()
        .stack_size(100 * 1024 * 1024)
        .spawn(|| run_diarization())?;

    child
        .join()
        .map_err(|e| anyhow::anyhow!("Thread panicked: {:?}", e))?
}

pub fn write_wav(file_path: &str, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(file_path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn run_diarization() -> Result<()> {
    // path to file
    // let audio_path = "6_speakers.wav";
    let audio_path = "segment_1.wav";

    // reading audio
    let (samples, sample_rate) = read_wav_optimized(&audio_path)?;

    // init models
    let extractor = EmbeddingExtractor::new("src/nn/speaker_identification/model.bpk")?;
    let plda_module = PldaModule::load("src/nn/plda/plda.npz", "src/nn/plda/xvec_transform.npz")
        .expect("Error loading PLDA");
    let mut manager = EmbeddingManager::new(6, Some(plda_module));
    // let mut manager = EmbeddingManager::new(6, None);
    let segmenter = Segmenter::new("src/nn/segmentation/model.bpk")?;

    let file_stem = Path::new(&audio_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("error getting file stem"))?;

    let output_folder = format!("{}_segments", file_stem);
    fs::create_dir_all(&output_folder)?;

    println!("strating diarization...");

    // main loop over segments
    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        let segment = segment?;

        // extract embedding
        let embedding = extractor.extract(&segment.samples, sample_rate)?;

        // set id for speaker
        let speaker_id = if manager.is_full() {
            manager.best_match(&embedding)
        } else {
            manager.upsert(&embedding, 0.5, UpdateStrategy::None)
        }
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".into());

        let file_name = format!(
            "{}/start_{:.2}_end_{:.2}_{}.wav",
            output_folder, segment.start, segment.end, speaker_id
        );

        // saving files
        write_wav(&file_name, &segment.samples, sample_rate)?;

        println!("saved: {}", file_name);
    }

    println!("Done! Output folder: {}", output_folder);
    Ok(())
}
