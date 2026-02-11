use anyhow::Result;
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager, PldaModule, Segmenter, read_wav};

fn main() -> Result<()> {
    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let (samples, sample_rate) = read_wav(&audio_path)?;
    let max_speakers = 6;

    let extractor = EmbeddingExtractor::new("src/nn/speaker_identification/model.bpk")?;
    let plda_module =
        PldaModule::load("plda.npz", "xvec_transform.npz").expect("Error loading PLDA");
    let mut manager = EmbeddingManager::new(max_speakers, Some(plda_module));
    let segmenter = Segmenter::new("src/nn/segmentation/model.bpk")?;

    for segment in segmenter.iter_segments(&samples, sample_rate)? {
        let segment = segment?;
        let embedding = extractor.extract(&segment.samples, sample_rate)?;

        let speaker = if manager.is_full() {
            manager.best_match(&embedding)
        } else {
            manager.upsert(&embedding, 0.5)
        }
        .map(|s| s.to_string())
        .unwrap_or_else(|| "?".into());

        println!(
            "start = {:.2}, end = {:.2}, speaker = {}",
            segment.start, segment.end, speaker
        );
    }

    Ok(())
}
