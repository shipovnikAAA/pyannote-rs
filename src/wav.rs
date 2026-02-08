use anyhow::{Result, Context};
use hound::WavReader;

pub fn read_wav(file_path: &str) -> Result<(Vec<i16>, u32)> {
    let mut reader = WavReader::open(file_path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<i16> = reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?;

    Ok((samples, sample_rate))
}

pub fn read_wav_optimized(file_path: &str) -> Result<(Vec<i16>, u32)> {
    let mut reader = WavReader::open(file_path)?;
    let spec = reader.spec();
    let original_sample_rate = spec.sample_rate;
    let channels = spec.channels;
    let raw_samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to read samples")?;

    let mono_samples = if channels > 1 {
        raw_samples
            .chunks_exact(channels as usize)
            .map(|chunk| {
                let sum: i32 = chunk.iter().map(|&s| s as i32).sum();
                (sum / channels as i32) as i16
            })
            .collect()
    } else {
        raw_samples
    };

    let target_rate = 16000;
    if original_sample_rate == target_rate {
        return Ok((mono_samples, target_rate));
    }

    let ratio = original_sample_rate as f64 / target_rate as f64;
    let target_len = (mono_samples.len() as f64 / ratio).floor() as usize;
    let mut resampled_samples = Vec::with_capacity(target_len);

    for i in 0..target_len {
        let pos = i as f64 * ratio;
        let index = pos.floor() as usize;
        let fraction = pos - index as f64;

        if index + 1 < mono_samples.len() {
            let sample_a = mono_samples[index] as f64;
            let sample_b = mono_samples[index + 1] as f64;
            let interpolated = sample_a + (sample_b - sample_a) * fraction;
            resampled_samples.push(interpolated as i16);
        } else {
            resampled_samples.push(mono_samples[index]);
        }
    }

    Ok((resampled_samples, target_rate))
}
