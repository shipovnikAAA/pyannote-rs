use crate::nn::{self, BurnBackend, BurnDevice};
use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Tensor, TensorData};
use std::{cmp::Ordering, collections::VecDeque, path::Path};

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub samples: Vec<i16>,
}

const FRAME_SIZE: usize = 270;
const FRAME_START: usize = 721;

#[derive(Debug)]
pub struct Segmenter {
    model: nn::segmentation::Model<BurnBackend>,
    device: BurnDevice,
    batch_size: usize,
    threshold: f32,
    max_silence_frames: usize,
    max_segment_frames: usize,
}

#[derive(Debug, Clone)]
struct SegmentationState {
    active_class: Option<usize>,
    start_sample: usize,
    last_valid_sample: usize,
    silence_counter: usize,
    frames_count: usize,
    max_silence_frames: usize,
    max_segment_frames: usize,
}

pub struct SegmentIterator<'a> {
    segmenter: &'a Segmenter,
    padded_samples: Vec<i16>,
    sample_rate: u32,
    window_size: usize,
    current_window_start: usize,
    segments_queue: VecDeque<Segment>,
    finished: bool,
    state: SegmentationState,
}

impl SegmentationState {
    fn new(max_silence: usize, max_segment: usize) -> Self {
        Self {
            active_class: None,
            start_sample: 0,
            last_valid_sample: 0,
            silence_counter: 0,
            frames_count: 0,
            max_silence_frames: max_silence,
            max_segment_frames: max_segment,
        }
    }

    fn process_frame(&mut self, class: Option<usize>, sample_idx: usize) -> Option<(usize, usize)> {
        match (self.active_class, class) {
            // current class
            (Some(p), Some(c)) if p == c => {
                self.last_valid_sample = sample_idx + FRAME_SIZE;
                self.silence_counter = 0;
                self.frames_count += 1;

                if self.frames_count > self.max_segment_frames {
                    return self.emit_and_reset(Some(c), sample_idx);
                }
            }
            // other class
            (Some(_), Some(c)) => return self.emit_and_reset(Some(c), sample_idx),
            // silence
            (Some(_), None) => {
                self.silence_counter += 1;
                if self.silence_counter >= self.max_silence_frames {
                    return self.emit_and_reset(None, 0);
                }
            }
            // new class
            (None, Some(c)) => {
                self.active_class = Some(c);
                self.start_sample = sample_idx;
                self.last_valid_sample = sample_idx + FRAME_SIZE;
            }
            _ => {}
        }
        None
    }

    fn emit_and_reset(
        &mut self,
        new_class: Option<usize>,
        sample_idx: usize,
    ) -> Option<(usize, usize)> {
        let res = (self.start_sample, self.last_valid_sample);
        self.active_class = new_class;
        self.start_sample = sample_idx;
        self.last_valid_sample = sample_idx + FRAME_SIZE;
        self.silence_counter = 0;
        self.frames_count = 0;
        Some(res)
    }
}

impl<'a> Iterator for SegmentIterator<'a> {
    type Item = Result<Segment>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(segment) = self.segments_queue.pop_front() {
                return Some(Ok(segment));
            }

            if self.finished {
                return None;
            }

            let raw_outputs = match self.run_inference_batch() {
                Ok(Some(outputs)) => outputs,
                Ok(None) => {
                    self.finished = true;
                    self.flush_state();
                    continue;
                }
                Err(e) => return Some(Err(e)),
            };

            for (class, sample_idx) in raw_outputs {
                if let Some((s, e)) = self.state.process_frame(class, sample_idx) {
                    if e > s + 1000 {
                        self.push_segment(s, e);
                    }
                }
            }
        }
    }
}

impl<'a> SegmentIterator<'a> {
    fn run_inference_batch(&mut self) -> Result<Option<Vec<(Option<usize>, usize)>>> {
        let start = self.current_window_start;
        let end =
            (start + self.segmenter.batch_size * self.window_size).min(self.padded_samples.len());

        if start >= end {
            return Ok(None);
        }

        let batch_slice = &self.padded_samples[start..end];
        let windows_count = batch_slice.len() / self.window_size;

        let f32_data: Vec<f32> = batch_slice.iter().map(|&x| x as f32).collect();

        let data = TensorData::new(f32_data, [windows_count, 1, self.window_size]);
        let input = Tensor::<BurnBackend, 3>::from_data(data, &self.segmenter.device);
        let output = self.segmenter.model.forward(input);

        let [_, frames, classes] = output.dims();
        let values = output.into_data().into_vec::<f32>().unwrap();

        let results = values
            .chunks(classes)
            .enumerate()
            .map(|(i, scores)| {
                let batch_idx = i / frames;
                let frame_in_win = i % frames;
                let abs_sample = start
                    + (batch_idx * self.window_size)
                    + FRAME_START
                    + (frame_in_win * FRAME_SIZE);

                let (max_idx, &max_score) = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap();

                let class = if max_score > self.segmenter.threshold {
                    Some(max_idx)
                } else {
                    None
                };
                (class, abs_sample)
            })
            .collect();

        self.current_window_start += windows_count * self.window_size;
        Ok(Some(results))
    }
}

impl<'a> SegmentIterator<'a> {
    fn push_segment_from_state(&mut self) {
        let start = self.state.start_sample;
        let end = self.state.last_valid_sample;

        if end > start + 1000 {
            self.push_segment(start, end);
        }
    }

    fn flush_state(&mut self) {
        if self.state.active_class.is_some() {
            self.push_segment_from_state();
            self.state.active_class = None;
        }
    }

    fn push_segment(&mut self, start_sample: usize, end_sample: usize) {
        let max_len = self.padded_samples.len();
        let safe_start = start_sample.min(max_len);
        let safe_end = end_sample.min(max_len);

        // let min_samples = (self.sample_rate as f32 * 0.25) as usize;
        // if safe_end > safe_start + min_samples {
        if safe_end > safe_start {
            let samples = self.padded_samples[safe_start..safe_end].to_vec();
            let start_sec = safe_start as f64 / self.sample_rate as f64;
            let end_sec = safe_end as f64 / self.sample_rate as f64;

            println!("DEBUG: Segment found: {:.2}s - {:.2}s", start_sec, end_sec);

            self.segments_queue.push_back(Segment {
                start: start_sec,
                end: end_sec,
                samples,
            });
        }
    }
}

impl Segmenter {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let device = BurnDevice::default();
        let model_path = model_path
            .as_ref()
            .to_str()
            .context("Model path must be valid UTF-8")?;
        let model = nn::segmentation::Model::from_file(model_path, &device);

        Ok(Self {
            model,
            device,
            batch_size: 2,
            threshold: 0.6,
            max_silence_frames: 30,
            max_segment_frames: 1200,
        })
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_max_silence(mut self, frames: usize) -> Self {
        self.max_silence_frames = frames;
        self
    }

    pub fn with_max_segment_length(mut self, frames: usize) -> Self {
        self.max_segment_frames = frames;
        self
    }

    pub fn iter_segments<'a>(
        &'a self,
        samples: &'a [i16],
        sample_rate: u32,
    ) -> Result<SegmentIterator<'a>> {
        if sample_rate == 0 {
            bail!("sample_rate cannot be zero");
        }
        if samples.is_empty() {
            bail!("samples cannot be empty");
        }

        let window_size = (sample_rate * 10) as usize;
        let padded_samples = pad_to_window(samples, window_size);

        Ok(SegmentIterator {
            segmenter: self,
            padded_samples,
            sample_rate,
            window_size,
            current_window_start: 0,
            segments_queue: VecDeque::new(),
            finished: false,
            state: SegmentationState::new(self.max_silence_frames, self.max_segment_frames),
        })
    }

    pub fn segments(&self, samples: &[i16], sample_rate: u32) -> Result<Vec<Segment>> {
        self.iter_segments(samples, sample_rate)?.collect()
    }

    pub fn segments_f32(&self, samples: &[f32], sample_rate: u32) -> Result<Vec<Segment>> {
        let converted = f32_to_i16(samples);
        self.segments(&converted, sample_rate)
    }
}

pub fn get_segments<P: AsRef<Path>>(
    samples: &[i16],
    sample_rate: u32,
    model_path: P,
) -> Result<Vec<Segment>> {
    let segmenter = Segmenter::new(model_path)?;
    segmenter.segments(samples, sample_rate)
}

pub fn get_segments_f32<P: AsRef<Path>>(
    samples: &[f32],
    sample_rate: u32,
    model_path: P,
) -> Result<Vec<Segment>> {
    let segmenter = Segmenter::new(model_path)?;
    segmenter.segments_f32(samples, sample_rate)
}

fn pad_to_window(samples: &[i16], window_size: usize) -> Vec<i16> {
    if window_size == 0 {
        return samples.to_vec();
    }
    let remainder = samples.len() % window_size;
    if remainder == 0 {
        return samples.to_vec();
    }
    let pad = window_size - remainder;
    samples
        .iter()
        .copied()
        .chain(std::iter::repeat(0).take(pad))
        .collect()
}

fn find_max_index_and_score(row: &[f32]) -> Result<(usize, f32)> {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(index, &score)| (index, score))
        .context("row should not be empty")
}

fn find_max_index(row: &[f32]) -> Result<usize> {
    find_max_index_and_score(row).map(|(i, _)| i)
}

fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| {
            let clipped = s.clamp(-1.0, 1.0);
            (clipped * i16::MAX as f32) as i16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::pad_to_window;

    #[test]
    fn does_not_add_padding_when_aligned() {
        let samples = vec![1i16; 20];
        let padded = pad_to_window(&samples, 10);
        assert_eq!(padded.len(), 20);
        assert_eq!(padded, samples);
    }

    #[test]
    fn pads_up_to_window_size() {
        let samples = vec![1i16; 15];
        let padded = pad_to_window(&samples, 10);
        assert_eq!(padded.len(), 20);
        assert_eq!(&padded[..15], samples.as_slice());
        assert!(padded[15..].iter().all(|&x| x == 0));
    }
}
