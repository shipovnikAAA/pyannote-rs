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

#[derive(Debug)]
pub struct Segmenter {
    model: nn::segmentation::Model<BurnBackend>,
    device: BurnDevice,
}

pub struct SegmentIterator<'a> {
    segmenter: &'a Segmenter,
    padded_samples: Vec<i16>,
    sample_rate: u32,
    window_size: usize,
    current_window_start: usize,
    segments_queue: VecDeque<Segment>,
    finished: bool,
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
            if self.current_window_start >= self.padded_samples.len() {
                self.finished = true;
                return None;
            }

            let window_end =
                (self.current_window_start + self.window_size).min(self.padded_samples.len());
            let window = &self.padded_samples[self.current_window_start..window_end];

            if window.is_empty() {
                self.finished = true;
                return None;
            }

            let window_f32: Vec<f32> = window.iter().map(|&x| x as f32).collect();
            let data = TensorData::new(window_f32, [1, 1, window.len()]);
            let input = Tensor::<BurnBackend, 3>::from_data(data, &self.segmenter.device);

            let output = self.segmenter.model.forward(input);
            let output_data = output.into_data();

            let shape = output_data.shape.clone();
            if shape.len() != 3 {
                return Some(Err(anyhow!(
                    "Unexpected segmentation output shape: {:?}",
                    shape
                )));
            }

            let frames = shape[1];
            let classes = shape[2];

            let values = match output_data.into_vec::<f32>() {
                Ok(v) => v,
                Err(e) => return Some(Err(anyhow!("Failed to read model output: {}", e))),
            };

            // --- parametrs ---
            let frame_size = 270;
            let frame_start = 721;
            let threshold = 0.5;

            let mut offset_in_window = frame_start;
            let mut active_speaker_class: Option<usize> = None;
            let mut start_offset_in_window = 0;

            let mut debug_frames_active = 0;

            for frame_index in 0..frames {
                let start_idx = frame_index * classes;
                let class_scores = &values[start_idx..start_idx + classes];

                let (max_index, max_score) = match find_max_index_and_score(class_scores) {
                    Ok(res) => res,
                    Err(e) => return Some(Err(e)),
                };

                let current_speaker = if max_score > threshold {
                    Some(max_index)
                } else {
                    None
                };

                if current_speaker.is_some() {
                    debug_frames_active += 1;
                }

                match active_speaker_class {
                    Some(prev_class) => {
                        match current_speaker {
                            Some(curr_class) => {
                                if curr_class != prev_class {
                                    let abs_start =
                                        self.current_window_start + start_offset_in_window;
                                    let abs_end = self.current_window_start + offset_in_window;
                                    self.push_segment(abs_start, abs_end);

                                    active_speaker_class = Some(curr_class);
                                    start_offset_in_window = offset_in_window;
                                }
                            }
                            None => {
                                let abs_start = self.current_window_start + start_offset_in_window;
                                let abs_end = self.current_window_start + offset_in_window;
                                self.push_segment(abs_start, abs_end);

                                active_speaker_class = None;
                            }
                        }
                    }
                    None => {
                        if let Some(curr_class) = current_speaker {
                            active_speaker_class = Some(curr_class);
                            start_offset_in_window = offset_in_window;
                        }
                    }
                }

                offset_in_window += frame_size;
            }

            println!("Window processed: start={}, active_frames={}/{}", self.current_window_start, debug_frames_active, frames);

            if let Some(_) = active_speaker_class {
                let abs_start = self.current_window_start + start_offset_in_window;
                let abs_end = self.current_window_start + offset_in_window;
                self.push_segment(abs_start, abs_end);
            }

            self.current_window_start += self.window_size;
        }
    }
}

impl<'a> SegmentIterator<'a> {
    fn push_segment(&mut self, start_sample: usize, end_sample: usize) {
        let safe_start = start_sample.min(self.padded_samples.len());
        let safe_end = end_sample.min(self.padded_samples.len());

        if safe_end > safe_start + 1000 {
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

        Ok(Self { model, device })
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
