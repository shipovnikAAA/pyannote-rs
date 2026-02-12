use crate::embedding::Embedding;
use crate::plda::PldaModule;
use ndarray::Array1;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EmbeddingManager {
    max_speakers: usize,
    speakers: HashMap<usize, Array1<f32>>,
    next_speaker_id: usize,
    plda: Option<PldaModule>,
}

impl EmbeddingManager {
    pub fn new(max_speakers: usize, plda: Option<PldaModule>) -> Self {
        Self {
            max_speakers,
            speakers: HashMap::new(),
            next_speaker_id: 1,
            plda,
        }
    }

    pub fn compute_score(&self, a: &[f32], b: &[f32]) -> f32 {
        if let Some(ref plda) = self.plda {
            let u = plda.transform_vector(a);
            let v = plda.transform_vector(b);
            PldaModule::sigmoid_scale(plda.score_transformed(&u, &v))
        } else {
            Self::cosine_similarity(a, b)
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let (dot, norm_a, norm_b) = a
            .iter()
            .zip(b.iter())
            .fold((0.0, 0.0, 0.0), |(dot, norm_a, norm_b), (x, y)| {
                (dot + x * y, norm_a + x * x, norm_b + y * y)
            });

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a.sqrt() * norm_b.sqrt())
    }

    fn l2_normalize(mut vec: Array1<f32>) -> Array1<f32> {
        let norm = vec.dot(&vec).sqrt();
        if norm > 1e-10 {
            vec /= norm;
        }
        vec
    }

    /// Try to match a speaker; if none is found above `threshold`, register a new speaker
    /// as long as capacity allows.
    pub fn upsert(&mut self, embedding: &Embedding, threshold: f32) -> Option<usize> {
        let input_vec = embedding.as_slice();

        let processed_input = if let Some(ref plda) = self.plda {
            plda.transform_vector(input_vec)
        } else {
            Self::l2_normalize(Array1::from_vec(input_vec.to_vec()))
        };

        let mut best_speaker_id = None;
        let mut best_similarity = threshold;

        for (&id, speaker_vec) in &self.speakers {
            let similarity = if let Some(ref plda) = self.plda {
                PldaModule::sigmoid_scale(plda.score_transformed(&processed_input, speaker_vec))
            } else {
                let dot = processed_input.dot(speaker_vec);
                let norm_a = processed_input.dot(&processed_input).sqrt();
                let norm_b = speaker_vec.dot(speaker_vec).sqrt();
                dot / (norm_a * norm_b + 1e-6)
            };

            println!("Comparing with speaker {}: score {}", id, similarity);

            if similarity > best_similarity {
                best_similarity = similarity;
                best_speaker_id = Some(id);
            }
        }

        match best_speaker_id {
            Some(id) => Some(id),
            None => {
                if self.is_full() {
                    return None;
                }
                let id = self.next_speaker_id;
                self.speakers.insert(id, processed_input);
                self.next_speaker_id += 1;
                Some(id)
            }
        }
    }

    pub fn best_match(&self, embedding: &Embedding) -> Option<usize> {
        if self.speakers.is_empty() {
            return None;
        }

        let input_slice = embedding.as_slice();

        self.speakers
            .iter()
            .map(|(&speaker_id, speaker_vec)| {
                let score = Self::compute_score(self, input_slice, speaker_vec.as_slice().unwrap());
                (speaker_id, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(id, _)| id)
    }

    fn add_speaker_raw(&mut self, processed_vec: Array1<f32>) -> Option<usize> {
        if self.is_full() {
            return None;
        }
        let speaker_id = self.next_speaker_id;
        self.speakers.insert(speaker_id, processed_vec);
        self.next_speaker_id += 1;
        Some(speaker_id)
    }

    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    pub fn is_full(&self) -> bool {
        self.speakers.len() >= self.max_speakers
    }

    pub fn speakers(&self) -> &HashMap<usize, Array1<f32>> {
        &self.speakers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn zero_vectors_do_not_produce_nan() {
        let a = Embedding::new(vec![0.0, 0.0]);
        let b = Embedding::new(vec![0.0, 0.0]);
        assert_eq!(
            EmbeddingManager::cosine_similarity(a.as_slice(), b.as_slice()),
            0.0
        );
    }

    #[test]
    fn upsert_adds_until_cap() {
        let mut manager = EmbeddingManager::new(1, None);
        let first = manager.upsert(&Embedding::new(vec![1.0, 0.0]), 0.5);
        assert_eq!(first, Some(1));

        // Second unique embedding should be rejected because max_speakers is 1.
        let second = manager.upsert(&Embedding::new(vec![0.0, 1.0]), 0.5);
        assert!(second.is_none());
    }
    #[test]
    fn test_plda_scoring_logic() {
        let dim = 4;
        let plda = PldaModule {
            transform_mean: Array1::zeros(dim),
            transform_mean_after: Array1::zeros(dim),
            transform_mat: Array2::eye(dim),
            plda_mean: Array1::zeros(dim),
            plda_mat: Array2::eye(dim),
            psi: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
        };

        let emb_a = Array1::from_vec(vec![1.0, 0.5, 0.0, -0.5]);
        let emb_b = Array1::from_vec(vec![1.0, 0.5, 0.0, -0.5]); // Same
        let emb_c = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5]); // diff

        let score_same = plda.score_transformed(&emb_a, &emb_b);
        let score_diff = plda.score_transformed(&emb_a, &emb_c);

        println!("Score same: {}, Score diff: {}", score_same, score_diff);

        assert!(score_same > score_diff);
    }
    #[test]
    fn test_manager_with_plda_switch() {
        let dim = 4;
        let plda = PldaModule {
            transform_mean: Array1::zeros(dim),
            transform_mean_after: Array1::zeros(dim),
            transform_mat: Array2::eye(dim),
            plda_mean: Array1::zeros(dim),
            plda_mat: Array2::eye(dim),
            psi: Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]),
        };
        let mut manager = EmbeddingManager::new(2, Some(plda));
        let emb = Embedding::new(vec![1.0, 0.0]);

        let score_cos = manager.compute_score(emb.as_slice(), emb.as_slice());
        assert_eq!(score_cos, 1.0);

        let dim = 2;
        let mock_plda = PldaModule {
            transform_mean: Array1::zeros(dim),
            transform_mean_after: Array1::zeros(dim),
            transform_mat: Array2::eye(dim),
            plda_mean: Array1::zeros(dim),
            plda_mat: Array2::eye(dim),
            psi: Array1::ones(dim),
        };
        manager.plda = Some(mock_plda);

        let score_plda = manager.compute_score(emb.as_slice(), emb.as_slice());
        assert!(score_plda != 1.0);
        assert!(score_plda > 0.0);
    }
}
