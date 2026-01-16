use crate::embedding::Embedding;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EmbeddingManager {
    max_speakers: usize,
    speakers: HashMap<usize, Embedding>,
    next_speaker_id: usize,
}

impl EmbeddingManager {
    pub fn new(max_speakers: usize) -> Self {
        Self {
            max_speakers,
            speakers: HashMap::new(),
            next_speaker_id: 1,
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

    /// Try to match a speaker; if none is found above `threshold`, register a new speaker
    /// as long as capacity allows.
    pub fn upsert(&mut self, embedding: &Embedding, threshold: f32) -> Option<usize> {
        let (best_speaker_id, _best_similarity) = self.speakers.iter().fold(
            (None, threshold),
            |(best_id, best_similarity), (&speaker_id, speaker_embedding)| {
                let similarity =
                    Self::cosine_similarity(embedding.as_slice(), speaker_embedding.as_slice());
                if similarity > best_similarity {
                    (Some(speaker_id), similarity)
                } else {
                    (best_id, best_similarity)
                }
            },
        );

        match best_speaker_id {
            Some(id) => Some(id),
            None => self.add_speaker(embedding),
        }
    }

    pub fn best_match(&self, embedding: &Embedding) -> Option<usize> {
        self.speakers
            .iter()
            .map(|(&speaker_id, speaker_embedding)| {
                (
                    speaker_id,
                    Self::cosine_similarity(embedding.as_slice(), speaker_embedding.as_slice()),
                )
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .map(|(speaker_id, _)| speaker_id)
    }

    fn add_speaker(&mut self, embedding: &Embedding) -> Option<usize> {
        if self.is_full() {
            return None;
        }
        let speaker_id = self.next_speaker_id;
        self.speakers.insert(speaker_id, embedding.clone());
        self.next_speaker_id += 1;
        Some(speaker_id)
    }

    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    pub fn is_full(&self) -> bool {
        self.speakers.len() >= self.max_speakers
    }

    pub fn speakers(&self) -> &HashMap<usize, Embedding> {
        &self.speakers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut manager = EmbeddingManager::new(1);
        let first = manager.upsert(&Embedding::new(vec![1.0, 0.0]), 0.5);
        assert_eq!(first, Some(1));

        // Second unique embedding should be rejected because max_speakers is 1.
        let second = manager.upsert(&Embedding::new(vec![0.0, 1.0]), 0.5);
        assert!(second.is_none());
    }
}
