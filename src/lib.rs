mod nn;

mod embedding;
mod identify;
mod segment;
mod wav;
mod plda;

pub use embedding::Embedding;
pub use embedding::EmbeddingExtractor;
pub use identify::{EmbeddingManager, UpdateStrategy};
pub use plda::PldaModule;
pub use segment::{Segment, Segmenter, get_segments, get_segments_f32};
pub use wav::{read_wav, read_wav_optimized};
