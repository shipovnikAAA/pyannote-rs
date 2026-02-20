mod nn;

mod embedding;
mod identify;
mod plda;
mod segment;
mod wav;

pub use embedding::Embedding;
pub use embedding::EmbeddingExtractor;
pub use identify::{EmbeddingManager, UpdateStrategy};
pub use plda::PldaModule;
pub use segment::{Segment, Segmenter};
pub use wav::{read_wav, read_wav_optimized};
