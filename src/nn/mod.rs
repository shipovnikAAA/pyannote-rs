pub mod segmentation;
pub mod speaker_identification;

#[cfg(feature = "wgpu")]
pub type BurnBackend = burn::backend::wgpu::Wgpu;
#[cfg(feature = "wgpu")]
pub type BurnDevice = burn::backend::wgpu::WgpuDevice;

#[cfg(feature = "cuda")]
pub type BurnBackend = burn::backend::cuda::Cuda<f32>;
#[cfg(feature = "cuda")]
pub type BurnDevice = burn::backend::cuda::CudaDevice;

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
pub type BurnBackend = burn::backend::ndarray::NdArray<f32>;
#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
pub type BurnDevice = burn::backend::ndarray::NdArrayDevice;