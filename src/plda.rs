use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array2, aview1};
use ndarray_npy::NpzReader;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct PldaModule {
    pub transform_mean: Array1<f32>,
    pub transform_mean_after: Array1<f32>,
    pub transform_mat: Array2<f32>,
    pub plda_mean: Array1<f32>,
    pub plda_mat: Array2<f32>,
    pub psi: Array1<f32>,
}

impl PldaModule {
    fn load_array1(reader: &mut NpzReader<File>, name: &str) -> Result<Array1<f32>> {
        let arr: Array1<f32> =
            if let Ok(a) = reader.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(name) {
                Array1::from_iter(a)
            } else {
                let a_f64 = reader
                    .by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(name)
                    .map_err(|_| anyhow!("Could not read array '{}' as f32 or f64", name))?;
                Array1::from_iter(a_f64.mapv(|x| x as f32))
            };

        if arr.is_empty() {
            bail!("Array '{}' is empty", name);
        }

        Ok(arr)
    }

    fn load_array2(reader: &mut NpzReader<File>, name: &str) -> Result<Array2<f32>> {
        if let Ok(arr) = reader.by_name(name) {
            return Ok(arr);
        }
        if let Ok(arr) = reader.by_name(&format!("{}.npy", name)) {
            return Ok(arr);
        }

        if let Ok(arr) = reader.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(name) {
            return Ok(arr.mapv(|x| x as f32));
        }
        if let Ok(arr) =
            reader.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix2>(&format!("{}.npy", name))
        {
            return Ok(arr.mapv(|x| x as f32));
        }

        bail!("Could not read 2D matrix '{}' as f32 or f64", name);
    }

    pub fn load(plda_path: &str, transform_path: &str) -> Result<Self> {
        let mut trans_npz =
            NpzReader::new(File::open(transform_path).context("Failed to open transform file")?)?;

        let transform_mean = Self::load_array1(&mut trans_npz, "mean1")?;
        let transform_mean_after = Self::load_array1(&mut trans_npz, "mean2")?;
        let transform_mat = Self::load_array2(&mut trans_npz, "lda")?;

        let mut plda_npz =
            NpzReader::new(File::open(plda_path).context("Failed to open PLDA file")?)?;

        let plda_mean = Self::load_array1(&mut plda_npz, "mu")?;
        let plda_mat = Self::load_array2(&mut plda_npz, "tr")?;
        let psi = Self::load_array1(&mut plda_npz, "psi")?;

        Ok(Self {
            transform_mean,
            transform_mean_after,
            transform_mat,
            plda_mean,
            plda_mat,
            psi,
        })
    }
}

impl PldaModule {
    // LDA, PLDA
    pub fn transform_vector(&self, embedding: &[f32]) -> Array1<f32> {
        let x = aview1(embedding);

        let x_lda = (&x - &self.transform_mean).dot(&self.transform_mat);

        let norm = x_lda.dot(&x_lda).sqrt();
        let x_lda_norm = if norm > 1e-10 { x_lda / norm } else { x_lda };

        let x_post_lda = x_lda_norm - &self.transform_mean_after;

        (&x_post_lda - &self.plda_mean).dot(&self.plda_mat)
    }

    pub fn score_transformed(&self, u: &Array1<f32>, v: &Array1<f32>) -> f32 {
        self.psi
            .iter()
            .zip(u.iter())
            .zip(v.iter())
            .map(|((&p, &ui), &vi)| {
                let sum_uv = ui + vi;
                let term1 = (p * sum_uv * sum_uv) / (1.0 + 2.0 * p);
                let term2 = (p * (ui * ui + vi * vi)) / (1.0 + p);
                term1 - term2
            })
            .sum()
    }

    pub fn sigmoid_scale(raw_score: f32) -> f32 {
        1.0 / (1.0 + (-(0.5 * raw_score)).exp())
    }
}

#[cfg(test)]
mod plda_tests {
    use super::*;
    use ndarray_npy::NpzReader;
    use std::path::Path;

    #[test]
    fn test_plda_loading() {
        let plda_path = "src/nn/plda/plda.npz";
        let trans_path = "src/nn/plda/xvec_transform.npz";

        if Path::new(plda_path).exists() && Path::new(trans_path).exists() {
            let result = PldaModule::load(plda_path, trans_path);
            assert!(
                result.is_ok(),
                "error while loading PLDA: {:?}",
                result.err()
            );

            let plda = result.unwrap();
            assert!(!plda.psi.is_empty());
            println!("PLDA loaded. psi length: {}", plda.psi.len());
        } else {
            println!("PLDA files not found. Skipping test.");
        }
    }
    #[test]
    fn test_plda_loading_aq() {
        let plda_path = "src/nn/plda/plda.npz";
        let trans_path = "src/nn/plda/xvec_transform.npz";

        if Path::new(plda_path).exists() {
            let file = std::fs::File::open(plda_path).unwrap();
            let mut npz = NpzReader::new(file).unwrap();
            println!("--- Contents of {:?} ---", plda_path);

            let names: Vec<String> = npz
                .names()
                .unwrap()
                .into_iter()
                .map(|s| s.to_string())
                .collect();

            for name in names {
                if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
                    println!("key: '{}', shape: {:?}", name, array.shape());
                } else {
                    if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&name)
                    {
                        println!("key: '{}', shape: {:?} (f64)", name, array.shape());
                    } else {
                        println!("key: '{}', could not read shape", name);
                    }
                }
            }
        } else {
            println!("PLDA file not found. Skipping test.");
        }
        if Path::new(trans_path).exists() {
            let file = std::fs::File::open(trans_path).unwrap();
            let mut npz = NpzReader::new(file).unwrap();
            println!("--- Contents of {:?} ---", trans_path);

            let names: Vec<String> = npz
                .names()
                .unwrap()
                .into_iter()
                .map(|s| s.to_string())
                .collect();

            for name in names {
                if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::IxDyn>(&name) {
                    println!("key: '{}', shape: {:?}", name, array.shape());
                } else {
                    if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f64>, ndarray::IxDyn>(&name)
                    {
                        println!("key: '{}', shape: {:?} (f64)", name, array.shape());
                    } else {
                        println!("key: '{}', could not read shape", name);
                    }
                }
            }
        } else {
            println!("xvec_transform file not found. Skipping test.");
        }
    }
}
