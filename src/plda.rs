use anyhow::{Context, Result, bail}; // Не забудь bail
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use std::fs::File;

#[derive(Debug, Clone)]
pub struct PldaModule {
    pub transform_mean: Array1<f32>,
    pub transform_mat: Array2<f32>,
    pub plda_mean: Array1<f32>,
    pub plda_mat: Array2<f32>,
    pub psi: Array1<f32>,
}

impl PldaModule {
    fn load_array1(reader: &mut NpzReader<File>, name: &str) -> Result<Array1<f32>> {
        if let Ok(arr) = reader.by_name(name) {
            return Ok(arr);
        }
        if let Ok(arr) = reader.by_name(&format!("{}.npy", name)) {
            return Ok(arr);
        }
        if let Ok(arr) = reader.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(name) {
            return Ok(arr.mapv(|x| x as f32));
        }
        if let Ok(arr) =
            reader.by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>(&format!("{}.npy", name))
        {
            return Ok(arr.mapv(|x| x as f32));
        }

        bail!("Could not read 1D array '{}' as f32 or f64", name);
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
        let transform_mat = Self::load_array2(&mut trans_npz, "lda")?;

        let mut plda_npz =
            NpzReader::new(File::open(plda_path).context("Failed to open PLDA file")?)?;

        let plda_mean = Self::load_array1(&mut plda_npz, "mu")?;
        let plda_mat = Self::load_array2(&mut plda_npz, "tr")?;
        let psi = Self::load_array1(&mut plda_npz, "psi")?;

        Ok(Self {
            transform_mean,
            transform_mat,
            plda_mean,
            plda_mat,
            psi,
        })
    }

    fn preprocess(&self, embedding: &[f32]) -> Array1<f32> {
        let x = Array1::from_vec(embedding.to_vec());
        let x = (x - &self.transform_mean).dot(&self.transform_mat);
        x - &self.plda_mean
    }

    pub fn score(&self, emb_a: &[f32], emb_b: &[f32]) -> f32 {
        let u = self.preprocess(emb_a).dot(&self.plda_mat);
        let v = self.preprocess(emb_b).dot(&self.plda_mat);

        let mut score = 0.0;
        for i in 0..u.len() {
            let p = self.psi[i];
            let term1 = (p * (u[i] + v[i]).powi(2)) / (1.0 + 2.0 * p);
            let term2 = (p * (u[i].powi(2) + v[i].powi(2))) / (1.0 + p);
            score += term1 - term2;
        }
        score
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
            println!("--- Keys in {:?} ---", plda_path);
            for name in npz.names().unwrap() {
                println!("key: '{}'", name);
            }
        }

        if Path::new(trans_path).exists() {
            let file = std::fs::File::open(trans_path).unwrap();
            let mut npz = NpzReader::new(file).unwrap();
            println!("--- Keys in {:?} ---", trans_path);
            for name in npz.names().unwrap() {
                println!("key: '{}'", name);
            }
        }

        let result = PldaModule::load(plda_path, trans_path);
        if let Err(e) = result {
            panic!("error while loading PLDA: {:?}", e);
        }
    }
}
