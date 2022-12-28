use anyhow::Error;
use flate2::read::GzDecoder;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use sprs::TriMat;
use std::fs::File;
use std::io::BufReader;
use std::ops::{DivAssign, SubAssign};
use std::path::Path;

#[pymodule]
fn gene_set_calc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn run_calc_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        num_genes: usize,
        num_cells: usize,
        top_perc: f64,
        num_fake_gene_sets: usize,
        gsoi: PyReadonlyArray1<usize>,
    ) -> &'py PyArray1<f32> {
        let all_genes = (0..num_genes).into_iter().collect::<Vec<usize>>();
        let result = run_gene_set_calc(
            data.as_slice().unwrap(),
            indices.as_slice().unwrap(),
            indptr.as_slice().unwrap(),
            num_cells,
            top_perc,
            num_fake_gene_sets,
            all_genes.as_slice(),
            gsoi.as_slice().unwrap(),
        );
        result.into_pyarray(py)
    }

    Ok(())
}
/// mat: num_genes x num_cells, stored in CSR
pub fn compute_mean_cols(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    gsoi: &[usize],
    mean: &mut Array1<f32>,
) {
    mean.fill(0.);
    for gi in gsoi.iter() {
        let low = indptr[*gi];
        let high = indptr[*gi + 1];
        for i in low..high {
            let ci = indices[i];
            let val = data[i];
            mean[ci] += val;
        }
    }

    mean.div_assign(gsoi.len() as f32);
}

pub fn load_mtx_coo(path: &Path) -> Result<TriMat<f32>, Error> {
    // Read in barcodes
    // let bc_path = path.join("barcodes.tsv.gz");
    // let barcodes: Vec<Vec<u8>> = BufReader::new(GzDecoder::new(File::open(&bc_path)?))
    //     .split(b'\n')
    //     .map(|x| x.unwrap())
    //     .collect();

    // Read in features
    // let ft_path = path.join("features.tsv.gz");
    // let features: Vec<Vec<u8>> = BufReader::new(GzDecoder::new(File::open(&ft_path)?))
    //     .split(b'\n')
    //     .map(|x| x.unwrap())
    //     .collect();

    // Read in matrix as csc sparse matrix
    let mat_path = path.join("matrix.mtx.gz");
    let mut mat_reader = BufReader::new(GzDecoder::new(File::open(&mat_path)?));
    let mat: TriMat<f32> = sprs::io::read_matrix_market_from_bufread(&mut mat_reader).unwrap();
    Ok(mat)
}

pub fn make_fake_gene_set(rng: &mut ThreadRng, all_genes: &[usize], gsoi: &[usize]) -> Vec<usize> {
    all_genes
        .choose_multiple(rng, gsoi.len())
        .cloned()
        .collect()
}

pub fn run_gene_set_calc(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    num_cells: usize,
    top_perc: f64,
    num_fake_gene_sets: usize,
    all_genes: &[usize],
    gsoi: &[usize],
) -> Array1<f32> {
    let max_index = (num_fake_gene_sets as f64 * (1.0 - top_perc / 100.0)).floor() as usize;
    assert!(max_index >= 0);

    let mut rng = rand::thread_rng();

    let mut x_i = Array1::zeros(num_cells);
    compute_mean_cols(&data, &indices, &indptr, &gsoi, &mut x_i);

    let mut top_n_mat = Array2::from_elem((num_cells, max_index + 2), -f32::INFINITY);

    let mut fake_counts = Array1::zeros(num_cells);
    for _ in 0..num_fake_gene_sets {
        let fake_gene_set = make_fake_gene_set(&mut rng, all_genes, &gsoi);
        compute_mean_cols(&data, &indices, &indptr, &fake_gene_set, &mut fake_counts);

        top_n_mat.column_mut(0).assign(&fake_counts);
        for i in 0..num_cells {
            top_n_mat
                .row_mut(i)
                .as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    x_i.sub_assign(&top_n_mat.column(1));

    x_i
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::distributions::uniform::UniformFloat;
    use rand::distributions::uniform::UniformSampler;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn test_load() -> Result<(), Error> {
        let mut rng = rand::thread_rng();

        let num_genes = 10_000;
        let num_cells = 100_000;
        let nnz = 1_000_000;

        let data = (0..nnz)
            .into_iter()
            .map(|_| UniformFloat::<f32>::new(0.1, 10.0).sample(&mut rng))
            .collect::<Vec<f32>>();

        let cell_dist = Uniform::from(0..num_cells);
        let indices = (0..nnz)
            .into_iter()
            .map(|_| cell_dist.sample(&mut rng))
            .collect::<Vec<usize>>();

        let mut indptr = (0..(num_genes + 1))
            .into_iter()
            .map(|_| Uniform::from(0..nnz).sample(&mut rng))
            .collect::<Vec<usize>>();
        indptr.sort();

        let all_genes = (0..num_genes).into_iter().collect::<Vec<usize>>();

        // Create a random gene set
        let gene_set_size = 100;
        let gsoi: Vec<usize> = all_genes
            .as_slice()
            .choose_multiple(&mut rng, gene_set_size)
            .cloned()
            .collect();

        let top_perc = 95.0;

        let _result = run_gene_set_calc(
            &data, &indices, &indptr, num_cells, top_perc, 100, &all_genes, &gsoi,
        );

        Ok(())
    }
}
