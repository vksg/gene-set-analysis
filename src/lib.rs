use ndarray::{Axis, Zip};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use std::ops::{AddAssign, DivAssign, SubAssign};
use std::time::SystemTime;

/// Rust-extension for basic gene set calculations in single-cell data
#[pymodule]
fn gene_set_calc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// Compute a gene module score per cell
    ///
    /// Given a gene-cell matrix with normalized log-TPM counts, we compute a
    /// gene-module score per cell. The matrix is assumed to be of dimensions
    /// `num_genes` x `num_cells` and stored in CSR format as specified by
    /// `data`, `indices` and `indptr`.
    ///
    /// `gsoi` consists of gene indices in this matrix that comprise the gene
    /// set of interest.
    ///
    /// `num_fake_gene_sets` is the number of background gene sets to generate
    ///
    /// `top_perc` refers to the percentile of the background gene set
    /// distribution to subtract from the expression of genes in the gene
    /// set of interest. If `None` compute the mean of the background gene sets.
    ///
    /// `num_gene_bins` is the number of quantile bins to divide the full set
    /// of genes into for the purpose of drawing background gene sets.
    #[pyfn(m)]
    fn run_multi_calc_py<'py>(
        py: Python<'py>,
        data: PyReadonlyArray1<f32>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        num_genes: usize,
        num_cells: usize,
        top_perc: Option<f64>,
        num_fake_gene_sets: usize,
        num_gene_bins: usize,
        gsois: Vec<Vec<usize>>,
        num_threads: usize,
    ) -> &'py PyArray2<f32> {
        let mat = CountMatrixCsr::new(
            data.as_slice().unwrap(),
            indices.as_slice().unwrap(),
            indptr.as_slice().unwrap(),
            num_genes,
            num_cells,
        );
        let start = SystemTime::now();
        let gene_props = MatrixGeneProperties::new(&mat, num_gene_bins);
        let num_gene_sets = gsois.len();
        let mut result = Array2::<f32>::zeros((num_gene_sets, num_cells));

        println!(
            "INFO: {:.1} s setting up data structures",
            SystemTime::now()
                .duration_since(start)
                .unwrap()
                .as_secs_f32()
        );
        if num_threads > 1 {
            // Multi-threaded version
            let index = Array1::from_iter(0..num_gene_sets)
                .into_shape((num_gene_sets, 1))
                .unwrap();
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
            Zip::from(result.axis_iter_mut(Axis(0)))
                .and(index.axis_iter(Axis(0)))
                .into_par_iter()
                .for_each(|(row, i)| {
                    let gsoi = &gsois[i[0]];
                    top_perc
                        .map_or(
                            run_gene_set_calc_mean(&mat, num_fake_gene_sets, &gsoi, &gene_props),
                            |t| run_gene_set_calc(&mat, t, num_fake_gene_sets, &gsoi, &gene_props),
                        )
                        .assign_to(row);
                });
        } else {
            // Single-threaded version w/o rayon for a simpler stack trace
            for (i, gsoi) in gsois.iter().enumerate() {
                let row = result.row_mut(i);
                top_perc
                    .map_or(
                        run_gene_set_calc_mean(&mat, num_fake_gene_sets, &gsoi, &gene_props),
                        |t| run_gene_set_calc(&mat, t, num_fake_gene_sets, &gsoi, &gene_props),
                    )
                    .assign_to(row);
            }
        }

        result.into_pyarray(py)
    }
    Ok(())
}

/// Sparse CSR matrix
struct CountMatrixCsr<'a> {
    data: &'a [f32],
    indices: &'a [usize],
    indptr: &'a [usize],
    num_genes: usize,
    num_cells: usize,
}

impl CountMatrixCsr<'_> {
    /// Create a new matrix
    fn new<'a>(
        data: &'a [f32],
        indices: &'a [usize],
        indptr: &'a [usize],
        num_genes: usize,
        num_cells: usize,
    ) -> CountMatrixCsr<'a> {
        CountMatrixCsr::<'a> {
            data,
            indices,
            indptr,
            num_genes,
            num_cells,
        }
    }

    /// Compute the per-cell mean expression for genes in a gene set
    ///
    /// Note: the result is passed in as a mutable input to save on memory
    /// allocations
    fn compute_mean_cols(&self, gsoi: &[usize], mean: &mut Array1<f32>) {
        mean.fill(0.);
        for gi in gsoi.iter() {
            let low = self.indptr[*gi];
            let high = self.indptr[*gi + 1];
            for i in low..high {
                let ci = self.indices[i];
                let val = self.data[i];
                mean[ci] += val;
            }
        }
        mean.div_assign(gsoi.len() as f32);
    }

    /// Compute mean expression over all cells per gene
    fn compute_mean_expression_over_cells(&self) -> Array1<f32> {
        // Compute mean expression over all cells for each gene
        let mut mean_gex = Array1::<f32>::zeros(self.num_genes);
        for i in 0..self.num_genes {
            let low = self.indptr[i];
            let high = self.indptr[i + 1];
            for val in self.data[low..high].iter() {
                mean_gex[i] += val
            }
        }
        mean_gex.div_assign(self.num_cells as f32);
        mean_gex
    }
}

/// Create a gene set with the same expression profile as the supplied one
fn make_fake_gene_set(
    rng: &mut ThreadRng,
    gsoi_dist: &Vec<usize>,
    genes_by_bin: &Vec<Vec<usize>>,
) -> Vec<usize> {
    let mut fake_genes = Vec::new();
    for (bin_index, num_pick) in gsoi_dist.iter().enumerate() {
        let bin_genes = genes_by_bin[bin_index].as_slice();
        fake_genes.extend(bin_genes.choose_multiple(rng, *num_pick));
    }
    fake_genes
}

/// Compute a gene module score per cell
///
/// Given a gene-cell matrix with normalized log-TPM counts, we compute a
/// gene-module score per cell. The matrix is assumed to be of dimensions
/// `num_genes` x `num_cells` and stored in CSR format as specified by
/// `data`, `indices` and `indptr`.
///
/// `gsoi` consists of gene indices in this matrix that comprise the gene
/// set of interest.
///
/// `num_fake_gene_sets` is the number of background gene sets to generate
///
/// `top_perc` refers to the percentile of the background gene set
/// distribution to subtract from the expression of genes in the gene
/// set of interest.
fn run_gene_set_calc<'a, 'b>(
    csr_mat: &'a CountMatrixCsr<'a>,
    top_perc: f64,
    num_fake_gene_sets: usize,
    gsoi: &'a [usize],
    gene_props: &'b MatrixGeneProperties,
) -> Array1<f32> {
    assert!(csr_mat.indptr.len() == csr_mat.num_genes + 1);

    let max_index = (num_fake_gene_sets as f64 * (1.0 - top_perc / 100.0)).floor() as usize;

    let mut gsoi_dist = vec![0; gene_props.num_gene_bins];
    for gi in gsoi.iter() {
        let exp = gene_props.mean_gex[*gi];
        let bi = gene_props
            .bin_edges
            .as_slice()
            .partition_point(|&x| x < exp)
            - 1;
        gsoi_dist[bi] += 1;
    }

    let mut rng = rand::thread_rng();

    let mut x_i = Array1::zeros(csr_mat.num_cells);
    csr_mat.compute_mean_cols(&gsoi, &mut x_i);

    let mut top_n_mat = Array2::from_elem((csr_mat.num_cells, max_index + 2), -f32::INFINITY);

    let mut fake_counts = Array1::zeros(csr_mat.num_cells);
    for _ in 0..num_fake_gene_sets {
        let fake_gene_set = make_fake_gene_set(&mut rng, &gsoi_dist, &gene_props.genes_by_bin);
        assert!(fake_gene_set.len() == gsoi.len());
        csr_mat.compute_mean_cols(&fake_gene_set, &mut fake_counts);

        top_n_mat.column_mut(0).assign(&fake_counts);
        for i in 0..csr_mat.num_cells {
            top_n_mat
                .row_mut(i)
                .as_slice_mut()
                .unwrap()
                .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        }
    }

    x_i.sub_assign(&top_n_mat.column(1));

    x_i
}

struct MatrixGeneProperties {
    num_gene_bins: usize,
    mean_gex: Array1<f32>,
    genes_by_bin: Vec<Vec<usize>>,
    bin_edges: Vec<f32>,
}

impl MatrixGeneProperties {
    fn new<'a>(csr_mat: &'a CountMatrixCsr<'a>, num_gene_bins: usize) -> Self {
        let mut all_genes = (0..csr_mat.num_genes).into_iter().collect::<Vec<usize>>();

        let mean_gex = csr_mat.compute_mean_expression_over_cells();

        // sort genes by expression
        all_genes.sort_unstable_by(|&a, &b| mean_gex[a].partial_cmp(&mean_gex[b]).unwrap());
        let mut mean_gex_sorted = Array1::<f32>::zeros(csr_mat.num_genes);
        for (i, gene) in all_genes.iter().enumerate() {
            mean_gex_sorted[i] = mean_gex[*gene];
        }

        // Bin the genes into quantiles
        let step = (csr_mat.num_genes / num_gene_bins as usize).max(1);
        let mut bin_edges = Vec::new();
        let mut genes_by_bin = Vec::new();
        for i in 0..num_gene_bins {
            let low = i * step;
            let high = ((i + 1) * step).min(csr_mat.num_genes);
            genes_by_bin.push(all_genes[low..high].to_vec());
            if i == 0 {
                bin_edges.push(-f32::INFINITY);
                bin_edges.push(mean_gex_sorted[high]);
            } else if i == num_gene_bins - 1 {
                bin_edges.push(f32::INFINITY);
            } else {
                bin_edges.push(mean_gex_sorted[high]);
            }
        }

        MatrixGeneProperties {
            num_gene_bins: num_gene_bins,
            mean_gex: mean_gex,
            genes_by_bin: genes_by_bin,
            bin_edges: bin_edges,
        }
    }
}

fn run_gene_set_calc_mean<'a, 'b>(
    csr_mat: &'a CountMatrixCsr<'a>,
    num_fake_gene_sets: usize,
    gsoi: &'a [usize],
    gene_props: &'b MatrixGeneProperties,
) -> Array1<f32> {
    assert!(csr_mat.indptr.len() == csr_mat.num_genes + 1);

    let mut gsoi_dist = vec![0; gene_props.num_gene_bins];
    for gi in gsoi.iter() {
        let exp = gene_props.mean_gex[*gi];
        let bi = gene_props
            .bin_edges
            .as_slice()
            .partition_point(|&x| x < exp)
            - 1;
        gsoi_dist[bi] += 1;
    }
    let mut rng = rand::thread_rng();

    let mut x_i = Array1::zeros(csr_mat.num_cells);
    csr_mat.compute_mean_cols(&gsoi, &mut x_i);

    let mut back_mean = Array1::<f32>::zeros(csr_mat.num_cells);
    let mut fake_counts = Array1::<f32>::zeros(csr_mat.num_cells);
    for _ in 0..num_fake_gene_sets {
        let fake_gene_set = make_fake_gene_set(&mut rng, &gsoi_dist, &gene_props.genes_by_bin);
        assert!(fake_gene_set.len() == gsoi.len());
        csr_mat.compute_mean_cols(&fake_gene_set, &mut fake_counts);

        back_mean.add_assign(&fake_counts);
    }
    back_mean.div_assign(num_fake_gene_sets as f32);
    x_i.sub_assign(&back_mean);

    x_i
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand::distributions::uniform::UniformFloat;
    use rand::distributions::uniform::UniformSampler;
    use rand::distributions::{Distribution, Uniform};

    #[test]
    fn test_load() {
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
        let csr_mat = CountMatrixCsr::new(&data, &indices, &indptr, num_genes, num_cells);
        let gene_props = MatrixGeneProperties::new(&csr_mat, 10);
        let _result = run_gene_set_calc(&csr_mat, top_perc, 100, &gsoi, &gene_props);
    }
}
