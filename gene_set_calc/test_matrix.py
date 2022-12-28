import unittest
import numpy as np
import scipy.sparse as sp_sparse
import gene_set_calc


def make_random_coo_matrix(
    rng: np.random.Generator, nnz: int, num_rows: int, num_cols: int
):
    data = rng.random(nnz)
    rows = rng.integers(low=0, high=num_rows, size=nnz)
    cols = rng.integers(low=0, high=num_cols, size=nnz)
    coo = sp_sparse.coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
    return coo


def compute_mean_expression(csr_mat, gsoi):
    return np.squeeze(np.array(csr_mat[gsoi, :].mean(axis=0)))


def make_fake_gene_set(
    rng,
    num_genes_total,
    gsoi,
    mean_expression=None,
):
    if not mean_expression:
        return rng.choice(num_genes_total, size=len(gsoi), replace=False)
    raise NotImplementedError("not done")


def do_gene_set_calc(csr_mat, gsoi, num_fake_gene_sets, top_perc: float):

    num_genes, num_cells = csr_mat.shape
    max_index = int(num_fake_gene_sets * (1.0 - top_perc / 100))
    assert max_index >= 0

    x_i = compute_mean_expression(csr_mat, gsoi)

    top_n_mat = np.full((num_cells, max_index + 2), fill_value=-np.inf)

    rng = np.random.default_rng(seed=0)

    for i in range(num_fake_gene_sets):
        fake_gene_set = make_fake_gene_set(rng, num_genes, gsoi)

        fake_counts = compute_mean_expression(csr_mat, fake_gene_set)

        top_n_mat[:, 0] = fake_counts

        top_n_mat.sort(axis=1)
    result = top_n_mat[:, 1]

    answer = x_i - result

    return answer


class TestFunc(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=0)
        num_cells = 1000
        num_genes = 2000
        coo = make_random_coo_matrix(
            rng, nnz=10000, num_rows=num_genes, num_cols=num_cells
        )
        self.csr = coo.tocsr()

        gene_set_size = 50
        self.num_fake_gene_sets = 20
        self.top_perc = 50
        self.gsoi = rng.choice(a=num_genes, size=gene_set_size, replace=False)

    def test_gene_set_calc(self):
        v = do_gene_set_calc(
            self.csr, self.gsoi, self.num_fake_gene_sets, self.top_perc
        )
        self.assertEqual(v, 2)

    def test_axpy(self):
        rng = np.random.default_rng(0)
        a = 0.2
        x = rng.random(20)
        y = rng.random(20)
        pyres = a * x + y
        rustres = gene_set_calc.axpy_py(a, x, y)
        self.assertIsNone(np.testing.assert_allclose(rustres, pyres, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
