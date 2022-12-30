#!/usr/bin/env python

import sys
import argparse
import json
import pickle
import csv
import time
from collections import defaultdict
import scipy.sparse as sp_sparse
import numpy as np

import gene_set_calc


def main(argv):
    parser = argparse.ArgumentParser(description="Process gene sets")
    parser.add_argument("csr_matrix_pickle", help="CSR count matrix pickle")
    parser.add_argument("gene_sets_json", help="gene sets in json")
    parser.add_argument("gene_info_csv", help="gene info csv")
    parser.add_argument(
        "--top_perc", default=95, help="percentile of background expression"
    )
    parser.add_argument(
        "--num_background_sets",
        default=100,
        help="number of background gene sets to generate",
    )
    parser.add_argument("--head", default=0, help="only do top N gene sets")
    args = parser.parse_args(argv[1:])

    with open(args.gene_sets_json, "r") as fin:
        gene_sets = json.load(fin)

    gene_symbols = defaultdict(list)
    num_rows = 0
    with open(args.gene_info_csv, "r") as fin:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            name = row["gene_symbol"]
            gene_symbols[name].append(i)

    with open(args.csr_matrix_pickle, "rb") as fin:
        csr_mat = pickle.load(fin)

    num_genes, num_cells = csr_mat.shape
    assert num_rows == num_genes
    rust_data = csr_mat.data.astype(np.float32)
    rust_indices = csr_mat.indices.astype(np.uint64)
    rust_indptr = csr_mat.indptr.astype(np.uint64)

    print("Done loading")

    for key, val in gene_sets.items():
        gene_names = val["geneSymbols"]
        gsoi = []
        for n in gene_names:
            ixs = gene_symbols.get(n, [])
            gsoi.extend(ixs)
        gsoi = np.array(gsoi, dtype=np.uint64)

        if gsoi.shape[0] == 0:
            print(f"gene set {key} is empty, skipping")
            continue
        start = time.time()
        gene_set_calc.run_calc_py(
            rust_data,
            rust_indices,
            rust_indptr,
            num_genes,
            num_cells,
            top_perc=args.top_perc,
            num_fake_gene_sets=args.num_background_sets,
            gsoi=gsoi,
        )
        print(f"{key}: {len(gsoi)} genes. time={time.time() - start:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
