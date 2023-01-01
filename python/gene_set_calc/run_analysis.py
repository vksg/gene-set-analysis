#!/usr/bin/env python3

import sys
import argparse
import json
import pickle
import csv
import os
import time
from collections import defaultdict
import scipy.sparse as sp_sparse
import numpy as np
import scanpy
import gene_set_calc


def main(argv):
    parser = argparse.ArgumentParser(description="Process gene sets")
    parser.add_argument(
        "--csr_matrix_pickle", default=None, help="CSR count matrix pickle"
    )
    parser.add_argument("--csr_matrix_h5ad", default=None, help="h5ad matrix")
    parser.add_argument("gene_sets_json", help="gene sets in json")
    parser.add_argument("gene_info_csv", help="gene info csv")
    parser.add_argument(
        "--top_perc", default=None, help="percentile of background expression"
    )
    parser.add_argument("--threads", default=1, type=int, help="number of threads")
    parser.add_argument(
        "--num_background_sets",
        default=100,
        help="number of background gene sets to generate",
    )
    parser.add_argument("--head", default=0, type=int, help="only do top N gene sets")
    args = parser.parse_args(argv[1:])

    if args.csr_matrix_pickle and args.csr_matrix_h5ad:
        print("Must supply pickle XOR h5ad")
        sys.exit(1)
    if not (args.csr_matrix_pickle or args.csr_matrix_h5ad):
        print("Must supply pickle XOR h5ad")
        sys.exit(1)

    start_load = time.time()
    with open(args.gene_sets_json, "r") as fin:
        gene_sets = json.load(fin)

    gene_symbols = defaultdict(list)
    num_rows = 0
    with open(args.gene_info_csv, "r") as fin:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader):
            name = row["gene_symbol"]
            gene_symbols[name].append(i)
            num_rows += 1

    csr_mat = None

    if args.csr_matrix_pickle:
        print("Reading csr matrix from pickle")
        with open(args.csr_matrix_pickle, "rb") as fin:
            csr_mat = pickle.load(fin)
    else:
        ts_h5ad = scanpy.read_h5ad(args.csr_matrix_h5ad)
        csr_mat = ts_h5ad.X.transpose().tocsr()

        # pickle_path = f"{args.csr_matrix_h5ad}.csr_matrix.pickle"
        # if not os.path.exists(pickle_path):
        #     with open(pickle_path, "wb") as out:
        #         pickle.dump(csr_mat, out)
        #         print(f"Wrote CSR matrix pickle to {pickle_path}")
        #         sys.exit(0)
        # else:
        #     print(f"pickle path already exists: {pickle_path}")
        #     sys.exit(1)

    assert csr_mat is not None
    num_genes, num_cells = csr_mat.shape
    assert num_rows == num_genes
    rust_data = csr_mat.data.astype(np.float32)
    rust_indices = csr_mat.indices.astype(np.uint64)
    rust_indptr = csr_mat.indptr.astype(np.uint64)

    print(f"Done loading {time.time()-start_load:.1f}s")

    gsois = []
    for key, val in gene_sets.items():
        gene_names = val["geneSymbols"]
        gsoi = []
        for n in gene_names:
            ixs = gene_symbols.get(n, [])
            gsoi.extend(ixs)
        if not gsoi:
            print(f"gene set {key} is empty, skipping")
            continue
        gsois.append((key, gsoi))

    # sort: largest first
    gsois.sort(key=lambda x: len(x[1]), reverse=True)

    truncate = args.head if args.head else len(gsois)

    first = time.time()

    start = time.time()
    gene_set_calc.run_multi_calc_py(
        rust_data,
        rust_indices,
        rust_indptr,
        num_genes,
        num_cells,
        top_perc=args.top_perc,
        num_fake_gene_sets=args.num_background_sets,
        num_gene_bins=20,
        gsois=[x[1] for x in gsois[:truncate]],
        num_threads=args.threads,
    )
    for key, gsoi in gsois[:truncate]:
        print(f"{key}: {len(gsoi)} genes.")
    print(f"total time ({args.threads} threads)={time.time() - first:.1f}s")


if __name__ == "__main__":
    main(sys.argv)
