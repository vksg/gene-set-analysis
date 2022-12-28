# Gene set calculations in rust

Utilities to perform simple gene set analysis on single cell data. The functions
are written in rust and can be used from python via maturin/pyo3.

## How to build

1. Run rust unit tests
```
> cargo t
```

2. Build python library
```
> maturin develop
```

3. Run python unit tests
```
> cd python/gene_set_calc
> python -m unittest test_matrix.py
```

