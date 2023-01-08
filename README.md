# Gene set calculations in rust

Utilities to perform simple gene set analysis on single cell data. The functions
are written in rust and can be used from python via maturin/pyo3.

## How to build

1. Run rust unit tests
```
> cargo t
```

2. Create a virtual env with required python dependencies
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Build gene_set_calc python library
```
> maturin develop -r
```
The `-r` builds in release mode.

3. Run python unit tests
```
> pytest
```

