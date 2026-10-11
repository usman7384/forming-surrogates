# Environment (recorded at Wave 0)

- OS: Windows 11, 16 logical cores, CPU only.
- Python 3.11.9, virtualenv at `.venv/` (git-ignored). Use `.venv/Scripts/python`.
- Install: `python -m pip install -r requirements.txt -r requirements-ml.txt`
- Data root: `FS_DATA_ROOT=~/forming-data` (outside the repo).

| package | version |
| --- | --- |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-fem | 12.0.2 |
| pandas | 3.0.6 |
| pyarrow | 25.0.1 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| torch | 2.14.1 |
| torch-geometric | 2.8.0.post1 |
| gpytorch | 1.15.2 |
| botorch | 0.18.1 |
| asyncua | 2.1.0 |
| onnx / onnxruntime | 1.23.2 / 1.31.0 |
| streamlit | 1.65.0 |
| pytest | 9.1.1 |

Import check: `import skfem, torch, torch_geometric, gpytorch, botorch, asyncua, streamlit` succeeds.
