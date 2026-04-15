# ScGFT_Evaluator — Full Documentation (Python)

Evaluation library for comparing real single-cell data against synthetic data generated via spectral perturbation (scGFT). It includes 9 statistical metrics covering variance structure, co-expression networks, differential expression, global distribution, and sparsity.

> [!NOTE]
> All metrics that receive matrices expect **Genes × Cells** format (rows = genes, columns = cells). `AnnData` objects store **Cells × Genes**, so the code transposes internally.

---

## Dependencies

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import adjusted_rand_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
```

Additionally, input objects use `anndata.AnnData` (not a direct import, but that structure is expected).

---

## Individual Metrics

### 1. [variance(mat_real, mat_sint)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#12-19) → `float`

Measures whether gene-level variance is preserved between real and synthetic data.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real expression matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic expression matrix |

**Method**: Computes the sample variance (`ddof=1`) for each gene, then calculates **Kendall's Tau** between both variance vectors.

**Interpretation**: τ ∈ [-1, 1]. Values close to 1 indicate that the gene variability ranking is preserved.

---

### 2. [networks(mat_real, mat_sint, k_modules=10)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#20-39) → `float`

Evaluates whether the gene co-expression structure is preserved.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real expression matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic expression matrix |
| `k_modules` | `int` | Number of modules (clusters) to form (default: 10) |

**Method**:
1. Computes the Pearson correlation matrix between genes
2. Converts to distance: `d = 1 - cor`
3. Applies hierarchical clustering (Ward) and cuts into `k_modules` clusters
4. Compares assignments using **Adjusted Rand Index (ARI)**

**Interpretation**: ARI ∈ [-1, 1]. ARI = 1 → identical clusters. ARI ≈ 0 → random assignments.

---

### 3. [pairs(mat_real, mat_sint, n_pairs=1000, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#40-71) → `float`

Overlap of the most correlated gene pairs.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real expression matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic expression matrix |
| `n_pairs` | `int` | How many top pairs to extract (default: 1000) |
| `gene_names` | `list[str]` or `None` | Gene names; if `None`, uses indices |

**Method**: For each matrix, extracts the `n_pairs` with the highest |correlation| and computes the **Simpson Index** (intersection / min(|A|, |B|)).

**Interpretation**: Simpson ∈ [0, 1]. High values → the same gene pairs are highly correlated in both matrices.

---

### 4. [_limma_approx(mat, covars_df)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#72-89) → `np.ndarray` (indices)

**Private helper method**. Approximation to limma (R) using gene-wise OLS.

| Parameter | Type | Description |
|---|---|---|
| `mat` | `np.ndarray` (G×C) | Expression matrix |
| `covars_df` | `pd.DataFrame` | Design DataFrame. First column: `Group` (0/1). Additional columns: covariates |

**Method**: Fits an OLS model per gene, extracts the p-value for the group coefficient, applies **FDR (Benjamini-Hochberg)** correction, and returns the indices of genes with `p_adj < 0.05`.

---

### 5. [limma(adata_real, adata_sint, genes_top, col_group, group_a, group_b, covariates=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#90-131) → `dict`

Differential expression between two groups.

| Parameter | Type | Description |
|---|---|---|
| `adata_real` | `AnnData` | Real data |
| `adata_sint` | `AnnData` | Synthetic data |
| `genes_top` | `list[str]` | List of genes to analyze |
| `col_group` | `str` | Column in `.obs` with group labels |
| `group_a` | `str` | Reference group label |
| `group_b` | `str` | Target group label |
| `covariates` | `list[str]` or `None` | Additional `.obs` columns for the model |

**Returns**:
```python
{'simpson': float, 'n_real': int, 'n_sint': int}
```
- `simpson`: Simpson index between the DEG sets from real and synthetic data
- `n_real` / `n_sint`: Number of DEGs found in each

---

### 6. [mmd(pca_real, pca_sint, sigma=None, n_perms=100, sample_size=500)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#132-178) → `dict`

Maximum Mean Discrepancy with RBF kernel and permutation test.

| Parameter | Type | Description |
|---|---|---|
| `pca_real` | `np.ndarray` (C×PCs) | PCA embedding of real data |
| `pca_sint` | `np.ndarray` (C×PCs) | PCA embedding of synthetic data |
| `sigma` | `float` or `None` | Kernel bandwidth; if `None`, uses the median of distances |
| `n_perms` | `int` | Number of permutations for the test (default: 100) |
| `sample_size` | `int` | Subsampling size (default: 500) |

**Returns**:
```python
{'distance': float, 'mmd2': float, 'sigma': float, 'p_value': float}
```

**Interpretation**: `distance` ≈ 0 and `p_value` > 0.05 → distributions are statistically indistinguishable.

---

### 7. [sparsity(mat_real, mat_sint, threshold=0.0)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#179-188) → `dict`

Compares sparsity (percentage of zeros or near-zero values).

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic matrix |
| `threshold` | `float` | Values ≤ threshold are considered "zero" (default: 0.0) |

**Returns**:
```python
{'pct_zeros_real': float, 'pct_zeros_sint': float, 'absolute_difference': float}
```

---

### 8. [jaccard_networks(mat_real, mat_sint, cor_threshold=0.5, n_random=100, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#189-233) → `float`

Gene-gene network similarity using density-corrected Jaccard index.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic matrix |
| `cor_threshold` | `float` | |correlation| threshold to define an edge (default: 0.5) |
| `n_random` | `int` | Permutations for density correction |
| `gene_names` | `list[str]` or `None` | Gene names |

**Method**: Builds a co-expression graph for each matrix (edge = |cor| ≥ threshold), computes Jaccard, then subtracts the expected Jaccard under random node permutations.

**Interpretation**: Values > 0 indicate more overlap than expected by chance. Can be negative.

---

### 9. [mad(mat_real, mat_sint)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#234-240) → `float`

Mean Absolute Difference cell by cell.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic matrix (**same dimensions**) |

**Requirement**: Matrices must have exactly the same dimensions (paired, gene by gene and cell by cell).

**Interpretation**: The lower, the more similar. Only meaningful when synthetic cells are aligned with real ones (e.g., same order).

---

### 10. [mean_correlation(mat_real, mat_sint, genes_deg=None, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#241-264) → `dict`

Correlation between mean expression profiles per gene.

| Parameter | Type | Description |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Real matrix |
| `mat_sint` | `np.ndarray` (G×C) | Synthetic matrix |
| `genes_deg` | `set[str]` or `None` | If provided, filters to these genes only |
| `gene_names` | `list[str]` or `None` | Names corresponding to the rows |

**Returns**:
```python
{'R2': float, 'Pearson': float, 'Spearman': float, 'N_genes': int}
```

---

## Orchestrator Function: [run_all()](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#265-349)

```python
ScGFT_Evaluator.run_all(
    adata_real, adata_sint, genes_top,
    col_group, group_a, group_b,
    covariates=None, sparsity_threshold=1e-4
)
```

Runs **all** metrics and returns a single-row `pd.DataFrame` with all columns. Per-group metrics use dynamic column names based on the values of `group_a` and `group_b`.

**Prerequisites**:
- `AnnData` objects must have PCA computed in `adata.obsm['X_pca']`
- The `col_group` column must exist in `adata.obs`

---

## Detailed Use Case

### Scenario

You have a single-cell RNA dataset with oligodendrocytes. You have generated synthetic data with scGFT by applying a spectral perturbation. You want to evaluate how similar the synthetic data is to the real data.

### Step 1: Prepare the Data

```python
import scanpy as sc
from scGFT_Evaluator import ScGFT_Evaluator

# ── Load data ──
adata_real = sc.read_h5ad("oligodendrocytes_real.h5ad")
adata_sint = sc.read_h5ad("oligodendrocytes_synthetic.h5ad")

# ── Check the group column ──
print(adata_real.obs["diagnosis"].value_counts())
# Control        1200
# Parkinson's     800
```

### Step 2: Select Genes of Interest

```python
# Highly variable genes or those used in the perturbation
sc.pp.highly_variable_genes(adata_real, n_top_genes=2000)
genes_top = adata_real.var_names[adata_real.var['highly_variable']].tolist()

# Or use a predefined list
# genes_top = ["MOBP", "PLP1", "MBP", "OLIG1", "OLIG2", ...]
```

### Step 3: Compute PCA (if not already done)

```python
if 'X_pca' not in adata_real.obsm:
    sc.pp.pca(adata_real, n_comps=50)
    
if 'X_pca' not in adata_sint.obsm:
    sc.pp.pca(adata_sint, n_comps=50)
```

### Step 4: Run the Full Evaluation

```python
results = ScGFT_Evaluator.run_all(
    adata_real=adata_real,
    adata_sint=adata_sint,
    genes_top=genes_top,
    col_group="diagnosis",             # column with group labels
    group_a="Control",                 # reference group
    group_b="Parkinson's",             # target group
    covariates=["age", "pmi_hours"],   # covariates for the limma model
    sparsity_threshold=1e-4            # sparsity threshold
)

print(results.T)  # transpose for better readability
```

### Step 5: Interpret the Results

```
Tau_Control              0.9234   ← Variance preserved in Control ✓
ARI_Control              0.6512   ← Co-expression partially preserved
Jaccard_Control          0.1823   ← Gene-gene network with moderate overlap
Pairs_Control            0.7145   ← Good correlated pairs
Tau_Parkinsons           0.8901   ← Variance preserved in Parkinson's ✓
ARI_Parkinsons           0.5634   ← Co-expression slightly worse
Jaccard_Parkinsons       0.1456   ← Lower network overlap
Pairs_Parkinsons         0.6823   ← Acceptable pairs
MAD                      0.0312   ← Low absolute difference ✓
R2_means                 0.9876   ← Very similar mean profile ✓
Pearson_means            0.9938   
Sparsity_diff            1.2340   ← Sparsity difference ~1.2% ✓
Limma_Simpson            0.4500   ← 45% of DEGs shared
Limma_n_real             234      
Limma_n_sint             189      
MMD_distance             0.0045   ← Distributions very close ✓
MMD_pvalue               0.3200   ← Not significant (p>0.05) ✓
```

### Step 6 (Optional): Run Individual Metrics

```python
# Variance only for a subgroup
mask_ctrl = (adata_real.obs["diagnosis"] == "Control").values
mat_ctrl_real = adata_real.X[mask_ctrl][:, idx_genes].T
mat_ctrl_sint = adata_sint.X[mask_ctrl_sint][:, idx_genes].T

if hasattr(mat_ctrl_real, "toarray"):
    mat_ctrl_real = mat_ctrl_real.toarray()

tau = ScGFT_Evaluator.variance(mat_ctrl_real, mat_ctrl_sint)
print(f"Kendall's Tau (Control): {tau:.4f}")
```

### Step 7 (Optional): With a Completely Different Dataset

```python
# Breast cancer dataset
adata_real_cancer = sc.read_h5ad("breast_cancer_real.h5ad")
adata_sint_cancer = sc.read_h5ad("breast_cancer_synth.h5ad")

results_cancer = ScGFT_Evaluator.run_all(
    adata_real_cancer, adata_sint_cancer, genes_cancer,
    col_group="tumor_stage",
    group_a="Normal",
    group_b="Invasive",
    covariates=["patient_age", "tumor_grade"]
)
# Columns will be: Tau_Normal, ARI_Normal, ..., Tau_Invasive, ARI_Invasive, ...
```

---

## Dependency Troubleshooting

### `ModuleNotFoundError: No module named 'h5py'`

`scanpy` depends on `anndata`, which in turn requires `h5py`. If pip reports it as installed but Python cannot import it, force a reinstall:

```bash
pip install --force-reinstall h5py
```

> [!WARNING]
> Reinstalling `h5py` may pull `numpy` up to version 2.x, which breaks packages compiled with numpy 1.x (`pandas`, `pyarrow`, `torch`, etc.), producing errors such as:
> ```
> AttributeError: _ARRAY_API not found
> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
> ```

**Solution**: pin numpy < 2 after reinstalling h5py:

```bash
pip install "numpy==1.26.4" "h5py>=3.7,<4"
```

### `OSError: Could not find/load shared object file 'llvmlite.dll'`

Occurs when `llvmlite` (a dependency of `numba`, used by `scanpy`) is incompatible with the installed numpy version. Solution:

```bash
pip install "numpy==1.26.4" "numba==0.60.0" "llvmlite==0.43.0" "h5py>=3.7,<4"
```

### Verified Compatible Versions

| Package | Version | Notes |
|---|---|---|
| `numpy` | 1.26.4 | Required < 2.0 by `torch`, `monai`, `synthcity`, etc. |
| `h5py` | 3.16.0 | Compatible with numpy 1.26.4 |
| `numba` | 0.60.0 | Compatible with numpy 1.26.4 |
| `llvmlite` | 0.43.0 | Required by numba 0.60.0 |
| `scanpy` | 1.11.5 | Works with the above versions |

> [!TIP]
> To verify that all dependencies import correctly:
> ```python
> import h5py, numpy, numba, scanpy
> print(h5py.__version__, numpy.__version__, numba.__version__)
> ```

---

## Metrics Summary and Ranges

| Metric | Range | Ideal | What it evaluates |
|---|---|---|---|
| Tau | [-1, 1] | → 1 | Gene variance ranking |
| ARI | [-1, 1] | → 1 | Co-expression modules |
| Pairs (Simpson) | [0, 1] | → 1 | Top correlated pairs |
| Corrected Jaccard | (-∞, 1] | > 0 | Gene-gene network (edges) |
| MAD | [0, ∞) | → 0 | Mean absolute difference |
| R² means | [0, 1] | → 1 | Mean expression profile |
| Sparsity diff | [0, 100] | → 0 | Difference in % zeros |
| Limma Simpson | [0, 1] | → 1 | Shared DEGs |
| MMD | [0, ∞) | → 0 | Distributional distance |
| MMD p-value | [0, 1] | > 0.05 | Distribution equality test |
