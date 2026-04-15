# ScGFT_Evaluator — Documentación completa (Python)

Librería de evaluación para comparar datos single-cell reales frente a datos sintéticos generados mediante perturbación espectral (scGFT). Contiene 9 métricas estadísticas que cubren estructura de varianza, redes de co-expresión, expresión diferencial, distribución global y dispersión.

> [!NOTE]
> Todas las métricas que reciben matrices esperan formato **Genes × Células** (filas = genes, columnas = células). Los objetos `AnnData` almacenan **Células × Genes**, por lo que el código transpone internamente.

---

## Dependencias

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

Además, los objetos de entrada usan `anndata.AnnData` (no es import directo, pero se espera esa estructura).

---

## Métricas individuales

### 1. [varianza(mat_real, mat_sint)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#12-19) → `float`

Mide si la varianza gen a gen se preserva entre real y sintético.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz de expresión real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz de expresión sintética |

**Método**: Calcula la varianza muestral (`ddof=1`) de cada gen, luego computa la **Tau de Kendall** entre ambos vectores de varianzas.

**Interpretación**: τ ∈ [-1, 1]. Valores cercanos a 1 indican que el ranking de variabilidad por gen se conserva.

---

### 2. [redes(mat_real, mat_sint, k_modulos=10)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#20-39) → `float`

Evalúa si la estructura de co-expresión génica se conserva.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz de expresión real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz de expresión sintética |
| `k_modulos` | `int` | Número de módulos (clusters) a formar (default: 10) |

**Método**:
1. Calcula la matriz de correlación de Pearson entre genes
2. Convierte a distancia: `d = 1 - cor`
3. Aplica clustering jerárquico (Ward) y corta en `k_modulos` clusters
4. Compara las asignaciones con **Adjusted Rand Index (ARI)**

**Interpretación**: ARI ∈ [-1, 1]. ARI = 1 → clusters idénticos. ARI ≈ 0 → asignaciones aleatorias.

---

### 3. [pares(mat_real, mat_sint, n_pares=1000, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#40-71) → `float`

Solapamiento de los pares de genes más correlacionados.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz de expresión real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz de expresión sintética |
| `n_pares` | `int` | Cuántos pares top extraer (default: 1000) |
| `gene_names` | `list[str]` o `None` | Nombres de genes; si `None`, usa índices |

**Método**: Para cada matriz, extrae los `n_pares` con mayor |correlación| y calcula el **Índice de Simpson** (intersección / min(|A|, |B|)).

**Interpretación**: Simpson ∈ [0, 1]. Valores altos → los mismos pares de genes están altamente correlacionados en ambas matrices.

---

### 4. [_limma_approx(mat, covars_df)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#72-89) → `np.ndarray` (índices)

**Método auxiliar privado**. Aproximación a limma (R) usando OLS gen a gen.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat` | `np.ndarray` (G×C) | Matriz de expresión |
| `covars_df` | `pd.DataFrame` | DataFrame de diseño. Primera columna: `Grupo` (0/1). Columnas adicionales: covariables |

**Método**: Ajusta un modelo OLS por gen, extrae el p-valor del coeficiente del grupo, aplica corrección **FDR (Benjamini-Hochberg)** y devuelve los índices de genes con `p_adj < 0.05`.

---

### 5. [limma(adata_real, adata_sint, genes_top, col_grupo, grupo_a, grupo_b, covariables=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#90-131) → `dict`

Expresión diferencial entre dos grupos.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `adata_real` | `AnnData` | Datos reales |
| `adata_sint` | `AnnData` | Datos sintéticos |
| `genes_top` | `list[str]` | Lista de genes a analizar |
| `col_grupo` | `str` | Columna de `.obs` con las etiquetas de grupo |
| `grupo_a` | `str` | Etiqueta del grupo de referencia |
| `grupo_b` | `str` | Etiqueta del grupo de interés |
| `covariables` | `list[str]` o `None` | Columnas adicionales de `.obs` para el modelo |

**Retorna**:
```python
{'simpson': float, 'n_real': int, 'n_sint': int}
```
- `simpson`: Índice de Simpson entre los conjuntos de DEGs de real y sintético
- `n_real` / `n_sint`: Número de DEGs encontrados en cada uno

---

### 6. [mmd(pca_real, pca_sint, sigma=None, n_perms=100, sample_size=500)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#132-178) → `dict`

Maximum Mean Discrepancy con kernel RBF y test de permutación.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `pca_real` | `np.ndarray` (C×PCs) | Embedding PCA de datos reales |
| `pca_sint` | `np.ndarray` (C×PCs) | Embedding PCA de datos sintéticos |
| `sigma` | `float` o `None` | Ancho de banda del kernel; si `None`, usa la mediana de las distancias |
| `n_perms` | `int` | Número de permutaciones para el test (default: 100) |
| `sample_size` | `int` | Tamaño de submuestreo (default: 500) |

**Retorna**:
```python
{'distancia': float, 'mmd2': float, 'sigma': float, 'p_valor': float}
```

**Interpretación**: `distancia` ≈ 0 y `p_valor` > 0.05 → las distribuciones son estadísticamente indistinguibles.

---

### 7. [sparsity(mat_real, mat_sint, umbral=0.0)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#179-188) → `dict`

Compara la dispersión (porcentaje de ceros/valores cercanos a cero).

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz sintética |
| `umbral` | `float` | Valores ≤ umbral se consideran "cero" (default: 0.0) |

**Retorna**:
```python
{'pct_ceros_real': float, 'pct_ceros_sint': float, 'diferencia_absoluta': float}
```

---

### 8. [jaccard_redes(mat_real, mat_sint, umbral_cor=0.5, n_random=100, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#189-233) → `float`

Similitud de las redes gen-gen usando índice de Jaccard corregido por densidad.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz sintética |
| `umbral_cor` | `float` | Umbral de |correlación| para definir una arista (default: 0.5) |
| `n_random` | `int` | Permutaciones para la corrección por densidad |
| `gene_names` | `list[str]` o `None` | Nombres de genes |

**Método**: Construye un grafo de co-expresión para cada matriz (arista = |cor| ≥ umbral), calcula Jaccard, y resta el Jaccard esperado bajo permutaciones aleatorias de los nodos.

**Interpretación**: Valores > 0 indican que el solapamiento es mayor que el esperado por azar. Puede ser negativo.

---

### 9. [mad(mat_real, mat_sint)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#234-240) → `float`

Diferencia Absoluta Media (Mean Absolute Difference) célula a célula.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz sintética (**mismas dimensiones**) |

**Requisito**: Las matrices deben tener exactamente las mismas dimensiones (paired, gen a gen y célula a célula).

**Interpretación**: Cuanto menor, más similares. Solo tiene sentido cuando las células sintéticas están alineadas con las reales (ej: mismo orden).

---

### 10. [correlacion_medias(mat_real, mat_sint, genes_deg=None, gene_names=None)](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#241-264) → `dict`

Correlación entre los perfiles medios de expresión por gen.

| Parámetro | Tipo | Descripción |
|---|---|---|
| `mat_real` | `np.ndarray` (G×C) | Matriz real |
| `mat_sint` | `np.ndarray` (G×C) | Matriz sintética |
| `genes_deg` | `set[str]` o `None` | Si se pasan, filtra solo estos genes |
| `gene_names` | `list[str]` o `None` | Nombres correspondientes a las filas |

**Retorna**:
```python
{'R2': float, 'Pearson': float, 'Spearman': float, 'N_genes': int}
```

---

## Función orquestadora: [run_all()](file:///c:/Users/Prestamo/Documents/Cuarto/TFG/scGFT/codigo/experimentos/exp%201/scGFT_Evaluator.py#265-349)

```python
ScGFT_Evaluator.run_all(
    adata_real, adata_sint, genes_top,
    col_grupo, grupo_a, grupo_b,
    covariables=None, umbral_spar=1e-4
)
```

Ejecuta **todas** las métricas y devuelve un `pd.DataFrame` de una fila con todas las columnas. Las métricas por grupo usan nombres de columna dinámicos basados en los valores de `grupo_a` y `grupo_b`.

**Requisitos previos**:
- Los `AnnData` deben tener PCA calculado en `adata.obsm['X_pca']`
- La columna `col_grupo` debe existir en `adata.obs`

---

## Caso de uso detallado

### Escenario

Tienes un dataset de RNA single-cell con oligodendrocitos. Has generado datos sintéticos con scGFT aplicando una perturbación espectral. Quieres evaluar cuánto se parecen los datos sintéticos a los reales.

### Paso 1: Preparar los datos

```python
import scanpy as sc
from scGFT_Evaluator import ScGFT_Evaluator

# ── Cargar datos ──
adata_real = sc.read_h5ad("oligodendrocitos_real.h5ad")
adata_sint = sc.read_h5ad("oligodendrocitos_sintetico.h5ad")

# ── Verificar la columna de grupo ──
print(adata_real.obs["diagnosis"].value_counts())
# Control        1200
# Parkinson's     800
```

### Paso 2: Seleccionar genes de interés

```python
# Genes altamente variables o los usados en la perturbación
sc.pp.highly_variable_genes(adata_real, n_top_genes=2000)
genes_top = adata_real.var_names[adata_real.var['highly_variable']].tolist()

# O usar una lista predefinida
# genes_top = ["MOBP", "PLP1", "MBP", "OLIG1", "OLIG2", ...]
```

### Paso 3: Calcular PCA (si no existe)

```python
if 'X_pca' not in adata_real.obsm:
    sc.pp.pca(adata_real, n_comps=50)
    
if 'X_pca' not in adata_sint.obsm:
    sc.pp.pca(adata_sint, n_comps=50)
```

### Paso 4: Ejecutar la evaluación completa

```python
resultados = ScGFT_Evaluator.run_all(
    adata_real=adata_real,
    adata_sint=adata_sint,
    genes_top=genes_top,
    col_grupo="diagnosis",            # columna con las etiquetas
    grupo_a="Control",                # grupo de referencia
    grupo_b="Parkinson's",            # grupo de interés
    covariables=["age", "pmi_hours"], # covariables para el modelo limma
    umbral_spar=1e-4                  # umbral de sparsity
)

print(resultados.T)  # transponer para mejor lectura
```

### Paso 5: Interpretar los resultados

```
Tau_Control              0.9234   ← Varianza preservada en Control ✓
ARI_Control              0.6512   ← Co-expresión parcialmente conservada
Jaccard_Control          0.1823   ← Red gen-gen con solapamiento moderado
Pares_Control            0.7145   ← Buenos pares correlacionados
Tau_Parkinsons           0.8901   ← Varianza preservada en Parkinson ✓
ARI_Parkinsons           0.5634   ← Co-expresión algo peor
Jaccard_Parkinsons       0.1456   ← Solapamiento de red menor
Pares_Parkinsons         0.6823   ← Pares aceptables
MAD                      0.0312   ← Diferencia absoluta baja ✓
R2_medias                0.9876   ← Perfil medio muy similar ✓
Pearson_medias           0.9938   
Sparsity_diff            1.2340   ← Diferencia de sparsity ~1.2% ✓
Limma_Simpson            0.4500   ← 45% de DEGs compartidos
Limma_n_real             234      
Limma_n_sint             189      
MMD_distancia            0.0045   ← Distribuciones muy cercanas ✓
MMD_pvalor               0.3200   ← No significativo (p>0.05) ✓
```

### Paso 6 (Opcional): Ejecutar métricas individuales

```python
# Solo varianza para un subgrupo
mask_ctrl = (adata_real.obs["diagnosis"] == "Control").values
mat_ctrl_real = adata_real.X[mask_ctrl][:, idx_genes].T
mat_ctrl_sint = adata_sint.X[mask_ctrl_sint][:, idx_genes].T

if hasattr(mat_ctrl_real, "toarray"):
    mat_ctrl_real = mat_ctrl_real.toarray()

tau = ScGFT_Evaluator.varianza(mat_ctrl_real, mat_ctrl_sint)
print(f"Tau de Kendall (Control): {tau:.4f}")
```

### Paso 7 (Opcional): Con un dataset completamente diferente

```python
# Dataset de cáncer de mama
adata_real_cancer = sc.read_h5ad("breast_cancer_real.h5ad")
adata_sint_cancer = sc.read_h5ad("breast_cancer_synth.h5ad")

resultados_cancer = ScGFT_Evaluator.run_all(
    adata_real_cancer, adata_sint_cancer, genes_cancer,
    col_grupo="tumor_stage",
    grupo_a="Normal",
    grupo_b="Invasive",
    covariables=["patient_age", "tumor_grade"]
)
# Las columnas serán: Tau_Normal, ARI_Normal, ..., Tau_Invasive, ARI_Invasive, ...
```

---

## Resolución de problemas de dependencias

### `ModuleNotFoundError: No module named 'h5py'`

`scanpy` depende de `anndata`, que a su vez requiere `h5py`. Si pip lo reporta como instalado pero Python no lo puede importar, reinstala forzando:

```bash
pip install --force-reinstall h5py
```

> [!WARNING]
> La reinstalación de `h5py` puede arrastrar `numpy` a la versión 2.x, lo que rompe paquetes compilados con numpy 1.x (`pandas`, `pyarrow`, `torch`, etc.) produciendo errores como:
> ```
> AttributeError: _ARRAY_API not found
> A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
> ```

**Solución**: fijar numpy < 2 después de reinstalar h5py:

```bash
pip install "numpy==1.26.4" "h5py>=3.7,<4"
```

### `OSError: Could not find/load shared object file 'llvmlite.dll'`

Ocurre cuando `llvmlite` (dependencia de `numba`, que usa `scanpy`) es incompatible con la versión de numpy instalada. Solución:

```bash
pip install "numpy==1.26.4" "numba==0.60.0" "llvmlite==0.43.0" "h5py>=3.7,<4"
```

### Versiones compatibles verificadas

| Paquete | Versión | Notas |
|---|---|---|
| `numpy` | 1.26.4 | Requerido < 2.0 por `torch`, `monai`, `synthcity`, etc. |
| `h5py` | 3.16.0 | Compatible con numpy 1.26.4 |
| `numba` | 0.60.0 | Compatible con numpy 1.26.4 |
| `llvmlite` | 0.43.0 | Requerido por numba 0.60.0 |
| `scanpy` | 1.11.5 | Funciona con las versiones anteriores |

> [!TIP]
> Para verificar que todas las dependencias se importan correctamente:
> ```python
> import h5py, numpy, numba, scanpy
> print(h5py.__version__, numpy.__version__, numba.__version__)
> ```

---

## Resumen de métricas y rangos

| Métrica | Rango | Ideal | Qué evalúa |
|---|---|---|---|
| Tau | [-1, 1] | → 1 | Ranking de varianza por gen |
| ARI | [-1, 1] | → 1 | Módulos de co-expresión |
| Pares (Simpson) | [0, 1] | → 1 | Top pares correlacionados |
| Jaccard corregido | (-∞, 1] | > 0 | Red gen-gen (aristas) |
| MAD | [0, ∞) | → 0 | Diferencia absoluta media |
| R² medias | [0, 1] | → 1 | Perfil medio de expresión |
| Sparsity diff | [0, 100] | → 0 | Diferencia de % ceros |
| Limma Simpson | [0, 1] | → 1 | DEGs compartidos |
| MMD | [0, ∞) | → 0 | Distancia distribucional |
| MMD p-valor | [0, 1] | > 0.05 | Test de igualdad distribucional |
