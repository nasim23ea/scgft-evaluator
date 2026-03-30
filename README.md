# scGFT Evaluator

`scGFT Evaluator` es una herramienta en Python diseñada para verificar la calidad y fidelidad de los datos sintéticos de expresión génica de célula única (scRNA-seq). El objetivo principal es comparar un dataset real con su equivalente generado artificialmente, calculando una serie de métricas estadísticas para determinar en qué grado se están preservando las propiedades biológicas.

El pipeline es agnóstico al tipo celular y la enfermedad a tratar. Genera un DataFrame de manera automatizada con métricas que abarcan desde el análisis exploratorio más sencillo hasta métricas topológicas y reducción de dimensiones multivariantes.

## Instalación

El evaluador requiere **Python 3.9 o superior**. Lo más cómodo es aislar las librerías estadísiticas usando un entorno virtual con `conda`.

```bash
conda create -n scgft_env python=3.10
conda activate scgft_env
```

Instala las dependencias principales empleando el gestor `pip`:

```bash
pip install numpy pandas scipy scikit-learn statsmodels scanpy
```

**Nota sobre I/O:**
Habitualmente trabajarás con matrices gigantes de RNAseq que se comprimen en parquet o formatos similares de hdf5. Para evitar errores de backend en pandas (del tipo `ModuleNotFoundError: No module named 'h5py'`), asegúrate de instalar en tu entorno las librerías dedicadas a la lectura:

```bash
pip install pyarrow fastparquet h5py
```

## Métricas Computadas

La clase `ScGFT_Evaluator` se basa en varios métodos englobados bajo el decorador `@staticmethod`, lo que te permite ejecutar métricas concretas por separado o hacer llamadas conjuntas. Internamente medimos factores como:

### Varianza y Nivel de Expresión Promedio
Monitoriza que el generador sintético reproduzca unas cifras basales y ruidos biológicos sensatos.
- **Correlación de medias**: Compara punto a punto los promedios de todos los genes sacando el coeficiente de Pearson, Spearman y $R^2$.
- **Preservación de Varianza**: En vez de sacar correciones lineales, se hace la varianza gen a gen de las células usando un estimador insesgado (`ddof=1`) y se verifica cómo solapan ordinalmente usando la medida no parámetrica `stats.kendalltau()`.

### Co-expresión y Topología
Analizamos cómo los genes se agrupan e interactúan entre sí bajo la campana.
- **Pares altamente correlacionados**: Extrae los *N* pares matriciales del triángulo superior de Pearson con más peso absoluto. Usamos el Índice de Simpson para cruzar los conjuntos reales y generados, que es muy elástico frente a los desajustes en el tamaño por falsos positivos.
- **Redes de modularidad (ARI)**: Monta clusters jerárquicos de Ward tras acotar las distancias de Pearson en el espectro `[0, 2]`. Las diferencias entre la estructura de módulos de la matriz real y la generada se ponderan mediante el Adjusted Rand Index (`adjusted_rand_score` de sklearn).
- **Jaccard Corregido de Redes**: Binariza interconexiones usando un límite duro (umbral de 0.5 por default). Se extrae también un factor ruido (basado en permutaciones aleatorias) para restárselo al índice observado y comprobar que el solapamiento va más allá del azar puro.

### Análisis Multivariante y MMD
Toma la perspectiva poblacional latente.
- **MMD (Maximum Mean Discrepancy)**: En vez de mirar gen a gen, metemos las células en el espacio del PCA (Componentes Principales) sacando distancias euclidianas mediante Kernel RBF Gaussiano. 
Calcula de manera automática la heurística de la varianza $\sigma$ tomando la mediana. Emite p-valores remuestreando aleatoriamente para verificar si hay certeza de que los sintéticos salen de la misma campana poblacional.

```python
# Así se calcula internamente en el script la distancia acotada MMD:
res = ScGFT_Evaluator.mmd(pca_r, pca_s)
print(f"Distancia: {res['distancia']}, p-valor paramétrico: {res['p_valor']}")
```

### Expresión Diferencial OLS
- **Limma en Python**: Para ahorrarnos la integración nativa y caótica de compilar R y llamar a Limma vía `rpy2`, reimplementamos una aproximación equivalente aplicando regresiones lineales Mínimos Cuadrados Ordinarios (`sm.OLS`). Saca p-valores con las etiquetas de test multivariable corrigiendolo por FDR de Benjamini-Hochberg. Permite inyectar covariables para limpiar el estudio:

```python
resultados_de = ScGFT_Evaluator.limma(
    adata_real, adata_sint,
    genes_top=lista_genes,
    col_grupo="Condicion",
    grupo_a="Sano", grupo_b="Enfermo",
    covariables=["Edad", "PMI", "RIN"] # Las mismas que soporta un DESeq o limma en R
)
```

## Ejemplo Completo

Este es un pipeline básico montando unas matrices exportadas desde `.parquet` en `AnnData` (el formato estándar de `scanpy`).

```python
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from scGFT_Evaluator import ScGFT_Evaluator
import warnings

warnings.filterwarnings('ignore') # Omitir avisos de OLS/divisiones por cero

# 1. Envolver los datos reales y sintéticos en AnnData
# (Asumimos lectura previa pd.read_parquet)
adata_real = sc.AnnData(X=csr_matrix(mat_real), obs=df_meta_r, var=pd.DataFrame(index=nombres_genes))
adata_sint = sc.AnnData(X=csr_matrix(mat_sint), obs=df_meta_s, var=pd.DataFrame(index=nombres_genes))

# El bloque de Maximum Mean Discrepancy demanda el espacio precomputado en PCA
sc.pp.pca(adata_real, n_comps=50)
sc.pp.pca(adata_sint, n_comps=50)

# 2. Bajar cardinalidad centrándose en genes más variables
varianzas = np.var(adata_real.X.toarray() if hasattr(adata_real.X, 'toarray') else adata_real.X, axis=0)
top_2000_genes = list(adata_real.var_names[np.argsort(-varianzas)[:2000]])

# Filtro de intersección seguro
genes_validos = [gene for gene in top_2000_genes if gene in adata_sint.var_names]

# 3. Ejecución completa
resultados = ScGFT_Evaluator.run_all(
    adata_real=adata_real,
    adata_sint=adata_sint,
    genes_top=genes_validos,
    col_grupo="Condicion_Enfermedad", # Nombre exacto dentro del metadato
    grupo_a="Control",
    grupo_b="Mutado",
    covariables=["Edad"] 
)

# Imprimir las 17 métricas estructuradas 
print(resultados.T)
```
