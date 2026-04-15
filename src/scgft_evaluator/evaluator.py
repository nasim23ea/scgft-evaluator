import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import adjusted_rand_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

class ScGFT_Evaluator:
    
    @staticmethod
    def varianza(mat_real, mat_sint):
        """Preservación de varianza. Tau de Kendall. (Entrada: Genes x Células)"""
        var_real = np.var(mat_real, axis=1, ddof=1) # ddof para varianza muestral
        var_sint = np.var(mat_sint, axis=1, ddof=1)
        tau, _ = stats.kendalltau(var_real, var_sint, nan_policy='omit')
        return tau

    @staticmethod
    def redes(mat_real, mat_sint, k_modulos=10):
        """Redes de co-expresión. ARI. (Entrada: Genes x Células)"""
        def get_clusters(matriz):
            # Correlación de Pearson entre genes
            with np.errstate(divide='ignore', invalid='ignore'):        # suprimir warnings
                cor_matrix = np.corrcoef(matriz)        # corr de pearson entre filas
            cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)    
            
            # Distancia basada en correlación y clustering jerárquico (Ward)
            dist_matrix = 1 - cor_matrix
            # Asegurar que no hay negativos por redondeo
            dist_matrix = np.clip(dist_matrix, 0, 2) 
            Z = hierarchy.linkage(squareform(dist_matrix, checks=False), method='ward')     # square pasa de matriz a vector 1d con solo la parte triangular superior
            return hierarchy.fcluster(Z, t=k_modulos, criterion='maxclust')     # fcluster devuelve un array con la etiqueta del cluster para cada gen

        labels_real = get_clusters(mat_real)
        labels_sint = get_clusters(mat_sint)
        return adjusted_rand_score(labels_real, labels_sint)

    @staticmethod
    def pares(mat_real, mat_sint, n_pares=1000, gene_names=None):
        """Pares altamente correlacionados. Índice de Simpson."""
        if gene_names is None:      # si no se pasan nombres de genes, se usan los índices
            gene_names = np.arange(mat_real.shape[0]).astype(str)
            
        def get_top_pairs(matriz):
            with np.errstate(divide='ignore', invalid='ignore'):
                cor_mat = np.corrcoef(matriz)
            cor_mat = np.nan_to_num(cor_mat, nan=0.0)
            
            # Extraer el triángulo superior (sin la diagonal, k=1)
            r, c = np.triu_indices_from(cor_mat, k=1)       # r filas y c columnas
            vals = cor_mat[r, c]
            
            # Ordenar por valor absoluto descendente
            idx_sort = np.argsort(-np.abs(vals))
            top_idx = idx_sort[:n_pares]
            
            # Crear identificadores únicos de pares
            pairs = set()
            for i in top_idx:
                g1, g2 = gene_names[r[i]], gene_names[c[i]]
                pairs.add(f"{min(g1, g2)}_{max(g1, g2)}")
            return pairs

        p_real = get_top_pairs(mat_real)
        p_sint = get_top_pairs(mat_sint)
        min_len = min(len(p_real), len(p_sint))
        
        if min_len == 0: 
            return 0.0
            
        return len(p_real.intersection(p_sint)) / min_len

    @staticmethod
    def _limma_approx(mat, covars_df):
        """
        Aproximacion a limma usando OLS gen a gen con correccion FDR.
        covars_df debe tener 'Grupo' (0/1) como primera columna y opcionalmente otras covariables.
        """
        X = sm.add_constant(covars_df)
        pvals = []
        for i in range(mat.shape[0]):
            y = mat[i, :]
            model = sm.OLS(y, X).fit()
            pvals.append(model.pvalues.iloc[1] if len(model.pvalues) > 1 else 1.0)
            
        pvals = np.array(pvals)
        pvals[np.isnan(pvals)] = 1.0
        _, pvals_adj, _, _ = multipletests(pvals, method='fdr_bh')
        return np.where(pvals_adj < 0.05)[0]

    @staticmethod
    def limma(adata_real, adata_sint, genes_top, col_grupo, 
              grupo_a, grupo_b, covariables=None):
        """
        Expresión diferencial entre grupo_a y grupo_b usando OLS gen a gen.
        
        Parámetros:
            col_grupo:    nombre de la columna en adata.obs con las etiquetas de grupo
            grupo_a:      etiqueta del grupo de referencia (ej: 'Control')
            grupo_b:      etiqueta del grupo de interés (ej: 'Enfermedad')
            covariables:  lista opcional de nombres de columnas adicionales en adata.obs
                          para incluir como covariables en el modelo (ej: ['Edad', 'PMI'])
        """
        def get_de(adata):
            genes_found = [g for g in genes_top if g in adata.var_names]
            idx_genes = [adata.var_names.get_loc(g) for g in genes_found]
            mat = adata.X[:, idx_genes].T
            if hasattr(mat, "toarray"): mat = mat.toarray()
            
            # Filtrar solo las células de los dos grupos
            etiquetas = adata.obs[col_grupo]
            valid_cells = etiquetas.isin([grupo_a, grupo_b])
            
            mat = mat[:, valid_cells]
            df_diseno = pd.DataFrame({
                'Grupo': (etiquetas[valid_cells] == grupo_b).astype(int)
            })
            
            # Añadir covariables opcionales
            if covariables:
                for col in covariables:
                    df_diseno[col] = adata.obs[col][valid_cells].values
            
            de_idx = ScGFT_Evaluator._limma_approx(mat, df_diseno)
            return set([genes_found[i] for i in de_idx])

        de_r = get_de(adata_real)
        de_s = get_de(adata_sint)
        min_len = min(len(de_r), len(de_s))
        simpson = len(de_r.intersection(de_s)) / min_len if min_len > 0 else 0.0
        return {'simpson': simpson, 'n_real': len(de_r), 'n_sint': len(de_s)}

    @staticmethod
    def mmd(pca_real, pca_sint, sigma=None, n_perms=100, sample_size=500):
        """Maximum Mean Discrepancy con Kernel RBF"""
        def calc_mmd2(X, Y, sig):
            n, m = X.shape[0], Y.shape[0]
            Kxx = np.exp(-cdist(X, X, 'sqeuclidean') / (2 * sig**2))
            np.fill_diagonal(Kxx, 0)
            Kyy = np.exp(-cdist(Y, Y, 'sqeuclidean') / (2 * sig**2))
            np.fill_diagonal(Kyy, 0)
            Kxy = np.exp(-cdist(X, Y, 'sqeuclidean') / (2 * sig**2))
            
            return np.sum(Kxx) / (n*(n-1)) + np.sum(Kyy) / (m*(m-1)) - 2*np.mean(Kxy)

        # Submuestreo equilibrado
        n_s = min(sample_size, pca_real.shape[0], pca_sint.shape[0])
        np.random.seed(42)
        idx_r = np.random.choice(pca_real.shape[0], n_s, replace=False)
        idx_s = np.random.choice(pca_sint.shape[0], n_s, replace=False)
        X = pca_real[idx_r, :]
        Y = pca_sint[idx_s, :]

        if sigma is None:
            Z = np.vstack([X, Y])
            # Heurística de la mediana (a partir de una muestra)
            dists = pdist(Z[:min(1000, Z.shape[0])])
            sigma = np.median(dists)
            print(f"sigma auto mediana: {sigma:.4f}")

        mmd2_obs = calc_mmd2(X, Y, sigma)       # MMD al cuadrado observada

        # Permutaciones para calcular p-valor
        Z_pool = np.vstack([X, Y])
        mmd2_perms = []
        for _ in range(n_perms):
            idx = np.random.choice(Z_pool.shape[0], n_s, replace=False)
            mask = np.ones(Z_pool.shape[0], dtype=bool)
            mask[idx] = False
            Xp = Z_pool[idx, :]
            Yp = Z_pool[mask, :]
            if Yp.shape[0] > n_s:
                Yp = Yp[np.random.choice(Yp.shape[0], n_s, replace=False), :]
            mmd2_perms.append(calc_mmd2(Xp, Yp, sigma))

        mmd_obs = np.sqrt(max(mmd2_obs, 0))
        p_valor = np.mean(np.array(mmd2_perms) >= mmd2_obs)

        return {'distancia': mmd_obs, 'mmd2': mmd2_obs, 'sigma': sigma, 'p_valor': p_valor}

    @staticmethod
    def sparsity(mat_real, mat_sint, umbral=0.0):
        ceros_real = np.mean(mat_real <= umbral)
        ceros_sint = np.mean(mat_sint <= umbral)
        return {
            'pct_ceros_real': ceros_real * 100,
            'pct_ceros_sint': ceros_sint * 100,
            'diferencia_absoluta': abs(ceros_real - ceros_sint) * 100
        }

    @staticmethod
    def jaccard_redes(mat_real, mat_sint, umbral_cor=0.5, n_random=100, gene_names=None):
        if gene_names is None: 
            gene_names = np.arange(mat_real.shape[0]).astype(str)
            
        def get_edges(matriz):
            with np.errstate(divide='ignore', invalid='ignore'):
                cor_mat = np.corrcoef(matriz)
            cor_mat = np.nan_to_num(cor_mat, nan=0.0)
            
            r, c = np.triu_indices_from(cor_mat, k=1)
            mask = np.abs(cor_mat[r, c]) >= umbral_cor
            edges = set()
            for i in np.where(mask)[0]:
                g1, g2 = gene_names[r[i]], gene_names[c[i]]
                edges.add(f"{min(g1, g2)}_{max(g1, g2)}")
            return edges

        edges_real = get_edges(mat_real)
        edges_sint = get_edges(mat_sint)
        
        interseccion = len(edges_real.intersection(edges_sint))
        union_total = len(edges_real.union(edges_sint))
        if union_total == 0: return 0.0
        jaccard_obs = interseccion / union_total

        # Calcular Jaccard promedio con permutaciones aleatorias
        if n_random > 0:
            genes = list(gene_names)
            jaccard_random = []
            for _ in range(n_random):
                perm = dict(zip(genes, np.random.permutation(genes)))
                edges_perm = set()
                for e in edges_sint:
                    nodos = e.split("_")
                    g1, g2 = perm[nodos[0]], perm[nodos[1]]
                    edges_perm.add(f"{min(g1, g2)}_{max(g1, g2)}")
                
                int_r = len(edges_real.intersection(edges_perm))
                uni_r = len(edges_real.union(edges_perm))
                jaccard_random.append(int_r / uni_r if uni_r > 0 else 0)
            
            beta = np.mean(jaccard_random)
            return jaccard_obs - beta
        
        return jaccard_obs

    @staticmethod
    def mad(mat_real, mat_sint):
        if mat_real.shape != mat_sint.shape:
            print("Warning: Las matrices no tienen las mismas dimensiones.")
            return np.nan
        return np.mean(np.abs(mat_real - mat_sint))

    @staticmethod
    def correlacion_medias(mat_real, mat_sint, genes_deg=None, gene_names=None):
        # Si se pasan DEGs, filtra solo esos genes
        if genes_deg is not None and gene_names is not None:
            idx_usar = [i for i, g in enumerate(gene_names) if g in genes_deg]
            if len(idx_usar) < 3:
                print("Warning: Menos de 3 DEGs encontrados, usando todos los genes")
            else:
                mat_real = mat_real[idx_usar, :]
                mat_sint = mat_sint[idx_usar, :]
        
        med_real = np.mean(mat_real, axis=1)
        med_sint = np.mean(mat_sint, axis=1)
        
        pearson, _ = stats.pearsonr(med_real, med_sint)
        spearman, _ = stats.spearmanr(med_real, med_sint)
        
        return {
            'R2': round(pearson**2, 4),
            'Pearson': round(pearson, 4),
            'Spearman': round(spearman, 4),
            'N_genes': len(med_real)
        }

    @staticmethod
    def run_all(adata_real, adata_sint, genes_top, 
                col_grupo, grupo_a, grupo_b,
                covariables=None, umbral_spar=1e-4):
        """
        Ejecuta todas las métricas de evaluación.
        
        Parámetros:
            adata_real:   AnnData con datos reales
            adata_sint:   AnnData con datos sintéticos
            genes_top:    lista de genes a evaluar
            col_grupo:    nombre de la columna en .obs con las etiquetas de grupo
            grupo_a:      etiqueta del grupo de referencia (ej: 'Control')
            grupo_b:      etiqueta del grupo de interés (ej: 'Enfermedad')
            covariables:  lista opcional de columnas adicionales para el modelo limma
            umbral_spar:  umbral para considerar un valor como cero en sparsity
        """
        print("\n[Evaluador] Ejecutando batería de métricas...\n")
        
        # Filtrar a los genes top y transponer a (Genes x Células)
        idx_r = [adata_real.var_names.get_loc(g) for g in genes_top if g in adata_real.var_names]
        idx_s = [adata_sint.var_names.get_loc(g) for g in genes_top if g in adata_sint.var_names]
        
        mat_r = adata_real.X[:, idx_r].T
        mat_s = adata_sint.X[:, idx_s].T
        if hasattr(mat_r, "toarray"): mat_r = mat_r.toarray()
        if hasattr(mat_s, "toarray"): mat_s = mat_s.toarray()
        
        # Índices por grupo
        mask_a_r = (adata_real.obs[col_grupo] == grupo_a).values
        mask_a_s = (adata_sint.obs[col_grupo] == grupo_a).values
        mask_b_r = (adata_real.obs[col_grupo] == grupo_b).values
        mask_b_s = (adata_sint.obs[col_grupo] == grupo_b).values
        
        mat_a_r, mat_a_s = mat_r[:, mask_a_r], mat_s[:, mask_a_s]
        mat_b_r, mat_b_s = mat_r[:, mask_b_r], mat_s[:, mask_b_s]
        
        print(f"Grupo A ({grupo_a}): real={mat_a_r.shape[1]} | sint={mat_a_s.shape[1]}")
        print(f"Grupo B ({grupo_b}): real={mat_b_r.shape[1]} | sint={mat_b_s.shape[1]}")
        
        # PCA. verificar que existe
        if 'X_pca' not in adata_real.obsm:
            raise ValueError("adata_real no tiene PCA calculado. Ejecuta sc.pp.pca primero.")
        if 'X_pca' not in adata_sint.obsm:
            raise ValueError("adata_sint no tiene PCA calculado. Ejecuta sc.pp.pca primero.")
        
        n_pcs_r = min(50, adata_real.obsm['X_pca'].shape[1])
        n_pcs_s = min(50, adata_sint.obsm['X_pca'].shape[1])
        pca_r = adata_real.obsm['X_pca'][:, :n_pcs_r]
        pca_s = adata_sint.obsm['X_pca'][:, :n_pcs_s]
        
        # Métricas
        res_mmd = ScGFT_Evaluator.mmd(pca_r, pca_s)
        res_spar = ScGFT_Evaluator.sparsity(mat_r, mat_s, umbral=umbral_spar)
        res_mad = ScGFT_Evaluator.mad(mat_r, mat_s)
        res_cor = ScGFT_Evaluator.correlacion_medias(mat_r, mat_s)
        res_limma = ScGFT_Evaluator.limma(adata_real, adata_sint, genes_top, 
                                          col_grupo=col_grupo, grupo_a=grupo_a, 
                                          grupo_b=grupo_b, covariables=covariables)
        
        res_tau_a = ScGFT_Evaluator.varianza(mat_a_r, mat_a_s)
        res_tau_b = ScGFT_Evaluator.varianza(mat_b_r, mat_b_s)
        res_ari_a = ScGFT_Evaluator.redes(mat_a_r, mat_a_s)
        res_ari_b = ScGFT_Evaluator.redes(mat_b_r, mat_b_s)
        res_jacc_a = ScGFT_Evaluator.jaccard_redes(mat_a_r, mat_a_s, gene_names=genes_top)
        res_jacc_b = ScGFT_Evaluator.jaccard_redes(mat_b_r, mat_b_s, gene_names=genes_top)
        res_pares_a = ScGFT_Evaluator.pares(mat_a_r, mat_a_s, gene_names=genes_top)
        res_pares_b = ScGFT_Evaluator.pares(mat_b_r, mat_b_s, gene_names=genes_top)
        
        # Nombres de columna dinámicos basados en los grupos
        label_a = grupo_a.replace("'", "").replace(" ", "_")
        label_b = grupo_b.replace("'", "").replace(" ", "_")
        
        return pd.DataFrame([{
            f'Tau_{label_a}': round(res_tau_a, 4), f'ARI_{label_a}': round(res_ari_a, 4), 
            f'Jaccard_{label_a}': round(res_jacc_a, 4), f'Pares_{label_a}': round(res_pares_a, 4),
            f'Tau_{label_b}': round(res_tau_b, 4), f'ARI_{label_b}': round(res_ari_b, 4), 
            f'Jaccard_{label_b}': round(res_jacc_b, 4), f'Pares_{label_b}': round(res_pares_b, 4),
            'MAD': round(res_mad, 4), 'R2_medias': res_cor['R2'], 'Pearson_medias': res_cor['Pearson'],
            'Sparsity_diff': round(res_spar['diferencia_absoluta'], 4),
            'Limma_Simpson': round(res_limma['simpson'], 4),
            'Limma_n_real': res_limma['n_real'], 'Limma_n_sint': res_limma['n_sint'],
            'MMD_distancia': round(res_mmd['distancia'], 6), 'MMD_pvalor': res_mmd['p_valor']
        }])