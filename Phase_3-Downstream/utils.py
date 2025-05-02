import numpy as np
import scanpy
import pandas as pd
import scipy
import statsmodels as sts

def agg(adata, normalized_mat, selected_genes, weights, attention = True):
    """Multiply with instance level weights from ADMIL and Agg at the subject level
        normalized_mat: normalized matrix of the genes
        selected_genes: list of selected genes
        weights: pandas df with instance level weights
        attention: Bool, whether using attention score weights or not, default False
    """
    if attention:
        multiplied_matrix = normalized_mat * weights.weight.to_numpy()[:, np.newaxis]
        df_product = pd.DataFrame(multiplied_matrix, columns = selected_genes)
    else:
        df_product = pd.DataFrame(normalized_mat, columns = selected_genes)
    
    df_product[['bag', 'inst']] = weights[['bag', 'inst']]
    df_subj = df_product.groupby('bag')[selected_genes].agg('sum').reset_index()
    df_labels = (
             pd.merge(adata.obs['Channel'].reset_index(),
             adata.obs['Dx_AD'].reset_index(), how = 'inner', on = 'index')
                .rename(columns = {'Channel':'bag'})
                .drop('index', axis = 1)
                .drop_duplicates())
              
    df_combined = df_labels.merge(df_subj, how = 'inner', on = 'bag')
    
    return df_combined
    
def agg_deseq2(adata, normalized_mat, selected_genes, weights, attention = True):
    """Multiply with instance level weights from ADMIL and Agg at the subject level
        normalized_mat: normalized matrix of the genes
        selected_genes: list of selected genes
        weights: pandas df with instance level weights
        attention: Bool, whether using attention score weights or not, default False
    """
    if attention:
        multiplied_matrix = normalized_mat * weights.weight.to_numpy()[:, np.newaxis]
        df_product = pd.DataFrame(multiplied_matrix, columns = selected_genes)
    else:
        df_product = pd.DataFrame(normalized_mat, columns = selected_genes)
    
    df_product[['bag', 'inst']] = weights[['bag', 'inst']]
    df_count = df_product['bag'].value_counts()
    df_subj = df_product.groupby('bag')[selected_genes].agg('sum').reset_index()  
    df_subj = df_subj.sort_values(by=['bag']).set_index('bag')
    
    if attention:
        print('Multiplying counts ... ')
        df_subj = df_subj.mul(df_count, axis=0)
    
    df_labels = (
             pd.merge(adata.obs['Channel'].reset_index(),
             adata.obs['Dx_AD'].reset_index(), how = 'inner', on = 'index')
                .rename(columns = {'Channel':'bag', 'Dx_AD':'Condition'})
                .drop('index', axis = 1)
                .drop_duplicates()
                )
    
    df_labels = df_labels.astype({'bag':str})
    df_labels = df_labels.sort_values(by=['bag']).set_index('bag')
    
    return df_labels, df_subj.round(0)

def agg_deseq2_simulated(adata, normalized_mat, selected_genes, weights, attention = True):
    """Multiply with instance level weights from ADMIL and Agg at the subject level
        normalized_mat: normalized matrix of the genes
        selected_genes: list of selected genes
        weights: pandas df with instance level weights
        attention: Bool, whether using attention score weights or not, default False
    """
    if attention:
        multiplied_matrix = normalized_mat * weights.weight.to_numpy()[:, np.newaxis]
        df_product = pd.DataFrame(multiplied_matrix, columns = selected_genes)
    else:
        df_product = pd.DataFrame(normalized_mat, columns = selected_genes)
    
    df_product[['bag', 'inst']] = weights[['bag', 'inst']]
    df_count = df_product['bag'].value_counts()
    df_subj = df_product.groupby('bag')[selected_genes].agg('sum').reset_index()  
    df_subj = df_subj.sort_values(by=['bag']).set_index('bag')
    
    if attention:
        print('Multiplying counts ... ')
        df_subj = df_subj.mul(df_count, axis=0)
    
    df_labels = (
             pd.merge(adata['donor'].reset_index(),
             adata['condition_donor'].reset_index(), how = 'inner', on = 'index')
                .rename(columns = {'donor':'bag', 'condition_donor':'Condition'})
                .drop('index', axis = 1)
                .drop_duplicates()
                )
    
    # df_labels = df_labels.astype({'bag':int})
    df_labels = df_labels.sort_values(by=['bag']).set_index('bag')
    
    return df_labels, df_subj.round(0)

def filtering(source_df, abd_thr = 0, prev_thr = 0.1):
    """Filters the instance level data for genes that are lowly expressed 
    Args:
        source_df: annData matrix of the genes
        abd_thr: abundance threshold default 0
        prev_thr: prevalence threshold default 0.1
    Returns:
        pandas df with set of filtered genes
    """
    matrix = source_df.X.toarray()
    df_g = pd.DataFrame(matrix, columns = source_df.var_names) # ['gene_'+str(i) for i in range(1, matrix.shape[1]+1)]
    
    ## filtering
    a = df_g.apply(lambda x: len(x[x>abd_thr])/len(x), axis=0) > prev_thr
    list_of_selected_genes = a[a].index
    
    return df_g[list_of_selected_genes], list_of_selected_genes

def normalization(filter_df):
    """Normalizes the genes using scanpy normalization
    Args:
        filter_df: annData matrix of the genes
    Returns:
        normalized matrix with set of normalized genes
    """
    selected_genes_mat = scipy.sparse.csr_matrix(filter_df.values)
    # normalized_mat = scanpy.pp.normalize_per_cell(selected_genes_mat, key_n_counts='n_counts_all', copy = True)
    normalized_mat = scanpy.pp.normalize_per_cell(selected_genes_mat, copy = True, min_counts=-9999)
    return normalized_mat

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def wilcox_comparison(dataframe, x_col, label_control, label_treatment, selected_genes):
    """Comparison of per gene output by Wilcoxon rank
    Arg:
        dataframe: aggregate data on the per gene level
        x_col: `str` column for per gene comparison
        label_control: `str` label for the control group
        label_treatment: `str` label for the treatment group
    """
    df_control = dataframe[dataframe[x_col] == label_control]
    df_ad = dataframe[dataframe[x_col] == label_treatment]
    wilcoxon = [scipy.stats.ranksums(df_control[col], df_ad[col])[1] for col in selected_genes]
    adj = p_adjust_bh(wilcoxon)
    print(min(adj),'>>>>>>>>>>>>>')
    return pd.DataFrame({'genes': selected_genes, 'adjusted_wilcoxon_pval': adj})

