from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import os
import argparse
import scanpy as sc
import pandas as pd
from pathlib import Path
from utils import *

def remove_single_batches(adata):
    batches = adata.obs[args.batch_key].cat.categories
    
    # Remove batches with single entry
    for batch in batches:
        adata_subset = adata[adata.obs[args.batch_key] == batch]
        if adata_subset.X.shape[0] == 1:
            rows_to_remove = adata.obs[adata.obs[args.batch_key] == batch].index
            adata = adata[~adata.obs.index.isin(rows_to_remove)]
    return adata    

def get_args_parser():
    parser = argparse.ArgumentParser('Set ABC parameters', add_help=False)
    parser.add_argument('--base_path', default='../dataset/', type=str)
    parser.add_argument('--weight_path', default='weights.csv', type=str)
    parser.add_argument('--dataset_name', default='../dataset/IS_new_random.csv', type=str)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--label_key', default='subclass', type=str)
    parser.add_argument('--batch_key', default='Channel', type=str)
    parser.add_argument('--label_value', default='Micro_PVM', type=str)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--output_dir', default='saved_data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    
    return parser

def main(args):
    # read dataset into an anndata object
    inPath = os.path.join(args.base_path, f"{args.dataset_name}.h5ad")
    adata = sc.read(inPath)
    attn = args.attention

    ctype_tf = (adata.obs[args.label_key] == args.label_value)
    print(np.sum( ctype_tf ), np.mean( ctype_tf ), adata.X.shape )
    data_ctype = adata[ ctype_tf ]
    
    adata = remove_single_batches(adata)
    ctype_tf = (adata.obs[args.label_key] == args.label_value)
    data_ctype = adata[ ctype_tf ]    

    idxs = np.arange(adata.X.shape[0])
    
    df_g = pd.DataFrame(data_ctype.X.toarray()[idxs,:], columns = data_ctype.var_names)  
    print('Cells for analysis:', df_g.shape)
    assert np.array_equal(data_ctype.X.toarray()[idxs,:], adata[idxs,:].X.toarray())
    idxs2inst = adata[idxs,:].obs.index
        
    weights = pd.read_csv(os.path.join(args.base_path, f"{args.weight_path}"))
    weights = weights[weights['inst'].isin(idxs2inst.tolist())].reset_index()
    weights = weights.set_index('inst').loc[idxs2inst].reset_index()
    weights = weights.rename(columns = {'level_0':'inst'}).reset_index()
    weights = weights[['bag','inst','weight']]
    
    assert (weights['inst'] == idxs2inst).all()
    column_labels = adata.var_names
    adata.X = adata.X[idxs,:]
    if attn:
        df_meta, df_agg = agg_deseq2(adata, df_g.to_numpy(), column_labels.tolist(), weights, attention=attn)
    else:
        df_meta, df_agg = agg_deseq2(adata, df_g.to_numpy(), column_labels.tolist(), weights, attention=attn)  
        
    assert not df_agg.isna().any().any()
    assert not df_meta.isna().any().any()
        
    dds = DeseqDataSet(
        counts=df_agg,
        metadata=df_meta,
        design_factors="Condition",
    )
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast = ('Condition', 'Control', 'AD'))
    stat_res.summary()
    res = stat_res.results_df
    
    # Saving the results
    if attn:
        res.sort_values('padj').to_csv('deseq_with_attention_simulated.csv', index=True)
    else:
        res.sort_values('padj').to_csv('saved_data/deseq_without_attention_simulated.csv', index=True)  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('The CLAM project', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)    

