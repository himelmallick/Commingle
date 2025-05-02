import datetime, os, sys
##
import numpy as np
import scanpy as sc
import argparse
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set Harmony parameters', add_help=False)
    parser.add_argument('--base_path', default='../dataset/', type=str)
    parser.add_argument('--dataset_name', default='Micro_PVM', type=str)
    parser.add_argument('--label_key', default='subclass', type=str)
    parser.add_argument('--batch_key', default='Channel', type=str)
    parser.add_argument('--output_dir', default='saved_data', type=str)
    
    return parser

def saveas_csv(args, adata):
    data_csv = [['label', 'bag', 'instance'] + ['X'+str(i) for i in range(adata.obsm['X_pca_harmony'].shape[-1])]]
    idxs = adata.obs.index
    for idx in idxs:
        tmp=[]
        adata_subset = adata[adata.obs.index == idx]
        tmp.append(adata_subset.obs['Dx_AD'][0])
        tmp.append(adata_subset.obs['Channel'][0])
        tmp.append(idx)
        tmp += adata_subset.obsm['X_pca_harmony'][0].tolist()
        data_csv.append(tmp)
    
    np.savetxt(args.output_dir+'/corrected.csv', data_csv, delimiter =",", fmt ='% s')
    
def main(args):
    inPath = os.path.join(args.base_path, f"{args.dataset_name}.h5ad")
    data = sc.read(inPath)
    
    batches = data.obs[args.batch_key].cat.categories

    # Remove batches with single entry
    for batch in batches:
        adata_subset = data[data.obs[args.batch_key] == batch]
        if adata_subset.X.shape[0] == 1:
            rows_to_remove = data.obs[data.obs[args.batch_key] == batch].index
            data = data[~data.obs.index.isin(rows_to_remove)]
    print(data.X.shape)
    ## Re-processing step 1: pre-procesing
    sc.pp.recipe_zheng17(data)
    ## Re-processing step 2: PCA
    sc.pp.pca(data)
    ## Re-processing step 3: Harmony
    sc.external.pp.harmony_integrate(data, args.batch_key )
    saveas_csv(args, data)
    print('Saved !!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('The CLAM project', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)    
