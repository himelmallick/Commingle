from ABC import ABC
import os
import scanpy as sc
import matplotlib
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path


def get_args_parser():
    parser = argparse.ArgumentParser('Set ABC parameters', add_help=False)
    parser.add_argument('--base_path', default='../dataset/', type=str)
    parser.add_argument('--dataset_name', default='Micro_PVM', type=str)
    parser.add_argument('--label_key', default='subclass', type=str)
    parser.add_argument('--batch_key', default='Channel', type=str)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--output_dir', default='saved_data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    
    return parser

def main(args):
    # read dataset into an anndata object
    inPath = os.path.join(args.base_path, f"{args.dataset_name}.h5ad")
    adata = sc.read(inPath)

    batches = adata.obs[args.batch_key].cat.categories

    # Remove batches with single entry
    for batch in batches:
        adata_subset = adata[adata.obs[args.batch_key] == batch]
        if adata_subset.X.shape[0] == 1:
            rows_to_remove = adata.obs[adata.obs[args.batch_key] == batch].index
            adata = adata[~adata.obs.index.isin(rows_to_remove)]
        
    # create the ABC model
    model = ABC(adata, args.batch_key, args.label_key)

    # train the model and integrate the dataset
    integrated = model.batch_correction(epochs=args.epoch) 
    integrated.write_h5ad(args.output_dir+'/corrected.h5ad',compression='gzip')
    print('Saved !!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('The CLAM project', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)    

