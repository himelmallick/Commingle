import argparse
import numpy as np
import pandas as pd
import os 
import random
import torch
from pathlib import Path
from datasets import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits, preprocess_sinai, create_cv_folds
from utils import train, save_pkl, extract_attention_scores

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters', add_help=False)
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--reg', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--drop_out', default=True, type=bool)
    parser.add_argument('--subtyping', default=False, type=bool)
    parser.add_argument('--early_stopping', default=False, type=bool)
    parser.add_argument('--no_inst_cluster', default=False, type=bool)
    parser.add_argument('--weighted_sample', default=True, type=bool)
    parser.add_argument('--B', default=1, type=int)
    parser.add_argument('--bag_weight', default=0.7, type=float)
    
    parser.add_argument('--harmony', action='store_true',
                        help="CLAM model is chosen acc. to Harmony preprocessed data if the flag is provided")

    # Model parameters
    parser.add_argument('--checkpoint', type=str, default='s_0_best_checkpoint.pt',
                        help="Specifies the checkpoint to be used for evaluation.")
    parser.add_argument('--model_size', type=str, default='tiny',
                        help="Specifies the model to be used.")
    parser.add_argument('--model_type', type=str, default='clam_mb',
                        help="Specifies the model to be used.")                    
                        
    # CV parameters
    parser.add_argument('--cv_folds', default=5, type=int)
    parser.add_argument('--cv_test', default=0.3, type=float)
    parser.add_argument('--cv_train', default=1.0, type=float)
    parser.add_argument('--train_type', default='train', type=str,
                        help="Can be test: balanced test; train: balanced training; proportional: main class ratio in test vs train.")
    parser.add_argument('--lr_train', default=0.3, type=float)                    
    
    # Loss parameters
    parser.add_argument('--bag_loss', type=str, default='svm')
    parser.add_argument('--inst_loss', type=str, default='ce')
    
    # dataset parameters
    parser.add_argument('--dataset', default='sinai')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--first_run', action='store_true')
    parser.add_argument('--dataset_path', type=str, default='../dataset/PsychAD_r0_Dec_28_2022.h5ad--ctype_Micro_PVM--bag_Channel.csv')
    
    parser.add_argument('--output_dir', default='./saved_data/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--tmp_dir', default='./tmp',
                        help='path where to save intermedicate data such as cv fold etc.')                    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--testing', default=False, type=bool)
    parser.add_argument('--log_data', default=True, type=bool)
    
    return parser

def main(args):
    
    # Fix the seed
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(args)
    
    cell_weights, sample_list, barcode_list, results_dict_list = extract_attention_scores(args,
                                                                                          label_dict = {'Control':0, 'AD':1})
    df = pd.DataFrame( {'bag': sample_list, 'inst': barcode_list, 'weight': cell_weights[0,:]} )
    path = "{}/{}".format(args.output_dir, "weights.csv") 
    df.to_csv(path, index=None)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('The CLAM project', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)    
