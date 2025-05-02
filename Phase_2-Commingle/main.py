import argparse
import numpy as np
import pandas as pd
import os 
import random
import torch
from pathlib import Path
from datasets import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits, preprocess_sinai, create_cv_folds, h5ad_to_csv, preprocess_simulated_JO
from utils import train, save_pkl

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters', add_help=False)
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--reg', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=500, type=int)
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
    parser.add_argument('--dataset_path', type=str, default='../data/IS_new_random.csv')
    
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

# %% Single Run of CLAM-SMILE
def runSingle(dataset, f, args):
    train_dataset, test_dataset = dataset.return_splits(from_id = False, 
                                                        csv_path = os.path.join(args.tmp_dir, 'folds/{}_{}_{}'.format(int( 100 * args.cv_test ),
                                                                                                                      args.cv_folds,
                                                                                                                      f) + '.csv'),
		                                          test_frac=args.cv_test
	                                                  )
    datasets = (train_dataset, test_dataset)
    return train(datasets, f, args)


def main(args):
    
    # Fix the seed
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # torch.cuda.set_device(1)
    print(args)
    
    if args.first_run:
        if args.dataset_path.endswith(".h5ad"):
            h5ad_to_csv(args.dataset_path)
            args.dataset_path = './saved_data/corrected.csv'
        preprocess_sinai(args)
    
    dataset = Generic_MIL_Dataset(data_dir = args.tmp_dir,
		                  csv_path = args.tmp_dir+'/bag.csv',
		                  label_dict = {'Control':0, 'AD':1},
		                  seed=args.seed, 
		                  patient_strat=True, 
		                  ignore=[])
    if args.cv_folds > 1:
        create_cv_folds(dataset, args)
    
    all_stopping_epoch, all_train_auc, all_train_acc, all_test_auc, all_test_acc = [], [], [], [], []
    all_train_acc_vec, all_train_auc_vec = [], []
    
    for f in range(args.cv_folds):
        (results, train_auc, train_acc, test_auc, test_acc,
         stopping_epoch, train_acc_vec, train_auc_vec) = runSingle(dataset, f, args)
        all_stopping_epoch.append(stopping_epoch)
        all_train_auc.append(train_auc)
        all_train_acc.append(train_acc)
        if args.cv_test > 0:
            all_test_auc.append(test_auc)
            all_test_acc.append(test_acc)
        else:
            all_test_auc.append(-1)
            all_test_acc.append(-1)
        all_train_acc_vec.append( train_acc_vec )
        all_train_auc_vec.append( train_auc_vec )
        
        results_dir = args.output_dir
        path = '{}/{}.pkl'.format(results_dir, f)
        save_pkl( path, results )
        
        conv = pd.DataFrame( {'train_acc': train_acc_vec, 'train_auc': train_auc_vec} )
        path = '{}/{}.csv'.format( results_dir, f )
        conv.to_csv(path)
    
    final_df = pd.DataFrame({'folds': args.cv_folds, 'stopping_epoch': all_stopping_epoch,
        'train_auc': all_train_auc, 'train_acc' : all_train_acc,
                             'test_auc': all_test_auc, 'test_acc' : all_test_acc} )
    
    final_df.loc[len(final_df.index)] = {'folds': 'Average', 
                                         'stopping_epoch': np.mean(all_stopping_epoch),
                                         'train_auc': np.mean(all_train_auc),
                                         'train_acc' : np.mean(all_train_acc),
                                         'test_auc': np.mean(all_test_auc),
                                         'test_acc': np.mean(all_test_acc)}
    final_df.loc[len(final_df.index)] = {'folds': 'Std. Dev.', 
                                         'stopping_epoch': np.std(all_stopping_epoch),
                                         'train_auc': np.std(all_train_auc),
                                         'train_acc' : np.std(all_train_acc),
                                         'test_auc': np.std(all_test_auc),
                                         'test_acc': np.std(all_test_acc)}
    
    max_index = final_df['test_acc'].idxmax()
    with open('{}/maxidx.txt'.format(results_dir), 'w') as fl:
        fl.write(str(max_index))
    
    print(final_df)
    path = '{}/summary.csv'.format(results_dir)
    final_df.to_csv(path)
    
    print("Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('The CLAM project', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)    
