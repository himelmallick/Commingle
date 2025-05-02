import numpy as np
import pandas as pd
import os
import re
import random
import torch
import pickle
import scanpy as sc
import anndata
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def preprocess_sinai(args):
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True )
        os.makedirs(args.tmp_dir+'/bags', exist_ok=True)
        os.makedirs(args.tmp_dir+'/folds', exist_ok=True)
    
    dataInst = pd.read_csv(args.dataset_path)
    dataInst['case'] = dataInst['bag']
    
    # Assign into NumPy Array
    insts = np.array( dataInst.iloc[:,2] )
    bags = np.array( dataInst.iloc[:,-1] )
    cases = np.array( dataInst[['case']] )
    labels = np.array( dataInst.iloc[:,0] )
    feats = np.array( dataInst.iloc[:,3:-1])
    
    # PyTorch Tensor
    label2bool = np.array([True if l == 'Case' else False for l in labels])
    data = TensorDataset(torch.Tensor(feats), torch.Tensor(label2bool))
    
    dataBag = dataInst[['bag', 'case', 'label']]
    dataBag = dataBag.drop_duplicates()
    dataBag = dataBag.rename( columns={
	    'bag': 'slide_id',
	    'case': 'case_id'
    } )
    dataBag.to_csv(args.tmp_dir+"/bag.csv", index = False)
    
    insts_in_bags = {}
    save_dir = "{}/bags".format(args.tmp_dir)
    for b in np.unique(bags): 
        insts_sub = np.where(bags == b)[0]
        feats_sub = data.tensors[0][insts_sub,:]
        insts_in_bags[b] = insts[insts_sub]
        path = "{}/{}.pt".format(save_dir, b )
        torch.save(feats_sub, path)
    path = args.tmp_dir+"/insts_in_bags.pkl"
    writer = open(path,'wb')
    pickle.dump(insts_in_bags, writer)
    writer.close()
    print('Preprocessing Complete')

def preprocess_simulated(args):
    # 'wellKey', 'DonorID', 'Status'
    mu = 0
    sigma = 0.5
    nv = float(mu**2 / (sigma**2 - mu))
    pv = float(mu / sigma**2)
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True )
        os.makedirs(args.tmp_dir+'/bags', exist_ok=True)
        os.makedirs(args.tmp_dir+'/folds', exist_ok=True)
    
    dataInst = pd.read_csv(args.dataset_path)
    dataInst['case_id'] = dataInst['DonorID']
        
    # Assign into NumPy Array
    insts = np.array(dataInst['wellKey'])
    bags = np.array( dataInst['DonorID'] )
    labels = np.array(dataInst['Status'])
    feats = np.array(dataInst.iloc[:,3:-1], dtype=np.float64)
    print('Count matrix shape:', feats.shape)
    # scaler = StandardScaler() # minmaxscaler, visualise, std.dev. per gene 
    
    adata = anndata.AnnData(X=feats)
    if adata.n_vars > 1024:
        sc.pp.normalize_total(adata) # , target_sum=feats.shape[-1]
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=1024)
        highly_variable_mask = adata.var['highly_variable']
        feats = adata.X[:, highly_variable_mask]
    
    # PyTorch Tensor
    label2bool = np.array([True if l == 'Case' else False for l in labels])
    data = TensorDataset(torch.Tensor(feats), torch.Tensor(label2bool))
    
    dataBag = dataInst[['DonorID', 'case_id', 'Status']]
    dataBag = dataBag.drop_duplicates()
    dataBag = dataBag.rename( columns={
	    'DonorID': 'slide_id',
	    'Status': 'label'
    } )
    dataBag.to_csv(args.tmp_dir+"/bag.csv", index = False)
    
    insts_in_bags = {}
    save_dir = "{}/bags".format(args.tmp_dir)
    for b in np.unique(bags): 
        insts_sub = np.where(bags == b)[0]
        feats_sub = data.tensors[0][insts_sub,:]
        insts_in_bags[b] = insts[insts_sub]
        path = "{}/{}.pt".format(save_dir, b )
        torch.save(feats_sub, path)
    path = args.tmp_dir+"/insts_in_bags.pkl"
    writer = open(path,'wb')
    pickle.dump(insts_in_bags, writer)
    writer.close()
    print('Preprocessing Complete')
    
def preprocess_simulated_CB(args):
    # 'wellKey', 'DonorID', 'Status'
    bag_id = 'individual_index'
    status_id = 'group'
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True )
        os.makedirs(args.tmp_dir+'/bags', exist_ok=True)
        os.makedirs(args.tmp_dir+'/folds', exist_ok=True)
    
    dataInst = pd.read_csv(args.dataset_path)
    dataInst['case_id'] = dataInst[bag_id]
        
    # Assign into NumPy Array
    # insts = np.array(dataInst['wellKey'])
    insts = np.array([f'sub-{x}_cell-{y}' for x,y in zip(dataInst[bag_id], dataInst.groupby(bag_id).cumcount())]) 
    bags = np.array( dataInst[bag_id] )
    labels = np.array(dataInst[status_id])
    feats = np.array(dataInst.iloc[:,3:-1], dtype=np.float64)
    print('Count matrix shape:', feats.shape)
    # scaler = StandardScaler() # minmaxscaler, visualise, std.dev. per gene 
    
    adata = anndata.AnnData(X=feats)
    if adata.n_vars > 1024:
        sc.pp.normalize_total(adata) # , target_sum=feats.shape[-1]
        # sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=1024)
        highly_variable_mask = adata.var['highly_variable']
        feats = adata.X[:, highly_variable_mask]
    
    # PyTorch Tensor
    label2bool = np.array([True if l == 'disease' else False for l in labels])
    data = TensorDataset(torch.Tensor(feats), torch.Tensor(label2bool))
    
    dataBag = dataInst[[bag_id, 'case_id', status_id]]
    dataBag = dataBag.drop_duplicates()
    dataBag = dataBag.rename( columns={
	    bag_id: 'slide_id',
	    status_id: 'label'
    } )
    dataBag.to_csv(args.tmp_dir+"/bag.csv", index = False)
    
    insts_in_bags = {}
    save_dir = "{}/bags".format(args.tmp_dir)
    for b in np.unique(bags): 
        insts_sub = np.where(bags == b)[0]
        feats_sub = data.tensors[0][insts_sub,:]
        insts_in_bags[b] = insts[insts_sub]
        path = "{}/{}.pt".format(save_dir, b )
        torch.save(feats_sub, path)
    path = args.tmp_dir+"/insts_in_bags.pkl"
    writer = open(path,'wb')
    pickle.dump(insts_in_bags, writer)
    writer.close()
    print('Preprocessing Complete')   
    
def preprocess_simulated_JO(args):
    # 'wellKey', 'DonorID', 'Status'
    bag_id = 'donor'
    status_id = 'condition_donor'
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True )
        os.makedirs(args.tmp_dir+'/bags', exist_ok=True)
        os.makedirs(args.tmp_dir+'/folds', exist_ok=True)
    
    dataInst = pd.read_csv(args.dataset_path)
    dataInst['case_id'] = dataInst[bag_id]
        
    # Assign into NumPy Array
    insts = np.array(dataInst['Cell']) 
    bags = np.array( dataInst[bag_id] )
    labels = np.array(dataInst[status_id])
    feats = np.array(dataInst.iloc[:,7:-1], dtype=np.float64)
    print('Count matrix shape:', feats.shape)
    # scaler = StandardScaler() # minmaxscaler, visualise, std.dev. per gene 
    
    adata = anndata.AnnData(X=feats)
    # if np.isnan(adata.X).sum() > 0:
    #     adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    if adata.n_vars > 1024:
        sc.pp.normalize_total(adata) # , target_sum=feats.shape[-1]
        # sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=1024)
        highly_variable_mask = adata.var['highly_variable']
        feats = adata.X[:, highly_variable_mask]
    
    # PyTorch Tensor
    label2bool = np.array([True if l == 'Disease' else False for l in labels])
    data = TensorDataset(torch.Tensor(feats), torch.Tensor(label2bool))
    
    dataBag = dataInst[[bag_id, 'case_id', status_id]]
    dataBag = dataBag.drop_duplicates()
    dataBag = dataBag.rename( columns={
	    bag_id: 'slide_id',
	    status_id: 'label'
    } )
    dataBag.to_csv(args.tmp_dir+"/bag.csv", index = False)
    
    insts_in_bags = {}
    save_dir = "{}/bags".format(args.tmp_dir)
    for b in np.unique(bags): 
        insts_sub = np.where(bags == b)[0]
        feats_sub = data.tensors[0][insts_sub,:]
        insts_in_bags[b] = insts[insts_sub]
        path = "{}/{}.pt".format(save_dir, b )
        torch.save(feats_sub, path)
    path = args.tmp_dir+"/insts_in_bags.pkl"
    writer = open(path,'wb')
    pickle.dump(insts_in_bags, writer)
    writer.close()
    print('Preprocessing Complete')        
    
def h5ad_to_csv(fpath):
    adata = sc.read(fpath)
    data_csv = [['label', 'bag', 'instance'] + ['X'+str(i) for i in range(adata.X.shape[-1])]]
    # adata.obs.index = adata.obs['wellKey']
    idxs = adata.obs.index
    
    for idx in idxs:
        tmp=[]
        adata_subset = adata[adata.obs.index == idx]
        tmp.append(adata_subset.obs['Dx_AD'][0])
        tmp.append(adata_subset.obs['Channel'][0])
        tmp.append(idx)
        tmp += adata_subset.X[0].tolist()
        data_csv.append(tmp)
    
    np.savetxt("./saved_data/corrected.csv", data_csv, delimiter =",", fmt ='% s')
    print('Saved in .csv format!')
        
    
