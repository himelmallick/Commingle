import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
#from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

class Accuracy_Logger(object):
	"""Accuracy logger"""
	def __init__(self, n_classes):
		super(Accuracy_Logger, self).__init__()
		self.n_classes = n_classes
		self.initialize()

	def initialize(self):
		self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
	
	def log(self, Y_hat, Y):
		Y_hat = int(Y_hat)
		Y = int(Y)
		self.data[Y]["count"] += 1
		self.data[Y]["correct"] += (Y_hat == Y)
	
	def log_batch(self, Y_hat, Y):
		Y_hat = np.array(Y_hat).astype(int)
		Y = np.array(Y).astype(int)
		for label_class in np.unique(Y):
			cls_mask = Y == label_class
			self.data[label_class]["count"] += cls_mask.sum()
			self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
	
	def get_summary(self, c):
		count = self.data[c]["count"] 
		correct = self.data[c]["correct"]
		
		if count == 0: 
			acc = None
		else:
			acc = float(correct) / count
		
		return acc, correct, count

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=20, stop_epoch=50, verbose=False):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 20
			stop_epoch (int): Earliest epoch possible for stopping
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
		"""
		self.patience = patience
		self.stop_epoch = stop_epoch
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf

	def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
		elif score < self.best_score:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience and epoch > self.stop_epoch:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, ckpt_name)
			self.counter = 0

	def save_checkpoint(self, val_loss, model, ckpt_name):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), ckpt_name)
		self.val_loss_min = val_loss

def train(datasets, cur, args):
	"""   
		train for a single fold
	"""
	print('\nTraining Fold {}!'.format(cur))
	writer_dir = os.path.join(args['results_dir'].item(), str(cur))
	if not os.path.isdir(writer_dir):
		os.mkdir(writer_dir)

	if args['log_data'].item():
		from tensorboardX import SummaryWriter
		writer = SummaryWriter(writer_dir, flush_secs=15)

	else:
		writer = None

	print('\nInit train/val/test splits...', end=' ')
	'''
	if args.train_only:
		train_split = datasets
		val_split = None
		test_split = None
	elif not args.test_include:
		train_split, val_split = datasets
		test_split = None
	else:
		train_split, val_split, test_split = datasets
	''' ## RL
	train_split, val_split, test_split = datasets ## JO
	if args['cv_val'].item() > 0:
		if args['cv_test'].item() > 0:
			save_splits(
				[ train_split, val_split, test_split ],
				[ 'train', 'val', 'test' ],
				os.path.join( args['results_dir'].item(), 'splits_{}.csv'.format(cur) )
			)
		else:
			save_splits(
				[ train_split, val_split ],
				[ 'train', 'val' ],
				os.path.join( args['results_dir'].item(), 'splits_{}.csv'.format(cur) )
			)
	else:
		save_splits(
			[ train_split ],
			[ 'train' ],
			os.path.join( args['results_dir'].item(), 'splits_{}.csv'.format(cur) )
		)

	print('Done!')
	print("Training on {} samples".format(len(train_split)))
	print("Validating on {} samples".format(len(val_split) if args['cv_val'].item() > 0 else 0))
	print("Testing on {} samples".format(len(test_split) if args['cv_test'].item() > 0 else 0))

	print('\nInit loss function...', end=' ')
	if args['bag_loss'].item() == 'svm':
		from topk.svm import SmoothTop1SVM
		loss_fn = SmoothTop1SVM(n_classes = args['n_classes'].item())
		if device.type == 'cuda':
			loss_fn = loss_fn.cuda()
	else:
		loss_fn = nn.CrossEntropyLoss()
	print('Done!')
	
	print('\nInit Model...', end=' ')
	model_dict = {"dropout": args['drop_out'].item(), 'n_classes': args['n_classes'].item()}
	if args['model_type'].item() == 'clam' and args['subtyping'].item():
		model_dict.update({'subtyping': True})
	
	if args['model_size'].item() is not None and args['model_type'].item() != 'mil':
		model_dict.update({"size_arg": args['model_size'].item()})
	
	if args['model_type'].item() in ['clam_sb', 'clam_mb']:
		if args['subtyping'].item():
			model_dict.update({'subtyping': True})
		
		if args['B'].item() > 0:
			model_dict.update({'k_sample': args['B'].item()})
		
		if args['inst_loss'].item() == 'svm':
			from topk.svm import SmoothTop1SVM
			instance_loss_fn = SmoothTop1SVM(n_classes = 2)
			if device.type == 'cuda':
				instance_loss_fn = instance_loss_fn.cuda()
		else:
			instance_loss_fn = nn.CrossEntropyLoss()
		
		if args['model_type'].item() =='clam_sb':
			model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
		elif args['model_type'].item() == 'clam_mb':
			model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
		else:
			raise NotImplementedError
	
	else: # args.model_type == 'mil'
		if args['n_classes'].item() > 2:
			model = MIL_fc_mc(**model_dict)
		else:
			model = MIL_fc(**model_dict)
	
	model.relocate()
	print('Done!')
	print_network(model)

	print('\nInit optimizer ...', end=' ')
	optimizer = get_optim(model, args)
	print('Done!')
	
	print('\nInit Loaders...', end=' ')
	train_loader = get_split_loader(train_split, training=True, testing = args['testing'].item(), weighted = args['weighted_sample'].item())
	if args['cv_val'].item() > 0:
		val_loader = get_split_loader(val_split,  testing = args['testing'].item())
	if args['cv_test'].item() > 0:
		test_loader = get_split_loader(test_split, testing = args['testing'].item())
	print('Done!')

	print('\nSetup EarlyStopping...', end=' ')
	if args['early_stopping'].item():
		early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

	else:
		early_stopping = None
	print('Done!')
	
	stopping_epoch = 0

	train_acc_vec = []
	train_auc_vec = []
	val_acc_vec = []
	val_auc_vec = []
	
	for epoch in range(args['max_epochs'].item()):
		## diagnosis
		print('epoch:' + str(epoch))
		if args['model_type'].item() in ['clam_sb', 'clam_mb'] and not args['no_inst_cluster'].item():	 
			train_loop_clam(epoch, model, train_loader, optimizer, args['n_classes'].item(), args['bag_weight'].item(), writer, loss_fn)
			# if not args.train_only: ## RL
			if args['cv_val'].item() > 0: ## JO
				stop = validate_clam(cur, epoch, model, val_loader, args['n_classes'].item(), 
									 early_stopping, writer, loss_fn, args['results_dir'].item())
			else:
				stop = False

			###
			_, train_error, train_auc, _ = summary(model, train_loader, args['n_classes'].item())
			train_acc_vec.append( 1- train_error )
			train_auc_vec.append( train_auc )
			###
			if args['cv_val'].item() > 0:
				_, val_error, val_auc, _ = summary(model, val_loader, args['n_classes'].item())
				val_acc_vec.append( 1- val_error )
				val_auc_vec.append( val_auc )
			else:
				val_acc_vec.append( -1 )
				val_auc_vec.append( -1 )				
			
		else:
			train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
			# if not args.train_only: ## RL
			if args['cv_val'].item() > 0: ## JO
				stop = validate(cur, epoch, model, val_loader, args['n_classes'].item(), 
								early_stopping, writer, loss_fn, args['results_dir'].item())
			else:
				stop = False
		
		if stop: 
			stopping_epoch = epoch
			print('stopping epoch:' + str(stopping_epoch))
			print('--------------------------------------')
			break
	
	if stopping_epoch == 0:
		stopping_epoch = args['max_epochs'].item()
	
	if not args['early_stopping'].item():
		stopping_epoch = args['max_epochs'].item()

	if args['early_stopping'].item():
		model.load_state_dict(torch.load(os.path.join(args['results_dir'].item(), "s_{}_checkpoint.pt".format(cur))))
	else:
		torch.save(model.state_dict(), os.path.join(args['results_dir'].item(), "s_{}_checkpoint.pt".format(cur)))
	
		
	results_dict_train, train_error, train_auc, acc_logger = summary(model, train_loader, args['n_classes'].item())
	print('Train error: {:.4f}, ROC AUC: {:.4f}'.format(train_error, train_auc))
	for i in range(args['n_classes'].item()):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

		if writer:
			writer.add_scalar('final/train_class_{}_acc'.format(i), acc, 0)

	if writer:
		writer.add_scalar('final/train_error', train_error, 0)
		writer.add_scalar('final/train_auc', train_auc, 0)
	writer.close()
	#return results_dict, train_auc, 1- train_error, stopping_epoch
	train_acc = 1 - train_error
		
	if args['cv_val'].item() > 0:
		results_dict_val, val_error, val_auc, _= summary(model, val_loader, args['n_classes'].item()) # original code
		print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
		if writer:
			writer.add_scalar('final/val_error', val_error, 0)
			writer.add_scalar('final/val_auc', val_auc, 0)
		writer.close()
		#return results_dict, val_auc, 1-val_error, stopping_epoch
		val_acc = 1 - val_error
	else:
		results_dict_val = None
		val_auc = None
		val_acc = None

	if args['cv_test'].item() > 0:
		results_dict_test, test_error, test_auc, acc_logger = summary(model, test_loader, args['n_classes'].item())
		print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
		for i in range(args['n_classes'].item()):
			acc, correct, count = acc_logger.get_summary(i)
			print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

			if writer:
				writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

		if writer:
			writer.add_scalar('final/test_error', test_error, 0)
			writer.add_scalar('final/test_auc', test_auc, 0)
		writer.close()
		#return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, stopping_epoch
		test_acc = 1 - test_error
	else:
		results_dict_test = None
		test_auc = None
		test_acc = None

	results_dict = {
		'train': results_dict_train,
		'val': results_dict_val,
		'test': results_dict_test
	}
	return results_dict, train_auc, train_acc, val_auc, val_acc, test_auc, test_acc, stopping_epoch, \
		train_acc_vec, train_auc_vec, val_acc_vec, val_auc_vec



def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.train()
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	inst_logger = Accuracy_Logger(n_classes=n_classes)
	
	train_loss = 0.
	train_error = 0.
	train_inst_loss = 0.
	inst_count = 0

	print('\n')
	for batch_idx, (data, label) in enumerate(loader):
		data, label = data.to(device), label.to(device)
		logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)

		acc_logger.log(Y_hat, label)
		loss = loss_fn(logits, label)
		loss_value = loss.item()

		instance_loss = instance_dict['instance_loss']
		inst_count+=1
		instance_loss_value = instance_loss.item()
		train_inst_loss += instance_loss_value
		
		total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

		inst_preds = instance_dict['inst_preds']
		inst_labels = instance_dict['inst_labels']
		inst_logger.log_batch(inst_preds, inst_labels)

		train_loss += loss_value
		if (batch_idx + 1) % 20 == 0:
			print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
				'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

		error = calculate_error(Y_hat, label)
		train_error += error
		
		# backward pass
		total_loss.backward()
		# step
		optimizer.step()
		optimizer.zero_grad()

	# calculate loss and error for epoch
	train_loss /= len(loader)
	train_error /= len(loader)
	
	if inst_count > 0:
		train_inst_loss /= inst_count
		print('\n')
		for i in range(2):
			acc, correct, count = inst_logger.get_summary(i)
			print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

	print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
		if writer and acc is not None:
			writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

	if writer:
		writer.add_scalar('train/loss', train_loss, epoch)
		writer.add_scalar('train/error', train_error, epoch)
		writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
	model.train()
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	train_loss = 0.
	train_error = 0.

	print('\n')
	for batch_idx, (data, label) in enumerate(loader):
		data, label = data.to(device), label.to(device)

		logits, Y_prob, Y_hat, _, _ = model(data)
		
		acc_logger.log(Y_hat, label)
		loss = loss_fn(logits, label)
		loss_value = loss.item()
		
		train_loss += loss_value
		if (batch_idx + 1) % 20 == 0:
			print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
		   
		error = calculate_error(Y_hat, label)
		train_error += error
		
		# backward pass
		loss.backward()
		# step
		optimizer.step()
		optimizer.zero_grad()

	# calculate loss and error for epoch
	train_loss /= len(loader)
	train_error /= len(loader)

	print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
		if writer:
			writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

	if writer:
		writer.add_scalar('train/loss', train_loss, epoch)
		writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	# loader.dataset.update_mode(True)
	val_loss = 0.
	val_error = 0.
	
	prob = np.zeros((len(loader), n_classes))
	labels = np.zeros(len(loader))

	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(loader):
			data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

			logits, Y_prob, Y_hat, _, _ = model(data)

			acc_logger.log(Y_hat, label)
			
			loss = loss_fn(logits, label)

			prob[batch_idx] = Y_prob.cpu().numpy()
			labels[batch_idx] = label.item()
			
			val_loss += loss.item()
			error = calculate_error(Y_hat, label)
			val_error += error
			

	val_error /= len(loader)
	val_loss /= len(loader)

	if n_classes == 2:
		auc = roc_auc_score(labels, prob[:, 1])
	
	else:
		auc = roc_auc_score(labels, prob, multi_class='ovr')
	
	
	if writer:
		writer.add_scalar('val/loss', val_loss, epoch)
		writer.add_scalar('val/auc', auc, epoch)
		writer.add_scalar('val/error', val_error, epoch)

	print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))	 

	if early_stopping:
		assert results_dir
		early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
		
		if early_stopping.early_stop:
			print("Early stopping")
			return True

	return False

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.eval()
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	inst_logger = Accuracy_Logger(n_classes=n_classes)
	val_loss = 0.
	val_error = 0.

	val_inst_loss = 0.
	val_inst_acc = 0.
	inst_count=0
	
	prob = np.zeros((len(loader), n_classes))
	labels = np.zeros(len(loader))
	sample_size = model.k_sample
	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(loader):
			data, label = data.to(device), label.to(device)	  
			logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
			acc_logger.log(Y_hat, label)
			
			loss = loss_fn(logits, label)

			val_loss += loss.item()

			instance_loss = instance_dict['instance_loss']
			
			inst_count+=1
			instance_loss_value = instance_loss.item()
			val_inst_loss += instance_loss_value

			inst_preds = instance_dict['inst_preds']
			inst_labels = instance_dict['inst_labels']
			inst_logger.log_batch(inst_preds, inst_labels)

			prob[batch_idx] = Y_prob.cpu().numpy()
			labels[batch_idx] = label.item()
			
			error = calculate_error(Y_hat, label)
			val_error += error

	val_error /= len(loader)
	val_loss /= len(loader)

	if n_classes == 2:
		auc = roc_auc_score(labels, prob[:, 1])
		aucs = []
	else:
		aucs = []
		binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
		for class_idx in range(n_classes):
			if class_idx in labels:
				fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
				aucs.append(calc_auc(fpr, tpr))
			else:
				aucs.append(float('nan'))

		auc = np.nanmean(np.array(aucs))

	print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
	if inst_count > 0:
		val_inst_loss /= inst_count
		for i in range(2):
			acc, correct, count = inst_logger.get_summary(i)
			print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
	
	if writer:
		writer.add_scalar('val/loss', val_loss, epoch)
		writer.add_scalar('val/auc', auc, epoch)
		writer.add_scalar('val/error', val_error, epoch)
		writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
		for i in range(len(loader)):
			writer.add_scalar('val/label', labels[i], epoch) 
			for class_idx in range(n_classes):
				var_name = 'val/prob_' + str(class_idx)
				writer.add_scalar(var_name, prob[i, class_idx], epoch)
		


	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
		
		if writer and acc is not None:
			writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
	 

	if early_stopping:
		assert results_dir
		early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
		
		if early_stopping.early_stop:
			print("Early stopping")
			return True

	return False

def summary(model, loader, n_classes):
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	model.eval()
	test_loss = 0.
	test_error = 0.

	all_probs = np.zeros((len(loader), n_classes))
	all_labels = np.zeros(len(loader))

	slide_ids = loader.dataset.slide_data['slide_id']
	patient_results = {}

	for batch_idx, (data, label) in enumerate(loader):
		data, label = data.to(device), label.to(device)
		slide_id = slide_ids.iloc[batch_idx]
		with torch.no_grad():
			logits, Y_prob, Y_hat, _, _ = model(data)

		acc_logger.log(Y_hat, label)
		probs = Y_prob.cpu().numpy()
		all_probs[batch_idx] = probs
		all_labels[batch_idx] = label.item()
		
		patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
		error = calculate_error(Y_hat, label)
		test_error += error

	test_error /= len(loader)

	if n_classes == 2:
		auc = roc_auc_score(all_labels, all_probs[:, 1])
		aucs = []
	else:
		aucs = []
		binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
		for class_idx in range(n_classes):
			if class_idx in all_labels:
				fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
				aucs.append(calc_auc(fpr, tpr))
			else:
				aucs.append(float('nan'))

		auc = np.nanmean(np.array(aucs))


	return patient_results, test_error, auc, acc_logger
