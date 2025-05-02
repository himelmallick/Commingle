import pandas as pd
from models.model_clam import CLAM_MB, CLAM_SB
from .utils import *
from .file_utils import save_pkl, load_pkl

def initiate_model(args, ckpt_path):
    print('Init Model') 
    model_dict = {
        'dropout': args.drop_out,
        'n_classes': args.num_classes,
        'k_sample': args.B,
        'for_harmony': args.harmony}
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: 
        if model_args['n_classes'].item() > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    print_network(model)
    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.relocate()
    model.eval()
    return model

def infer_single_slide(model, features, label, reverse_label_dict, k=1, instance_eval = False, label_tensor = None):
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (CLAM_SB,)):
            logits, Y_prob, Y_hat, A, results_dict = model(features, instance_eval = instance_eval, label = label_tensor)
            # print(results_dict, logits, Y_prob.shape)
            Y_hat = Y_hat.item()
            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]
            A = A.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError
        # print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])
    return ids, preds_str, probs, A, results_dict

def extract_attention_scores(args, label_dict):
    ###
    feature_path = args.tmp_dir+"/bags"
    ckpt_path = "{}/{}".format(args.output_dir, args.checkpoint)
    cell_barcode_dict = load_pkl(args.tmp_dir+"/insts_in_bags.pkl")
    metadata = pd.read_csv(args.tmp_dir+"/bag.csv")
    files = os.listdir(feature_path)
    cell_weights = torch.Tensor()
    ###
    sample_list = []
    barcode_list = []
    results_dict_list = []
    ###
    model = initiate_model(args, ckpt_path)
    for f in files:
        print(f, '>>>>>>>>>>')
        features = torch.load(os.path.join(feature_path, f))
        sample_id = f.split('.')[0]
        if sample_id not in str( cell_barcode_dict.keys() ):
            continue
        # sample_id = int(sample_id)    
        cell_barcodes = cell_barcode_dict[sample_id]
        label = metadata[metadata['slide_id'] == sample_id].label.values[0]
        class_labels = list(label_dict.keys())
        class_encodings = list(label_dict.values())
        label_tensor = torch.tensor([label_dict[label]])
        reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
        ids, preds_str, probs, A, results_dict  = infer_single_slide(model, features, label, reverse_label_dict, k=1,
                                                        instance_eval = True, label_tensor= label_tensor)
        weights = F.softmax(torch.Tensor(A), dim = 0).view(1, -1)
        cell_weights = torch.cat((cell_weights, weights), dim = 1)
        sample_list.extend(np.repeat(sample_id, features.shape[0]))
        barcode_list.extend(cell_barcodes)
        results_dict_list.append(results_dict)
    return cell_weights, sample_list, barcode_list, results_dict_list
