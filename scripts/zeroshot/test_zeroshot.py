'''
Evaluate zeroshot classification performance of MM-ECG foundation model.
'''
import ipdb
import json
import os
import warnings
import yaml
import torch 
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, f1_score
from melp.models.merl_model import MERLModel
from melp.models.melp_model import MELPModel
from melp.datasets.finetune_datamodule import ECGDataModule
from melp.paths import RAW_DATA_PATH, PROMPT_PATH, RESULTS_PATH, ECGFM_PATH

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str, ckpt_path: str):
    if model_name == "merl":
        if ckpt_path == "":
            model = MERLModel()
        else:
            # model = MERLModel.load_from_checkpoint(ckpt_path)
            model = MERLModel()
            model.load_state_dict(torch.load(ckpt_path))
    elif model_name == "melp":
        if ckpt_path == "":
            model = MELPModel()
        else:
            model = MELPModel.load_from_checkpoint(ckpt_path, ecg_encoder_weight=str(ECGFM_PATH))
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    model = model.to(device)
    model.eval()

    return model


def get_dataloader(dataset_name: str, batch_size: int, num_workers: int):

    print('***********************************')
    print('zeroshot classification set is {}'.format(dataset_name))

    dm = ECGDataModule(        
        dataset_dir=str(RAW_DATA_PATH),
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        train_data_pct=1)
    test_loader = dm.test_dataloader()
    return test_loader


def get_class_emd(model, class_name, device='cuda'):
    model.eval()
    with torch.no_grad(): # to(device=torch.device("cuda"iftorch.cuda.is_available()else"cpu")) 
        zeroshot_weights = []
        # compute embedding through model for each class
        for texts in tqdm(class_name, total=len(class_name), desc='Computing class embeddings'):
            texts = texts.lower()
            texts = [texts] # convert to list
            texts = model._tokenize(texts) # tokenize
            class_embeddings = model.get_text_emb(texts.input_ids.to(device=device),
                                                  texts.attention_mask.to(device=device)
                                                ) # embed with text encoder

            # normalize class_embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # average over templates 
            class_embedding = class_embeddings.mean(dim=0) 
            # norm over new averaged templates
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


def get_ecg_emd(model, loader, zeroshot_weights, device='cuda', softmax_eval=True):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=len(loader), desc='Computing ECG embeddings')):
            ecg = batch['ecg']
            ecg = ecg.to(device=device) 
            # predict
            # ecg_emb = model.ext_ecg_emb(ecg)
            ecg_emb = model.encode_ecg(ecg, normalize=True, proj_contrast=True)
            ecg_emb /= ecg_emb.norm(dim=-1, keepdim=True)

            # obtain logits (cos similarity)
            logits = ecg_emb @ zeroshot_weights
            logits = torch.squeeze(logits, 0) # (N, num_classes)
            if softmax_eval is False: 
                norm_logits = (logits - logits.mean()) / (logits.std())
                logits = torch.sigmoid(norm_logits) 
            
            y_pred.append(logits.cpu().data.numpy())
        
    y_pred = np.concatenate(y_pred, axis=0)
    return np.array(y_pred)


def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        can either be probability estimates of the positive class,
        confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt
    pred_np = pred
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i], average='macro', multi_class='ovo'))

    return AUROCs


def compute_metrics(gt, pred, class_name):
    ''' 
    Compute AUROC, F1, ACC for each class and average
    '''
    # compute original metrics
    AUROCs = compute_AUCs(gt, pred, len(class_name))
    AUROCs = [i*100 for i in AUROCs]

    max_f1s = []
    accs = []
    for i in range(len(class_name)):   
        gt_np = gt[:, i]
        pred_np = pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np>max_f1_thresh))

    max_f1s = [i*100 for i in max_f1s]
    accs = [i*100 for i in accs]
    res_dict = {}
    for i in range(len(class_name)):
        res_dict.update({f'AUROC_{class_name[i]}': AUROCs[i],
                        f'F1_{class_name[i]}': max_f1s[i],
                        f'ACC_{class_name[i]}': accs[i]
        })
    return res_dict


def compute_summary_metrics(ori_df: pd.DataFrame, class_names: List, metrics: List,
                            mode="marco"):
    summary_metrics_dict = {}
    for metric in metrics:
        if mode == "marco":
            summary_metrics_dict[metric] = ori_df[[f"{metric}_{i}" for i in class_names]].values.mean(axis=1)
        else:
            raise NotImplementedError("Micro averaging not implemented.")

    return summary_metrics_dict


def bootstrap_ci(gt, pred, class_name, confidence_level=0.05):
    num_samples = len(gt)
    all_res_dict = []
    for i in tqdm(range(1000), desc='Bootstrapping'):
        sample_ids = np.random.choice(num_samples, num_samples, replace=True)
        gt_sample = gt[sample_ids]  
        pred_sample = pred[sample_ids]
        try:
            res_dict = compute_metrics(gt_sample, pred_sample, class_name)
        except Exception as e:
            print(e)
            continue

        all_res_dict.append(res_dict)
    
    boot_df = pd.DataFrame(all_res_dict)
    lower_values = boot_df.quantile(confidence_level / 2, axis=0)
    upper_values = boot_df.quantile(1 - confidence_level / 2, axis=0)
    mean_values = boot_df.mean(axis=0)
    boot_ci_df = pd.concat([lower_values, mean_values, upper_values], axis=1).T
    return boot_df, boot_ci_df


@torch.no_grad()
def zeroshot_eval(model, test_loader, results_dir, compute_ci=False, save_results=False):
    '''
    Zero-shot evaluation of the model on the given dataset.
    '''
    
    with open(PROMPT_PATH, 'r') as f:
       prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    class_name = test_loader.dataset.labels_name
    # get prompt for each class
    target_class = [prompt_dict[i] for i in class_name]

    # get the target array from testset
    gt = test_loader.dataset.labels

    # get class embedding
    # (embed_dim, num_classes)
    zeroshot_weights = get_class_emd(model, target_class, device=device)
    # get ecg prediction
    pred = get_ecg_emd(model, test_loader, 
                       zeroshot_weights, device=device, softmax_eval=True)

    res_dict = compute_metrics(gt, pred, class_name)
    ori_df = pd.DataFrame(res_dict, index=[0])
    ori_summary_dict = compute_summary_metrics(ori_df, class_name, metrics=['AUROC', 'F1', 'ACC'],
                                               mode="marco")
    ori_summary_df = pd.DataFrame(ori_summary_dict, index=[0])
    ori_df = pd.concat([ori_df, ori_summary_df], axis=1)
    ori_df = ori_df.T
    ori_df.sort_index(ascending=True, inplace=True)
    print("Zero-shot evaluation metrics: ")
    print(ori_summary_df)
    if save_results:
        ori_df.to_csv(os.path.join(results_dir, 'ori.csv'))
    
    if compute_ci:
        # bootstrap confidence interval
        boot_df, boot_ci_df = bootstrap_ci(gt, pred, class_name)
        boot_df.to_csv(os.path.join(results_dir, 'boot.csv'))
        boot_summary_dict = compute_summary_metrics(boot_ci_df, class_name, metrics=['AUROC', 'F1', 'ACC'],
                                                    mode="marco")
        boot_summary_df = pd.DataFrame(boot_summary_dict, index=["lower", "mean", "upper"])
        boot_ci_df.to_csv(os.path.join(results_dir, 'boot_ci.csv'))
        boot_summary_df.to_csv(os.path.join(results_dir, 'boot_summary.csv'))
        print("Zero-shot evaluation metrics: ")
        print(boot_summary_df)
        print("Metrics per class: ")
        print(boot_ci_df)


def main(args):
    model = load_model(model_name=args.model_name, ckpt_path=args.ckpt_path)
    for dataset_name in args.test_sets:
        dataloader = get_dataloader(dataset_name=dataset_name, batch_size=args.batch_size, num_workers=args.num_workers)
        results_dir = str(RESULTS_PATH / f"zeroshot_{args.model_name}_{dataset_name}")
        os.makedirs(results_dir, exist_ok=True)
        zeroshot_eval(model, dataloader, results_dir, compute_ci=args.compute_ci, save_results=args.save_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='merl')
    parser.add_argument('--ckpt_path', type=str, 
                        default='')
    parser.add_argument('--test_sets', type=str, nargs='+', 
                        # default=["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm"])
                        default=["ptbxl_rhythm", "ptbxl_form", "ptbxl_sub_class", "ptbxl_super_class",  
                                "icbeb", "chapman"])
                        # default=["ptbxl_super_class", "icbeb", "chapman"])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--compute_ci', action="store_true", help=r"Compute 95% confidence interval.")
    parser.add_argument('--save_results', action="store_true", help="Save results to csv.")
    args = parser.parse_args()
    
    main(args)