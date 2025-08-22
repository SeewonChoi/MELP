from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from lightning import seed_everything, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from melp.datasets.finetune_datamodule import ECGDataModule
from melp.models.kardianet_model import KardiaNetModel
from melp.paths import ROOT_PATH as REPO_ROOT_DIR
from melp.paths import RAW_DATA_PATH

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

'''
CUDA_VISIBLE_DEVICES=0 python main_generation.py \
    --model_name melp --dataset_name icbeb \
    --train_data_pct 0.01 \
    --ckpt_path CKPT_PATH \
    --num_devices 1
'''
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain.py --num_devices 4 --train_data_pct 1 \
    --text_encoder_name fuyingw/heart_bert \
    --lr 2e-4 --model_name melp --batch_size 64 --max_epochs 100 \
    --ecg_encoder_name ecgfm \
    --clip_loss_weight 1.0 --caption_loss_weight 2.0 --local_loss_weight 0.2
'''

def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"melp_genration_{hparams.model_name}_{extension}"
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"logs/melp_genration/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
                            save_last=False, mode="min", save_top_k=2,
                            auto_insert_metric_name=True),
            EarlyStopping(monitor="val/loss", min_delta=0,
                        patience=5, verbose=True, mode="min"),
    ]

    logger_dir = os.path.join(REPO_ROOT_DIR, "logs/melp_genration")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="melp_genration", save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/{extension}/exp_logs")
    
    datamodule = ECGDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_name=hparams.dataset_name,
        batch_size=hparams.batch_size,  
        num_workers=hparams.num_workers,
        train_data_pct=hparams.train_data_pct
    )
    
    if hparams.model_name == "kardianet":
        model = KardiaNetModel(**vars(hparams))
    else:
        raise NotImplementedError

    model.training_steps_per_epoch = len(datamodule.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    pprint(vars(hparams))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # tuner = Tuner(trainer)
    # Find optimal batch size
    # optimal_batch_size = tuner.scale_batch_size(model=model, datamodule=datamodule, init_val=128,
    #                                             mode="binsearch")
    # datamodule.batch_size = optimal_batch_size
    # print(f"Optimal batch size: {optimal_batch_size}")
    # lr_finder = tuner.lr_find(model=model, datamodule=datamodule, max_lr=1e-3)
    # model.lr = lr_finder.suggestion()
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pretraining Multimodal ECG Foundation Model.")
    parser.add_argument("--model_name", type=str, default="kardianet", choices=['kardianet'])
    parser.add_argument("--dataset_name", type=str, default="ptbxl_super_class",
                        choices=["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
                                 "icbeb", "chapman"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="/home/seewon/MELP/data/model.safetensors")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_features", type=int, default=256)
    parser.add_argument("--ecg_encoder_name", type=str, default="ecgfm")
    parser.add_argument("--ecg_encoder_weight", type=str, default="")
    parser.add_argument("--val_dataset_list", type=str, nargs="+", 
                        default=["ptbxl_super_class"])
    hparams = parser.parse_args()

    # set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)
    main(hparams)