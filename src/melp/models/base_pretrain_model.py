from typing import Any
import torch
import torch.nn as nn
from lightning import LightningModule
from timm.optim import create_optimizer_v2
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class BasePretrainModel(LightningModule):
    def __init__(self, 
                 ecg_encoder_name: str = "resnet18",
                 text_encoder_name: str = "ncbi/MedCPT-Query-Encoder",
                 num_leads: int = 12,
                 shared_emb_dim: int = 256,
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 training_steps_per_epoch: int = 1000,
                 max_epochs: int = 100,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ecg_encoder_name = ecg_encoder_name
        self.text_encoder_name = text_encoder_name
        self.shared_emb_dim = shared_emb_dim
        self.num_leads = num_leads
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps_per_epoch = training_steps_per_epoch
        self.max_epochs = max_epochs
        self.warmup_epochs = int(0.5 * self.max_epochs)
        assert self.training_steps_per_epoch > 1, "training_steps_per_epoch must be greater than 1"

    def configure_optimizers(self):

        # optimizer = create_optimizer_v2(
        #     self.parameters(),
        #     "adamw",
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     eps=1e-8,
        #     betas=(0.9, 0.95),
        # )

        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.lr, 
                                      weight_decay=self.weight_decay,
                                      betas=(0.9, 0.95))

        warmup_steps = self.training_steps_per_epoch * self.warmup_epochs
        total_steps = self.training_steps_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=warmup_steps,
                T_mult=1,
                eta_min=1e-8,
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        # scheduler = {
        #     "scheduler": CosineAnnealingWarmupRestarts(
        #         optimizer,
        #         first_cycle_steps=total_steps,
        #         cycle_mult=1.0,
        #         max_lr=self.lr,
        #         min_lr=1e-8,
        #         warmup_steps=warmup_steps,
        #         gamma=1.0
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict

    def test_step(self, batch, batch_idx):
        loss_dict, metrics_dict = self.shared_step(batch, batch_idx)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in metrics_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_dict