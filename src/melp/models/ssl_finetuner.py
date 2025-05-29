from typing import Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
import torch.nn.functional as F
from torchmetrics import AUROC
from einops import rearrange


class SSLFineTuner(LightningModule):
    """Finetunes a self-supervised learning backbone using the standard evaluation protocol of a linear layer
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 256,
        num_classes: int = 2,
        epochs: int = 100,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        decay_epochs: Tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 1e-5,
        use_ecg_patch: bool = False,
        *args,
        **kwargs
    ) -> None:
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.use_ecg_patch = use_ecg_patch

        self.backbone = backbone
        # Freeze the backbone for linear probing
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        self.train_auc = AUROC(task="multilabel", num_labels=num_classes)
        self.val_auc = AUROC(task="multilabel", num_labels=num_classes)
        self.test_auc = AUROC(task="multilabel", num_labels=num_classes)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        auc = self.train_auc(logits.softmax(-1), y.long())

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auc_step", auc, prog_bar=True)
        self.log("train_auc_epoch", self.train_auc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_auc(logits.softmax(-1), y.long())

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_auc", self.val_auc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_auc(logits.softmax(-1), y.long())

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_auc", self.test_auc)

        return loss

    def shared_step(self, batch):
        # Extract features from the backbone
        with torch.no_grad():
            if self.use_ecg_patch:
                ecg_patch = batch["ecg_patch"]
                ecg_patch = rearrange(ecg_patch, 'B N (A T) -> B N A T', T=96)
                # Do attention only on visible ECG patches ...
                mask = ecg_patch.sum(-1) != 0
                t_indices = batch["t_indices"]
                feats = self.backbone.ext_ecg_emb(ecg_patch, mask, t_indices)
            else:
                ecg = batch["ecg"]
                y = batch["label"]
                feats = self.backbone.ext_ecg_emb(ecg)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.linear_layer.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]