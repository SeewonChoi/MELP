import torch
import math
import ipdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from einops import rearrange
from timm.models.layers import Mlp
from fairseq_signals_backbone.models.wav2vec2.wav2vec2_cmsc import Wav2Vec2CMSCModel, Wav2Vec2CMSCConfig
from melp.utils.openclip_loss import ClipLoss
from melp.models.base_pretrain_model import BasePretrainModel
from melp.backbone.transformer import AttentionalPooler


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ECGFMModel(BasePretrainModel):
    def __init__(self, 
                 model_size: str = "small", # small by default
                 shared_emb_dim: int = 256,
                 embed_dim_caption: int = 768,
                 use_attentional_pool_contrast: bool = False,
                 use_attentional_pool_caption: bool = False,
                 n_queries_contrast: int = 10,
                 n_queries_caption: int = 128,
                 attn_pooler_heads: int = 8,
                 norm_layer: nn.Module = nn.LayerNorm,
                 proj: str = "linear",
                 drop: float = 0.,
                 proj_bias: bool = False,
                 num_leads: int = 12,
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 softmax_temperature: float = 0.1,
                 lambd: float = 0.0051,
                 *args,
                 **kwargs):
        
        """" Implementation of ECG-FM model.
        Using the Wave2Vec2 model as the ECG encoder: CNN + Transformer
        
        """
        super().__init__(shared_emb_dim=shared_emb_dim,
                         lr=lr,
                         num_leads=num_leads,
                         weight_decay=weight_decay,
                         *args,
                         **kwargs)
        
        self.save_hyperparameters()
        self.num_leads = num_leads
        self.temperature = softmax_temperature

        if model_size == "small":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 8
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "base":
            self.encoder_embed_dim = 768
            self.encoder_attention_heads = 12
            self.encoder_layers = 12
            self.encoder_ffn_embed_dim = 3072
        elif model_size == "large":
            self.encoder_embed_dim = 1024
            self.encoder_attention_heads = 16
            self.encoder_layers = 24
            self.encoder_ffn_embed_dim = 4096
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        print("Using ECG encoder with the following configuration:")
        print(f"encoder_embed_dim: {self.encoder_embed_dim}")
        print(f"encoder_attention_heads: {self.encoder_attention_heads}")
        print(f"encoder_layers: {self.encoder_layers}")
        print(f"encoder_ffn_embed_dim: {self.encoder_ffn_embed_dim}")
        
        self.init_ecg_encoder()

        self.embed_dim_caption = embed_dim_caption
        self.use_attentional_pool_contrast = use_attentional_pool_contrast
        self.use_attentional_pool_caption = use_attentional_pool_caption
    
        head_layers = OrderedDict()
        prev_chs = self.ecg_encoder.cfg.encoder_embed_dim
        if use_attentional_pool_contrast:
            scale = prev_chs ** -0.5
            self.attn_pool_contrast = AttentionalPooler(
                d_model=shared_emb_dim, 
                context_dim=prev_chs, 
                n_head=attn_pooler_heads, 
                n_queries=n_queries_contrast)
            self.ln_contrast = norm_layer(shared_emb_dim)
            self.proj_contrast = nn.Parameter(scale * torch.randn(shared_emb_dim, shared_emb_dim))
        else:
            assert proj, 'projection layer needed if not using attentional pooling.'
            # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
            if proj == 'linear':
                head_layers['drop'] = nn.Dropout(drop)
                head_layers['proj'] = nn.Linear(prev_chs, shared_emb_dim, bias=proj_bias)
            elif proj == 'mlp':
                head_layers['mlp'] = Mlp(prev_chs, 2 * shared_emb_dim, shared_emb_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

        if use_attentional_pool_caption:
            self.attn_pool_caption = AttentionalPooler(
                d_model=embed_dim_caption, context_dim=prev_chs, n_head=attn_pooler_heads, n_queries=n_queries_caption)
            self.ln_caption = norm_layer(embed_dim_caption)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.bn = nn.BatchNorm1d(768, affine=False)
        self.lambd = lambd

    def init_ecg_encoder(self):
        # Here we define Wav2Vec2CMSC model as the ECG encoder
        cfg = Wav2Vec2CMSCConfig(
            apply_mask = True,
            mask_prob = 0.65,
            quantize_targets = True,
            final_dim = 256,
            dropout_input = 0.1,
            dropout_features = 0.1,
            feature_grad_mult = 0.1,
            encoder_embed_dim = self.encoder_embed_dim,
            encoder_attention_heads = self.encoder_attention_heads,
            in_d = 12,
            encoder_layers = self.encoder_layers,
            encoder_ffn_embed_dim = self.encoder_ffn_embed_dim
        )
        self.ecg_encoder = Wav2Vec2CMSCModel(cfg)

    def _global_pool(self, x):
        return torch.mean(x, dim=1)
        
    def forward(self, ecg, normalize=False, return_target=False):
        # only float is possible ...
        if ecg.dim() == 4:
            ecg = rearrange(ecg, "b c n d -> (b c) n d")
        else:
            assert ecg.dim() == 3, "Input tensor must be 3D or 4D"

        ecg_out = self.ecg_encoder(source=ecg.float())
        logits = self.ecg_encoder.get_logits(ecg_out).float()   # 48, 101
        features = self.ecg_encoder.get_features(net_output=ecg_out, aggregate=False).float()

        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast.unsqueeze(0)
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self._global_pool(features)
            pooled = self.head(pooled)
        
        tokens = None
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(features)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None

        if return_target:
            targets = self.ecg_encoder.get_targets(sample=ecg.float(), net_output=ecg_out) # 48

            return {
                "logits": logits,
                "features": pooled,
                "targets": targets,
                "token": tokens,
                "raw_features": features.mean(dim=1)
            }
        else:
            return {
                "logits": logits,
                "features": pooled,
                "token": tokens,
                "raw_features": features.mean(dim=1)
            }
    
    @torch.no_grad()
    # only used for finetune ...
    def ext_ecg_emb(self, ecg, normalize=False):
        assert ecg.dim() == 3, "Input tensor must be 3D"

        ecg_out = self.ecg_encoder(source=ecg.float(), mask=False, features_only=True)
        features = ecg_out["x"].float()

        if self.use_attentional_pool_contrast:
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self._global_pool(features)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled

    def _encode_ecg(self, ecg):
        assert ecg.dim() == 3, "Input tensor must be 3D"
        ecg_out = self.ecg_encoder(source=ecg.float(), mask=False, features_only=True)
        # features = self.ecg_encoder.get_features(net_output=ecg_out, aggregate=False).float()
        # results after CNN-Transformer
        features = ecg_out["x"].float()

        if self.use_attentional_pool_contrast:
            # hierarchical pooling
            pooled = self.attn_pool_contrast(features)
            pooled = self.ln_contrast(pooled)
            pooled = pooled @ self.proj_contrast.unsqueeze(0)
            pooled_beat = pooled.clone()
            pooled = torch.mean(pooled, dim=1)
        else:
            pooled = self._global_pool(features)
            pooled = self.head(features)

        tokens = None
        if self.use_attentional_pool_caption:
            tokens = self.attn_pool_caption(features)
            tokens = self.ln_caption(tokens)
        else:
            tokens = None

        return pooled, pooled_beat, tokens
    
    def encode_ecg(self, ecg):
        ecg_latent, _, _ = self._encode_ecg(ecg)
        return ecg_latent

    def shared_step(self, batch, batch_idx):
        batch_size = len(batch["id"])
        ecg_output = self(batch['ecg'], return_target=True)
        recon_loss = F.cross_entropy(ecg_output["logits"], ecg_output["targets"])
        
        # compute CMSC loss
        ecg_crops = rearrange(ecg_output["features"], "(b n) d -> b n d", b=batch_size)
        ecg_crop_1 = ecg_crops[:, 0]
        ecg_crop_2 = ecg_crops[:, 1]
        patient_id_crop_1 = batch["patient_id"][:, 0].contiguous()
        patient_id_crop_2 = batch["patient_id"][:, 1].contiguous()

        # normalize the embeddings
        ecg_crop_1 = F.normalize(ecg_crop_1, p=2, dim=-1)
        ecg_crop_2 = F.normalize(ecg_crop_2, p=2, dim=-1)

        # write infonce loss for lightning 
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        loss = ClipLoss(
            local_loss=True,
            gather_with_grad=True,
            cache_labels=True,
            rank=rank,
            world_size=world_size,
            use_horovod=False
        )

        logits_1, logits_2 = loss.get_logits(
            ecg_crop_1, ecg_crop_2, self.logit_scale.exp())
        
        # ecg_raw_feature = rearrange(ecg_output["raw_features"], "(b n) d -> b n d", b=batch_size)
        # ecg_raw_feature_1 = ecg_raw_feature[:, 0]
        # ecg_raw_feature_2 = ecg_raw_feature[:, 1]
        # ecg_raw_feature_1 = self.bn(ecg_raw_feature_1)
        # ecg_raw_feature_2 = self.bn(ecg_raw_feature_2)
        # c = ecg_raw_feature_1.T @ ecg_raw_feature_2
        # c.div_(ecg_raw_feature_1.size(0))
        # torch.distributed.all_reduce(c)
        # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # off_diag = off_diagonal(c).pow_(2).sum()
        # bt_loss = on_diag + off_diag * self.lambd
        # bt_loss *= 0.005

        if world_size > 1:
            gathered_patient_id_1 = [torch.zeros_like(patient_id_crop_1) for _ in range(world_size)]
            gathered_patient_id_2 = [torch.zeros_like(patient_id_crop_2) for _ in range(world_size)]
            dist.all_gather(gathered_patient_id_1, patient_id_crop_1)
            dist.all_gather(gathered_patient_id_2, patient_id_crop_2)
            all_patient_id_1 = torch.cat(gathered_patient_id_1, dim=0)
            all_patient_id_2 = torch.cat(gathered_patient_id_2, dim=0)
        else:
            all_patient_id_1 = patient_id_crop_1
            all_patient_id_2 = patient_id_crop_2

        cmsc_targets_1 = (patient_id_crop_1.unsqueeze(1) == all_patient_id_2.unsqueeze(0)).float()
        cmsc_targets_1[cmsc_targets_1 == 0] = float("-inf")
        cmsc_targets_1 = F.softmax(cmsc_targets_1, dim=-1)
        cmsc_targets_2 = (patient_id_crop_2.unsqueeze(1) == all_patient_id_1.unsqueeze(0)).float()
        cmsc_targets_2[cmsc_targets_2 == 0] = float("-inf")
        cmsc_targets_2 = F.softmax(cmsc_targets_2, dim=-1)
        cmsc_loss = (
            F.cross_entropy(logits_1, cmsc_targets_1) + 
            F.cross_entropy(logits_2, cmsc_targets_2)
        ) / 2

        loss_dict = {
            "loss": recon_loss + cmsc_loss,
            "recon_loss": recon_loss,
            "cmsc_loss": cmsc_loss,
        }
        # loss_dict = {
        #     "loss": recon_loss + cmsc_loss + bt_loss,
        #     "recon_loss": recon_loss,
        #     "cmsc_loss": cmsc_loss,
        #     "bt_loss": bt_loss
        # }

        return loss_dict, {}

    def on_train_batch_end(self, *args, **kwargs) -> None:
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100)) 


if __name__ == "__main__":
    from melp.datasets.pretrain_datamodule import ECGTextDataModule
    from melp.paths import RAW_DATA_PATH
    dm = ECGTextDataModule(
        dataset_dir=str(RAW_DATA_PATH),
        dataset_list=["mimic-iv-ecg"],
        val_dataset_list=None,
        batch_size=4,
        num_workers=1,
        train_data_pct=0.1,
        use_cmsc=True,
        use_rlm=True
    )
    
    for batch in dm.val_dataloader():
        break

    model = ECGFMModel()
    # ecg_emb = model.ext_ecg_emb(batch["ecg"][:, 0])
    loss_dict, _ = model.shared_step(batch, 0)
    # print(ecg_emb.shape)
    print(loss_dict)
    ipdb.set_trace()