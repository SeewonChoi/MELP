'''
Our proposed approach: KardiaNet
'''
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import ot
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import create_model
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from melp.models.merl_model import MERLModel
from melp.backbone.transformer import  (
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from melp.models.base_pretrain_model import BasePretrainModel
from melp.utils.openclip_loss import CoCaLoss
from melp.models.ecgfm_model import ECGFMModel
from melp.backbone.transformer import AttentionalPooler
from melp.backbone import modeling_finetune
from melp.paths import ECGFM_PATH
from safetensors.torch import load_file
try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StopStringCriteria,
        EosTokenCriteria,
        StoppingCriteriaList,
        Gemma3ForCausalLM
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False
    

class KardiaNetModel(BasePretrainModel):
    """
    KardiaNetModel adapted from MELPModel format
    Architecture: ECG Embedder -> Projection -> Frozen Gemma -> Projection
    """
    def __init__(self, 
                 ecg_encoder_name: str = "ecgfm",
                 ckpt_path: str = "/home/seewon/MELP/data/model.safetensors",
                 gemma_model_name: str = "google/gemma-3-1b-it",
                 val_dataset_list: List = ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
                                          "icbeb", "chapman"],
                 max_seq_len: int = 128,
                 n_queries_contrast: int = 13,
                 prediction_loss_weight: float = 1.0,
                 reconstruction_loss_weight: float = 0.0,
                 shared_emb_dim: int = 256,
                 num_leads: int = 12,
                 num_freeze_layers: int = 0,  # Not used since Gemma is always frozen
                 lr: float = 2e-4,
                 weight_decay: float = 0.2,
                 criterion: str = 'mse',
                 projection_hidden_dim: int = 512,
                 in_features: int = 256,
                 *args,
                 **kwargs):
        
        super().__init__()
        
        # Store hyperparameters following MELP pattern
        self.ecg_encoder_name = ecg_encoder_name
        self.n_queries_contrast = n_queries_contrast
        self.ckpt_path = ckpt_path
        self.shared_emb_dim = shared_emb_dim
        self.prediction_loss_weight = prediction_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.max_seq_len = max_seq_len
        self.gemma_model_name = gemma_model_name
        self.val_dataset_list = val_dataset_list
        self.num_leads = num_leads
        self.lr = lr
        self.weight_decay = weight_decay
        self.projection_hidden_dim = projection_hidden_dim
        self.in_features = in_features
        
        # Loss criterion
        if criterion == 'mse': 
            self.criterion = F.mse_loss
        elif criterion == 'smoothl1': 
            self.criterion = F.smooth_l1_loss
        else: 
            raise Exception(f'Unknown Loss: {criterion}')
        
        # Initialize components
        self.init_ecg_encoder()
        self.init_gemma_model()
        self.init_projections()

    def init_ecg_encoder(self):
        """Initialize ECG encoder exactly like MELP"""
        if self.ecg_encoder_name == "ecgfm":
            from melp.models.ecgfm_model import ECGFMModel
            
            self.ecg_encoder = ECGFMModel(
                use_attentional_pool_contrast=True,
                use_attentional_pool_caption=True,
                n_queries_caption=128,
                model_size="small"
            )
            if self.ckpt_path:
                print("Loading ECGFM model from checkpoint {}".format(self.ckpt_path))
                state_dict = load_file(self.ckpt_path)
                new_ckpt = dict()
                for k, v in state_dict.items():
                    if "ecg_encoder.ecg_encoder" in k:
                        new_key = k.replace("ecg_encoder.ecg_encoder.", "ecg_encoder.")
                        new_ckpt[new_key] = v
                load_result = self.ecg_encoder.load_state_dict(new_ckpt, strict=False)
                print(f"{load_result}")
        else:
            raise NotImplementedError

    def init_gemma_model(self):
        """Initialize frozen Gemma model"""
        print(f"Loading Gemma model: {self.gemma_model_name}")
        
        self.gemma = Gemma3ForCausalLM.from_pretrained(
            self.gemma_model_name, 
            attn_implementation='eager'
        )

        
        # Always freeze Gemma - no trainable parameters
        for param in self.gemma.parameters():
            param.requires_grad = False
        self.gemma.eval()  # Keep in evaluation mode
        
        print("Gemma model frozen - no trainable parameters")
        
        self.gemma_hidden_size = self.gemma.config.hidden_size

    def init_projections(self):
        """Initialize input and output projection layers - these are the only trainable parts"""
        
        # Input Projection: ECG embeddings -> Gemma space
        print(f"Input projection: {self.in_features} -> {self.gemma_hidden_size}")
        self.input_projection = nn.Sequential(
            nn.Linear(self.in_features, self.gemma_hidden_size),
            nn.LayerNorm(self.gemma_hidden_size)
        )

        # Output Projection: Gemma space -> ECG embeddings
        print(f"Output projection: {self.gemma_hidden_size} -> {100}")
        self.output_projection = nn.Sequential(
            nn.Linear(self.gemma_hidden_size, 100),
            # nn.Tanh()  # Bounded output if ECG is normalized to [-1,1]
        )
        
        # Initialize projection weights
        self._init_projection_weights()

    def _init_projection_weights(self):
        """Initialize projection layer weights"""
        for module in [self.input_projection, self.output_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _encode_ecg(self, ecg, normalize: bool = True):
        """Encode ECG exactly like MELP"""
        proj_ecg_emb, ecg_beat_emb, ecg_token_emb = self.ecg_encoder._encode_ecg(ecg)

        if normalize:
            proj_ecg_emb = F.normalize(proj_ecg_emb, dim=-1)

        return proj_ecg_emb, ecg_beat_emb, ecg_token_emb

    def encode_ecg(self, ecg, normalize=True, proj_contrast=True):
        """Public ECG encoding method like MELP"""
        if proj_contrast:
            ecg_latent, _, _ = self._encode_ecg(ecg, normalize=normalize)
        else:
            ecg_latent = self.ecg_encoder.forward_no_head(ecg, normalize=normalize)

        return ecg_latent

    @torch.no_grad()
    def ext_ecg_emb(self, ecg, normalize=False):
        """Extract global ECG embedding exactly like MELP"""
        pooled = self.ecg_encoder.ext_ecg_emb(ecg, normalize=normalize)
        return pooled

    def forward(self, 
                ecg: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True):
        """
        Main forward pass: Embedder -> Projection -> Frozen Gemma -> Projection
        
        Args:
            ecg: (batch_size, num_leads, seq_len) ECG signals
            attention_mask: Optional attention mask
            return_dict: Whether to return dict
            
        Returns:
            Dictionary containing predictions, targets, and loss
        """
        batch_size, n_leads, _ = ecg.shape
        target_ecg = ecg.reshape(batch_size, n_leads, -1, 100)[:, :, 1:, :]
        
        # TODO: this is wrong becuase embedding removed lead info
        target = target_ecg.mean(dim=1)
        
        # 1. ECG Embedder: Get token-level embeddings
        ecg_latent, ecg_beat_emb, ecg_token_emb = self._encode_ecg(ecg, normalize=True)
        # ecg_token_emb shape: (batch_size, seq_len, 768)
        
        seq_len = ecg_beat_emb.shape[1]
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=ecg.device)
        
        # Prepare for next-token prediction
        input_embeddings = ecg_beat_emb[:, :-1]     # Input: positions 0 to n-2
        attention_mask = attention_mask[:, :-1]     # Adjust mask
        target_embeddings = ecg_beat_emb[:, 1:]     # Target: positions 1 to n-1
        
        # 2. Input Projection: ECG space -> Gemma space
        projected_inputs = self.input_projection(input_embeddings) # (batch_size, seq_len-1, gemma_hidden_size)
        
        # 3. Frozen Gemma: Process projected embeddings
        with torch.no_grad():
            gemma_outputs = self.gemma(
                inputs_embeds=projected_inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False
            )
        
        # Get final hidden states
        hidden_states = gemma_outputs.hidden_states[-1]  # (batch, seq_len-1, gemma_hidden_size)
        
        # 4. Output Projection: Gemma space -> ECG space
        predictions = self.output_projection(hidden_states)  # (batch, seq_len-1, ecg_embed_dim)
        
        # Calculate loss
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len-1, 1)
            loss = self.criterion(predictions * mask, target * mask)
        else:
            loss = self.criterion(predictions, target)
        
        if return_dict:
            return {
                'predictions': predictions,
                'targets': target,
                'loss': loss,
                'ecg_latent': ecg_latent,
                'ecg_embeddings': input_embeddings,
                'predicted_embeds': target_embeddings,
                'hidden_states': hidden_states
            }
        else:
            return predictions, target, loss

    def shared_step(self, batch, batch_idx):
        """Training step following MELP's shared_step pattern"""
        
        # Periodic monitoring
        if (batch_idx % 1000 == 0):
            print(f"Training step {batch_idx}")
            # Print number of trainable parameters
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        # Forward pass
        output_dict = self(batch["ecg"]) # batch_size x N leads x sampling rate
        
        # Main prediction loss
        prediction_loss = output_dict["loss"] * self.prediction_loss_weight
        
        # Optional reconstruction loss on global embeddings
        if self.reconstruction_loss_weight > 0:
            # Reconstruct global embedding from sequence
            pred_global = output_dict["predicted_embeds"].mean(dim=1)  # Average over sequence
            target_global = output_dict["ecg_latent"]
            reconstruction_loss = self.criterion(pred_global, target_global) * self.reconstruction_loss_weight
        else:
            reconstruction_loss = torch.tensor(0.0, device=batch["ecg"].device)

        # Total loss
        total_loss = prediction_loss + reconstruction_loss

        loss_dict = {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "reconstruction_loss": reconstruction_loss
        }

        if torch.isnan(loss_dict["loss"]):
            import ipdb; ipdb.set_trace()

        metrics_dict = {}
        
        return loss_dict, metrics_dict
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step following MELP's validation pattern"""
        cur_dataset_name = self.val_dataset_list[dataloader_idx]
        
        with torch.no_grad():
            output_dict = self(batch['ecg'])
            ecg_emb = self.encode_ecg(batch['ecg'], normalize=True, proj_contrast=True)
        
        val_output = {
            'dataloader_idx': dataloader_idx,
            'val_loss': output_dict['loss'],
            'ecg_embeddings': ecg_emb,
            'dataset_name': cur_dataset_name,
            'predictions': output_dict['predictions'],
            'targets': output_dict['targets']
        }
        
        return val_output

    def configure_optimizers(self):
        """Configure optimizers - only projection layers are trainable"""
        # Only projection parameters are trainable (Gemma is frozen)
        trainable_params = []
        trainable_params.extend(list(self.input_projection.parameters()))
        trainable_params.extend(list(self.output_projection.parameters()))
        
        print(f"Optimizing {sum(p.numel() for p in trainable_params):,} parameters")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer

    @torch.no_grad()
    def generate_sequence(self, ecg, num_steps=10):
        """Generate ECG sequence using the trained model"""
        self.eval()
        
        # Get initial embeddings
        _, _, ecg_token_emb = self._encode_ecg(ecg, normalize=True)
        
        # Start with first embedding
        current_input = ecg_token_emb[:, :1]  # First token
        generated_sequence = [current_input]
        
        for _ in range(num_steps):
            # Project to Gemma space
            projected = self.input_projection(current_input)
            
            # Pass through frozen Gemma
            with torch.no_grad():
                outputs = self.gemma(
                    inputs_embeds=projected,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Project back to ECG space
            next_embedding = self.output_projection(outputs.hidden_states[-1])
            generated_sequence.append(next_embedding)
            
            # Use generated embedding as next input
            current_input = next_embedding
        
        return torch.cat(generated_sequence, dim=1)

if __name__ == "__main__":
    # from melp.datasets.pretrain_datamodule import ECGTextDataModule
    # dm = ECGTextDataModule(
    #     dataset_dir="/disk1/*/ECG/raw",
    #     dataset_list=["mimic-iv-ecg"],
    #     val_dataset_list=None,
    #     batch_size=4,
    #     num_workers=1,
    #     train_data_pct=0.1,
    # )
    
    # for batch in dm.val_dataloader():
    #     break
    
    model = KardiaNetModel(
        ckpt_path="/home/seewon/MELP/data/model.safetensors",
        gemma_model_name="google/gemma-3-1b-it",
        projection_hidden_dim=512
    )
    # out = model.shared_step(batch, 0)
    ipdb.set_trace()


    # Only projections are trainable
    # Much smaller than full model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    