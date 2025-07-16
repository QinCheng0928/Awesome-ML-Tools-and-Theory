import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ViTImageEncoder(nn.Module):
    """Vision Transformer (ViT) image encoder."""
    def __init__(self, model_name="./data/vit-base-patch16-224", embed_dim=512, freeze_vit=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained(os.path.join(base_dir, model_name))
        self.projection = nn.Linear(768, embed_dim)
        
        if freeze_vit:
            # freeze the ViT parameters
            for param in self.vit.parameters():
                param.requires_grad_(False)
            # unfreeze the last 4 layers
            for layer in self.vit.encoder.layer[-4:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embedding)
        return F.normalize(embeddings, p=2, dim=1)