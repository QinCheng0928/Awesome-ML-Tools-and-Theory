import torch
import torch.nn as nn
from .image_encoder import ViTImageEncoder
from .text_encoder import TextEncoder

class CLIP(nn.Module):
    """CLIP model"""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ViTImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())
    
    def forward(self, images, text_input_ids, text_attention_mask):
        image_embeddings = self.image_encoder(images)
        text_embeddings = self.text_encoder(text_input_ids, text_attention_mask)
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2