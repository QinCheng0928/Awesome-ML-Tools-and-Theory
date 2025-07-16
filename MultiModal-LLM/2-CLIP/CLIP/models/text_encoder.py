import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TextEncoder(nn.Module):
    """BERT text encoder."""
    def __init__(self, model_name="./data/bert-base-uncased", embed_dim=512, freeze_bert=True):
        super().__init__()
        self.bert = BertModel.from_pretrained(os.path.join(base_dir, model_name))
        self.projection = nn.Linear(768, embed_dim)
        
        if freeze_bert:
            # freeze the BERT parameters
            for param in self.bert.parameters():
                param.requires_grad_(False)
            # unfreeze the last 4 layers
            for layer in self.bert.encoder.layer[-4:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings = self.projection(cls_embedding)
        return F.normalize(embeddings, p=2, dim=1)