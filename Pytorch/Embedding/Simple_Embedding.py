import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

word_to_idx = {"hello" : 0, "world" : 1}
# 2 word in vocab, 5 dim embedding
embeds = nn.Embedding(2, 5)
lookup_tensor = torch.tensor([word_to_idx["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
'''
    output:
    tensor([[ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845]],
        grad_fn=<EmbeddingBackward0>)
'''