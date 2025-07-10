import torch
import torch.nn as nn
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from PositionalEncoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    # src_mask.shape: (32, 1, 1, 100), (batch_size, num_heads, query_len, key_len)
    # tgt_mask.shape: (32, 1, 90, 90), (batch_size, num_heads, query_len, key_len)
    def generate_mask(self, src, tgt):
        # The shape of src_mask is (32, 1, 1, 100)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # The shape of src_mask is (32, 1, 1, 90)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        # The shape of nopeak_mask is (1, 90, 90)
        # Get the lower triangular matrix
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        # embedding shape: (32, 100, 512)
        # positional encoding shape: (32, 100, 512)
        # src_embedded shape: (32, 100, 512)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        # enc_output shape: (32, 100, 512)
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        # embedding shape: (32, 90, 512)
        # positional encoding shape: (32, 90, 512)
        # tgt_embedded shape: (32, 90, 512)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        # dec_output shape: (32, 90, 512)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Output
        # output shape: (32, 90, 5000)
        output = self.fc(dec_output)
        
        return output
    
    
if __name__ == "__main__":
    src_vocab_size = 5000  # Size of source vocabulary (number of unique tokens in input language)
    tgt_vocab_size = 5000  # Size of target vocabulary (number of unique tokens in output language)
    d_model = 512          # Dimension of model embeddings
    num_heads = 8          # Number of attention heads in multi-head attention layers
    num_layers = 6         # Number of encoder/decoder layers in the transformer stack
    d_ff = 2048            # Dimension of position-wise FFN hidden size
    dropout = 0.1          # Dropout rate for regularization

    # Create model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout=dropout)

    '''
        # Original sequences (varying lengths)
        ["I"    , "love" , "AI"],
        ["Hello", "world"]

        # After padding (uniform length=3)
        ["I"    , "love" , "AI"        ],
        ["Hello", "world", "<pad> or 0"]
    '''
    # torch.randint(low, high, size)
    src = torch.randint(0, src_vocab_size, (32, 100))  # batch_size=32, seq_len=100
    tgt = torch.randint(0, tgt_vocab_size, (32, 90))   # batch_size=32, seq_len=90

    # Forward pass
    output = model(src, tgt)
    print(output.shape)
    