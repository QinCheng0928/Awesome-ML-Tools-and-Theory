"""
Definition:
    nn.Embedding(vocab_size, embed_dim)
    where 
        vocab_size : The glossary size
        embed_dim  : The embedding vector dimension of each word
    
Forward:
    input.shape  : (batch_size, seq_len)
    output.shape : (batch_size, seq_len, embedding_dim)
"""

if __name__ == "__main__":
    print("This is the README document of Embedding modol.")
