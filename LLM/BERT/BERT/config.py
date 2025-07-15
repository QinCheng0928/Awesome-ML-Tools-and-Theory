import torch

class Config:
    def __init__(self):
        # Model parameters
        self.vocab_size = 30522                       # Vocabulary size of BERT-base
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2                      # Segment types for NSP task
        self.initializer_range = 0.02
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.epochs = 3
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data parameters
        self.max_length = 128          # Maximum length per sentence
        self.max_seq_length = 256      # Maximum length for combined sentences
        self.mask_prob = 0.15
        self.short_seq_prob = 0.1      # Probability to create short sequences
        
config = Config()