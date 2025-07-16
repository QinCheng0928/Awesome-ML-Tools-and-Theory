import torch

class Config:
    # model configuration
    EMBED_DIM = 512
    IMAGE_ENCODER = "./data/vit-base-patch16-224"
    TEXT_ENCODER = "./data/bert-base-uncased"
    
    # training configuration
    BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # data configuration
    MAX_LENGTH = 32
    