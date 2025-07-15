import torch
import time
from datetime import timedelta

def format_time(elapsed):
    return str(timedelta(seconds=int(round(elapsed))))

def save_model(model, output_dir, name="bert_simple"):
    torch.save(model.state_dict(), f"{output_dir}/{name}.pth")

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model