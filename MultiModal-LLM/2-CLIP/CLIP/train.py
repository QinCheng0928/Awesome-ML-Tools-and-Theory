import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.clip_model import CLIP, contrastive_loss
from utils.dataloader import CLIPDataset
from utils.config import Config

import os
from tqdm import tqdm

def load_local_flickr30k(data_dir="data/flickr30k"):
    image_dir = os.path.join(data_dir, "images")
    anno_path = os.path.join(data_dir, "annotations", "results_20130124.token")
    
    image_paths = []
    texts = []
    with open(anno_path, "r", encoding="utf-8") as f:
        for line in f:
            img_id, caption = line.split("\t")
            img_name = img_id.split("#")[0]
            image_paths.append(os.path.join(image_dir, img_name))
            texts.append(caption.strip())
    
    return image_paths, texts

def train():
    # initialize configuration
    config = Config()
    
    image_paths, texts = load_local_flickr30k()
    
    dataset = CLIPDataset(image_paths, texts, config.MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # initialize model
    model = CLIP(config.EMBED_DIM).to(config.DEVICE)
    
    # optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(config.DEVICE)
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(pixel_values, input_ids, attention_mask)
            
            loss = contrastive_loss(logits_per_image, logits_per_text)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "clip_model.pth")

if __name__ == "__main__":
    train()