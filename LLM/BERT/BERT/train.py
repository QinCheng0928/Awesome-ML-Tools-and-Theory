import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from model import BertForPreTraining
from dataset import BertDataset, create_examples
from tokenizer import SimpleTokenizer
from config import config
import os
from utils import format_time
import time

def train():
    # Initialize tokenizer
    tokenizer = SimpleTokenizer('../bert-base-uncased')
    
    # Prepare some example data
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "Another example of text data.",
        "The quick brown fox jumps over the lazy dog.",
        "PyTorch is a deep learning framework.",
        "Natural language processing is fascinating.",
        "BERT is a transformer-based model.",
        "It was pretrained on a large corpus.",
        "The model uses masked language modeling.",
        "Next sentence prediction is another task."
    ]
    
    # Create training examples (sentence pairs)
    # [[test[i], test[j], True/False], ...]
    train_examples = create_examples(texts, num_examples=1000)
    
    # Create dataset and data loader
    dataset = BertDataset(train_examples, tokenizer, config.max_seq_length, config.mask_prob, config.short_seq_prob)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize model
    model = BertForPreTraining(config)
    model.to(config.device)
    
    # Prepare optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        t0 = time.time()
        total_loss = 0
        total_mlm_loss = 0
        total_nsp_loss = 0
        
        model.train()
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss, mlm_loss, nsp_loss = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                masked_lm_labels=batch['masked_lm_labels'],
                next_sentence_label=batch['next_sentence_label']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            
            if step % 10 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step}/{len(dataloader)}, Elapsed: {elapsed}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        avg_mlm_loss = total_mlm_loss / len(dataloader)
        avg_nsp_loss = total_nsp_loss / len(dataloader)
        training_time = format_time(time.time() - t0)
        
        print(f"\nAverage Loss: {avg_loss:.4f}")
        print(f"Average MLM Loss: {avg_mlm_loss:.4f}")
        print(f"Average NSP Loss: {avg_nsp_loss:.4f}")
        print(f"Training epoch took: {training_time}\n")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bert_mlm_nsp.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train()