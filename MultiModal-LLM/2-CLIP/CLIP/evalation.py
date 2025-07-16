import torch
from PIL import Image
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer, ViTImageProcessor

from models.clip_model import CLIP
from utils.config import Config

import os
base_dir = os.path.dirname(os.path.abspath(__file__)) 
print("Base directory: ", base_dir)

class CLIPEvaluate:
    def __init__(self, model_path="clip_model.pth"):
        self.config = Config()
        self.device = self.config.DEVICE
        
        self.model = CLIP(self.config.EMBED_DIM).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.TEXT_ENCODER)
        self.image_processor = ViTImageProcessor.from_pretrained(self.config.IMAGE_ENCODER)
    
    def encode_image(self, image_path):
        image = Image.open(image_path)
        pixel_values = self.image_processor(
            images=image, 
            return_tensors="pt"
        )["pixel_values"].to(self.device)
        
        with torch.no_grad():
            image_embedding = self.model.image_encoder(pixel_values)
        
        return image_embedding
    
    def encode_text(self, text):
        inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.config.MAX_LENGTH, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embedding = self.model.text_encoder(
                inputs["input_ids"], 
                inputs["attention_mask"]
            )
        
        return text_embedding
    
    def compute_similarity(self, image_path, texts):
        image_embedding = self.encode_image(image_path)
        
        text_embeddings = []
        for text in texts:
            text_embedding = self.encode_text(text)
            text_embeddings.append(text_embedding)
        
        text_embeddings = torch.cat(text_embeddings, dim=0)
        logit_scale = self.model.logit_scale.exp()
        
        logits_per_image = logit_scale * image_embedding @ text_embeddings.t()
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy().flatten()
        
        return {text: prob for text, prob in zip(texts, probs)}

if __name__ == "__main__":
    clip = CLIPEvaluate()
    
    # 244443352.jpg is a dog
    image_path = os.path.join(base_dir, "data", "flickr30k", "test_img", "snake.jpg")
    print("Image path: ", image_path)
    texts = [
        "a photo of a cat",
        "a picture of a dog",
        "an image of a bird",
        "a photograph of a snake",
    ]
    
    results = clip.compute_similarity(image_path, texts)
    for text, prob in results.items():
        print(f"{text}: {prob:.4f}")
    print("Max probability text: ", max(results, key=results.get))