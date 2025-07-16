from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor

class CLIPDataset(Dataset):
    """CLIP dataset for image-text pairs."""
    def __init__(self, image_paths, texts, max_length=32):
        self.image_paths = image_paths
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained("./data/bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained("./data/vit-base-patch16-224")
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # process image
        image = Image.open(self.image_paths[idx])
        image = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        
        # process text
        text = self.texts[idx]
        text_inputs = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs["input_ids"][0],
            "attention_mask": text_inputs["attention_mask"][0]
        }