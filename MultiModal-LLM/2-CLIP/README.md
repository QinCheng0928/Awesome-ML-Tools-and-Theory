## 📁  File Structure

```
2-CLIP/
│
├── [Code] CLIP/ 
│   ├── data/                       
│   │   ├── flickr30k/
│   │   │   ├── annotations/results_20130124.token
│   │   │   └── images/
│   │   │   
│   │   ├── bert-base-uncased/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── pytorch_model.bin
│   │   │   ├── special_tokens_map.json.txt
│   │   │   ├── tokenizer_config.json
│   │   │   └── vocab.txt
│   │   │   
│   │   └── vit-base-patch16-224/
│   │   │   ├── config.json
│   │   │   ├── model.safetensors
│   │   │   ├── preprocessor_config.json
│   │   │   └── pytorch_model.bin
│   │   
│   ├── models/                     
│   │   ├── clip_model.py
│   │   ├── image_encoder.py
│   │   └── text_encoder.py
│   │   
│   ├── utils/                      
│   │   ├── config.py
│   │   └── dataloader.py
│   │   
│   ├── train.py                    # Main training script
│   ├── evaluate.py                 # Evaluation script
│   ├── requirements.txt            # Python dependencies
│   └── gitignore                   # Git ignore rules
│
├── img/ 
│
├── [Paper] Learning-Transferable-Visual-Models-From-Natural-Language-Supervision
│
└── README.md                       # Project documentation
```

## 🔍 Data Preparation Guide

### 1. Download BERT Model (bert-base-uncased)

**1.1 Model Page**: [bert-base-uncased on Hugging Face](https://huggingface.co/bert-base-uncased)

**1.2 Instructions**:

- Click on "Files and versions" tab
- Download the key files from the `data` folder in the File Structure module.
- Create directory: `./data/bert-base-uncased/`
- Place all downloaded files in this directory

### 2. Download ViT Model (vit-base-patch16-224)
**2.1 Model Page**: [vit-base-patch16-224 on Hugging Face](https://huggingface.co/google/vit-base-patch16-224)

**2.2 Instructions**:

- Click on "Files and versions" tab
- Download the key files from the `data` folder in the File Structure module.
- Create directory: `./data/vit-base-patch16-224/`
- Place all downloaded files in this directory

### 3. Download Flickr30K Dataset
**3.1 Official Website**: [Flickr30K Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)

**3.2 Instructions**:

- Download:
   - Image archive (`flickr30k-images.tar.gz`)
   - Annotation file (`flickr30k.tar.gz`)
- Create directory structure:
   ```
   ./data/flickr30k/
   ├── images/
   └── annotations/
   ```
- Extract images to `./data/flickr30k/images/`
- Place annotation file in `./data/flickr30k/annotations/`

**3.3 Example Data**

- **images (1000268201.jpg) :**

![可视化数据集图片](./img/1000268201.jpg)

- **annotations :**

```
1000268201.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
1000268201.jpg#1	A little girl in a pink dress going into a wooden cabin .
1000268201.jpg#2	A little girl climbing the stairs to her playhouse .
1000268201.jpg#3	A little girl climbing into a wooden playhouse .
1000268201.jpg#4	A girl going into a wooden building .
```

