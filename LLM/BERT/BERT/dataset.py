import torch
from torch.utils.data import Dataset
import random

class BertDataset(Dataset):
    def __init__(self, text_pairs, tokenizer, max_seq_length=256, mask_prob=0.15, short_seq_prob=0.1):
        self.text_pairs = text_pairs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.short_seq_prob = short_seq_prob
        
    def __len__(self):
        return len(self.text_pairs)
    
    # Function called when indexing an object
    def __getitem__(self, idx):
        text_a, text_b, is_next = self.text_pairs[idx]
        
        # Tokenization
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b) if text_b else []
        
        # Truncate sequences to fit max length
        self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)  # -3 for [CLS], [SEP], [SEP]
        
        # Build input sequence
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2)
        
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # input_mask: If the value is 1, the token is valid. And if 0, it is padded
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.tokenizer.special_tokens['[PAD]'])
            input_mask.append(0)
            segment_ids.append(0)
        
        # Create masked LM labels
        masked_lm_labels = [-100] * len(input_ids)
        for i in range(len(input_ids)):
            if input_ids[i] in {self.tokenizer.special_tokens['[CLS]'], 
                              self.tokenizer.special_tokens['[SEP]'], 
                              self.tokenizer.special_tokens['[PAD]']}:
                continue
                
            if random.random() < self.mask_prob:
                masked_lm_labels[i] = input_ids[i]
                # 80% chance to replace with [MASK]
                if random.random() < 0.8:
                    input_ids[i] = self.tokenizer.special_tokens['[MASK]']
                # 10% chance to replace with random token
                elif random.random() < 0.5:
                    input_ids[i] = random.randint(5, len(self.tokenizer)-1)
                # 10% chance to keep original
                else:
                    pass
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        masked_lm_labels = torch.tensor(masked_lm_labels, dtype=torch.long)
        next_sentence_label = torch.tensor(1 if is_next else 0, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'masked_lm_labels': masked_lm_labels,
            'next_sentence_label': next_sentence_label
        }
    
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
                
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

# Whether the two sentences appear consecutively in the original text
def create_examples(texts, num_examples=1000):
    examples = []
    for _ in range(num_examples):
        # 50% chance to create positive example
        if random.random() > 0.5:
            i = random.randint(0, len(texts)-2)
            examples.append((texts[i], texts[i+1], True))
        # 50% chance to create negative example
        else:
            i, j = random.sample(range(len(texts)), 2)
            examples.append((texts[i], texts[j], False))
    return examples