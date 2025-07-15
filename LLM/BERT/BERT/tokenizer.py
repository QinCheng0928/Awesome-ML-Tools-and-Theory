from transformers import BertTokenizer

class SimpleTokenizer:
    def __init__(self, pretrained_path='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)
        print(f"len of vocab: {len(self.tokenizer)}")
        self.special_tokens = {
            '[PAD]': self.tokenizer.pad_token_id,
            '[UNK]': self.tokenizer.unk_token_id,
            '[CLS]': self.tokenizer.cls_token_id,
            '[SEP]': self.tokenizer.sep_token_id,
            '[MASK]': self.tokenizer.mask_token_id
        }
    
    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def __len__(self):
        return len(self.tokenizer)
