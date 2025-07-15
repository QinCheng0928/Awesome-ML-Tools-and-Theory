import torch
from torch.utils.data import DataLoader
from model import BertForPreTraining
from dataset import BertDataset, create_examples
from tokenizer import SimpleTokenizer
from config import config
from utils import format_time
import time

def evaluate():
    # Initialize tokenizer
    tokenizer = SimpleTokenizer('../bert-base-uncased')
    
    # Prepare test data
    test_texts = [
        "Evaluate the performance of this BERT model.",
        "How well does it predict masked words?",
        "Testing is an important part of machine learning.",
        "This is another test sentence.",
        "The model should predict next sentences correctly."
    ]
    
    # Create test examples
    test_examples = create_examples(test_texts, num_examples=100)
    
    # Create test dataset and data loader
    test_dataset = BertDataset(test_examples, tokenizer, config.max_seq_length, config.mask_prob)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Load trained model
    model = BertForPreTraining(config)
    model.load_state_dict(torch.load("models/bert_mlm_nsp.pth", weights_only=True))
    model.to(config.device)
    model.eval()
    
    # Evaluation
    print("Starting evaluation...")
    t0 = time.time()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    mlm_correct = 0
    mlm_total = 0
    nsp_correct = 0
    nsp_total = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            
            # Get predictions
            loss, mlm_loss, nsp_loss = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                masked_lm_labels=batch['masked_lm_labels'],
                next_sentence_label=batch['next_sentence_label']
            )
            
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_nsp_loss += nsp_loss.item()
            
            # Calculate MLM accuracy
            mlm_predictions = torch.argmax(model.cls.predictions(
                model.bert(input_ids=batch['input_ids'],
                          token_type_ids=batch['token_type_ids'],
                          attention_mask=batch['attention_mask'])[0]
            ), dim=-1)
            
            mlm_mask = batch['masked_lm_labels'] != -100
            mlm_correct += (mlm_predictions[mlm_mask] == batch['masked_lm_labels'][mlm_mask]).sum().item()
            mlm_total += mlm_mask.sum().item()
            
            # Calculate NSP accuracy
            nsp_predictions = torch.argmax(model.cls.seq_relationship(
                model.bert(input_ids=batch['input_ids'],
                          token_type_ids=batch['token_type_ids'],
                          attention_mask=batch['attention_mask'])[1]
            ), dim=-1)
            
            nsp_correct += (nsp_predictions == batch['next_sentence_label']).sum().item()
            nsp_total += batch['next_sentence_label'].size(0)
    
    avg_loss = total_loss / len(test_dataloader)
    avg_mlm_loss = total_mlm_loss / len(test_dataloader)
    avg_nsp_loss = total_nsp_loss / len(test_dataloader)
    mlm_accuracy = mlm_correct / mlm_total if mlm_total > 0 else 0
    nsp_accuracy = nsp_correct / nsp_total if nsp_total > 0 else 0
    eval_time = format_time(time.time() - t0)
    
    print("\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average MLM Loss: {avg_mlm_loss:.4f}")
    print(f"Average NSP Loss: {avg_nsp_loss:.4f}")
    print(f"MLM Accuracy: {mlm_accuracy:.4f}")
    print(f"NSP Accuracy: {nsp_accuracy:.4f}")
    print(f"Evaluation took: {eval_time}")
    
    # Example predictions
    print("\nExample Predictions:")
    for i in range(len(test_examples)):
        if test_examples[i][2] == True:
            sample = test_dataset[i]
            break
    
    with torch.no_grad():
        sample_input = {k: v.unsqueeze(0).to(config.device) for k, v in sample.items()}
        sequence_output, pooled_output = model.bert(
            input_ids=sample_input['input_ids'],
            token_type_ids=sample_input['token_type_ids'],
            attention_mask=sample_input['attention_mask']
        )
        prediction_scores, seq_relationship_score = model.cls(sequence_output, pooled_output)
        
    # MLM predictions
    mlm_predictions = torch.argmax(prediction_scores, dim=-1)[0]
    original_tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist())
    predicted_tokens = tokenizer.convert_ids_to_tokens(mlm_predictions.tolist())
    label_tokens = tokenizer.convert_ids_to_tokens(sample['masked_lm_labels'].tolist())
    
    print("\nMasked Language Model Prediction:")
    print("Original:", " ".join(original_tokens))
    print("Labels:  ", " ".join([t if t != "[PAD]" else "." for t in label_tokens]))
    print("Predicted:", " ".join(predicted_tokens))
    
    # NSP prediction
    nsp_prediction = torch.argmax(seq_relationship_score, dim=-1)[0].item()
    actual_label = sample['next_sentence_label'].item()
    print("\nNext Sentence Prediction:")
    print(f"Predicted: {'IsNext' if nsp_prediction else 'NotNext'}")
    print(f"Actual: {'IsNext' if actual_label else 'NotNext'}")

if __name__ == "__main__":
    evaluate()