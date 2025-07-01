# Continuous Bag of Words
# CBOW is often used to quickly train word vectors, and the results are used to initialize the embedding of some complex models.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparameters
CONTEXT_SIZE = 2    
EMBEDDING_DIM = 10  

# Sample training sentence
test_sentence = "The quick brown fox jumps over the lazy dog"
tokens = test_sentence.lower().split()

# Prepare training data
data = []
for i in range(1, len(tokens) - 1):
    context = [tokens[i - 1], tokens[i + 1]]  
    target = tokens[i]                        
    data.append((context, target))

# Build vocabulary and mappings
vocab = set(tokens)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.mean(dim=0).view(1, -1)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

if __name__ == "__main__":
    model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_vector = make_context_vector(context, word_to_ix)
            target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_vector)
            loss = loss_function(log_probs, target_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Test prediction: given context, predict center word
    test_context = ['quick', 'fox']
    with torch.no_grad():
        context_vector = make_context_vector(test_context, word_to_ix)
        log_probs = model(context_vector)
        predicted_idx = torch.argmax(log_probs, dim=1).item()
        print(f"Given context {test_context}, predicted center word: {ix_to_word[predicted_idx]}")
    
    '''
        Given context ['quick', 'fox'], predicted center word: 'brown'
    '''