import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

class NGameLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(NGameLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)  # input.shape: (2, ), output.shape: (2, embed_dim) or (2, 10), embeds.shape: (1, 20)
        out = F.relu(self.linear1(embeds))            # out.shape: (1, 128)
        out = self.linear2(out)                       # out.shape: (1, vocab_size)
        log_probs = F.log_softmax(out, dim=1)         # log_probs.shape: (1, vocab_size) 
        return log_probs

def predict_next(model, word_to_ix, ix_to_word, context_words):
    context_idxs = torch.tensor([word_to_ix[w] for w in context_words], dtype=torch.long)
    with torch.no_grad():
        log_probs = model(context_idxs)
    predicted = torch.argmax(log_probs, dim=1).item()
    return ix_to_word[predicted]

if __name__ == "__main__":
    test_sentence = "I love natural language processing and I love deep learning"
    tokens = test_sentence.lower().split()
    print(tokens)
    '''
        output:
        ['i', 'love', 'natural', 'language', 'processing', 'and', 'i', 'love', 'deep', 'learning']
    '''

    trigrams = [((tokens[i], tokens[i+1]), tokens[i+2]) 
                for i in range(len(tokens) - 2)]

    vocab = set(tokens)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for word, i in word_to_ix.items()}

    model = NGameLanguageModel(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = []

    # Training loop
    for epoch in range(100):
        total_loss = 0
        for context, target in trigrams:
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)

            model.zero_grad()

            log_probs = model(context_idxs)

            loss = loss_function(log_probs, target_idx)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)

    # Test prediction
    print("Next for ['i','love'] →", predict_next(model, word_to_ix, ix_to_word, ['i', 'love']))
    print("Next for ['love','natural'] →", predict_next(model, word_to_ix, ix_to_word, ['love', 'natural']))
