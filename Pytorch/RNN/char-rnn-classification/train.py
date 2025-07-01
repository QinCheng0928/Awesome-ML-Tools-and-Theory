# train.py
import time
import math
import random
import torch
import torch.nn as nn
from data import all_categories, category_lines, line_to_tensor
from model import RNN
from config import *

def category_from_output(output):
    _, top_i = output.topk(1, dim=1)
    return all_categories[top_i[0][0]], top_i[0][0]

def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

def train(model, criterion, optimizer, category_tensor, line_tensor):
    hidden = model.init_hidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        # line_tensor[i] is a one-hot encoding.
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

def time_since(since):
    s = time.time() - since
    return f'{int(s // 60)}m {int(s % 60)}s'

def main():
    rnn = RNN(n_letters, n_hidden, len(all_categories))
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    all_losses = []
    current_loss = 0
    start = time.time()

    for epoch in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = random_training_pair()
        output, loss = train(rnn, criterion, optimizer, category_tensor, line_tensor)
        current_loss += loss

        if epoch % print_every == 0:
            guess, _ = category_from_output(output)
            correct = "✓" if guess == category else f"✗ ({category})"
            print(f"{epoch} {epoch / n_epochs * 100:.2f}% ({time_since(start)}) "
                  f"{loss:.4f} {line} / {guess} {correct}")

        if epoch % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    torch.save(rnn.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()
