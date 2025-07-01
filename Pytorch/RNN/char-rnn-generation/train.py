import torch
import torch.nn as nn
import random
import time
import math
from config import *
from model import RNN
from data import all_categories, n_categories, category_lines, line_to_tensor

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def random_training_pair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

def categoryTensor(category):
    idx = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][idx] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i in range(len(line)):
        letter = line[i]
        tensor[i][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_idx = [all_letters.find(line[i]) for i in range(1, len(line))]
    letter_idx.append(n_letters - 1) # add EOS mark
    return torch.tensor(letter_idx, dtype=torch.long)

def randomTrainingExample():
    category, line = random_training_pair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor
    
def time_since(since):
    s = time.time() - since
    return f'{int(s // 60)}m {int(s % 60)}s'

run = RNN(n_letters, n_hidden, n_letters)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(run.parameters(), lr=learning_rate)


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = run.init_hidden()
    
    run.zero_grad()
    
    loss = 0
    
    for i in range(input_line_tensor.size()[0]):
        output, hidden = run(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
        
    loss.backward()
    optimizer.step()
    
    return output, loss.item() / input_line_tensor.size()[0]

if __name__ == '__main__':    
    all_losses = []
    total_loss = 0
    start_time = time.time()
    
    for iter in range(1, n_epochs + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss
        
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (time_since(start_time), iter, iter / n_epochs * 100, loss))
            
        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
    torch.save(run.state_dict(), model_save_path)
    