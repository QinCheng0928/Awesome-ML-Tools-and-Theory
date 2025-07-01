# config.py
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1
# EOS mark

# Training parameters
n_hidden = 128
n_epochs = 100_000
print_every = 5_000
plot_every = 1_000
learning_rate = 0.0005

# Paths
model_save_path = "./char-rnn-generation/generation_model.pt"
data_path = "data/names/*.txt"
