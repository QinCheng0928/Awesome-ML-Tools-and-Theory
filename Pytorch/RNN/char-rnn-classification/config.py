# config.py
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

# Training parameters
n_hidden = 128
n_epochs = 100_000
print_every = 5_000
plot_every = 1_000
learning_rate = 0.005

# Paths
model_save_path = "./char-rnn-classification/classification_model.pt"
data_path = "data/names/*.txt"
