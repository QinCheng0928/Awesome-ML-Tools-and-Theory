# predict.py

# python char-rnn-classification/predict.py "QinCheng"
# output: 
#   (-0.44) names\Chinese
#   (-1.60) names\Irish
#   (-3.02) names\Korean
import torch
import sys
from model import RNN
from data import all_categories, line_to_tensor
from config import model_save_path, n_hidden, n_letters

def load_model():
    rnn = RNN(n_letters, n_hidden, len(all_categories))
    state_dict = torch.load(model_save_path, weights_only=True)
    rnn.load_state_dict(state_dict)
    rnn.eval()
    return rnn

def evaluate(model, line_tensor):
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)
    return output

def predict(model, line: str, n_predictions=3):
    with torch.no_grad():
        line_tensor = line_to_tensor(line)
        output = evaluate(model, line_tensor)

        topv, topi = output.topk(n_predictions, dim=1)
        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print(f"({value:.2f}) {all_categories[category_index]}")

if __name__ == '__main__':
    model = load_model()
    name = sys.argv[1] if len(sys.argv) > 1 else input("Enter a name: ")
    predict(model, name)
