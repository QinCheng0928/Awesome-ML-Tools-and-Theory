# =====================================
# Implement gradient descent using PyTorch
# =====================================

import numpy as np
import torch

# model prediction
def forward(x, w):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


if __name__ == "__main__":
    # f = w * x
    # f = 2 * x  

    X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
    
    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    
    learning_rate = 0.01
    n_iterations = 100
    
    print("Implement gradient descent using PyTorch.")
    print(f'Prediction before training: f(5) = {forward(5, w):.3f}')

    for epoch in range(n_iterations):
        # prediction
        y_pred = forward(X, w)
        # loss
        l = loss(Y, y_pred)
        # gradient
        l.backward()
        dw = w.grad
        # update weights
        with torch.no_grad():
            w -= learning_rate * dw
        # zero the gradients
        w.grad.zero_()
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.3f}')
    
    print(f'Prediction after training: f(5) = {forward(5, w):.3f}')