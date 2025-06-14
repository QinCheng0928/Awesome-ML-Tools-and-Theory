# =====================================
# Implement gradient descent manually
# =====================================

import numpy as np

# model prediction
def forward(x, w):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# gradient
# MSE = 1/N * (w * x - y)**2
# dL/dw = 1/N * 2 * x * (w * x - y)
# dL/dw = 1/N * 2 * x * (y_pred - y)
def gradient(x, y, y_pred):
    return (2 * x * (y_pred - y)).mean()

if __name__ == "__main__":
    # f = w * x
    # f = 2 * x  

    X = np.array([1, 2, 3, 4], dtype=np.float32)
    Y = np.array([2, 4, 6, 8], dtype=np.float32)

    w = 0.0
    
    learning_rate = 0.01
    n_iterations = 100
    
    print("Implement gradient descent manually.")
    print(f'Prediction before training: f(5) = {forward(5, w):.3f}')

    for epoch in range(n_iterations):
        # prediction
        y_pred = forward(X, w)
        # loss
        l = loss(Y, y_pred)
        # gradient
        dw = gradient(X, Y, y_pred)
        # update weights
        w -= learning_rate * dw
        print(f'Epoch {epoch + 1}: w = {w:.3f}, loss = {l:.3f}')
    
    print(f'Prediction after training: f(5) = {forward(5, w):.3f}')