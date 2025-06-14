# Implement all using PyTorch
# 1 ) Design the model (input, output, forward pass)
# 2 ) Define the loss and optimizer
# 3 ) Training loop
#     - forward pass: compute the prediction
#     - backward pass: compute the gradients
#     - update the weights

import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    # y1 = x1 + 1 * x2 + 1
    # y2 = x1 + 2 * x2 + 2
    # y3 = x1 + 3 * x2 + 3
    # 1. Data
    X = torch.tensor([[1, 1], [2, 3], [3, 5], [4, 7]], dtype=torch.float32)
    Y = torch.tensor([[3, 5, 7], [6, 10, 14], [9, 15, 21], [12, 20, 28]], dtype=torch.float32)   
    
    # 2. Parameters
    input_size = 2
    output_size = 3
    learning_rate = 0.01
    n_iterations = 5000    

    # 3. Model
    model = nn.Linear(input_size, output_size)
    
    # 4. Loss and optimizer
    # For Linear regresson, the parameters are usually weight and bias.
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    '''
        Name: weight
        Value:
            Parameter containing:
            tensor([[ 0.5973,  0.0950],
                    [ 0.5307,  0.1740],
                    [ 0.2321, -0.3062]], requires_grad=True)
            Shape: torch.Size([3, 2])

        Name: bias
        Value:
            Parameter containing:
            tensor([0.4985, 0.6172, 0.1896], requires_grad=True)
            Shape: torch.Size([3])
    '''
    for name, param in model.named_parameters():
        print(f"Name: {name}")
        print(f"Value:\n{param}")
        print(f"Shape: {param.shape}\n")
    
    # 5. Before training
    print(f'Prediction before training: f(5) = {model(torch.tensor([[5.0, 9.0]]))}')

    # 6. Training loop
    for epoch in range(n_iterations):
        y_pred = model(X)
        l = loss(y_pred, Y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        # The shape of the weight is [output_size, input_size]
        print(f'Epoch {epoch + 1}: loss = {l.item():.3f}')

    # 7. After training
    print("Model weright:", model.weight)
    print("Bias:", model.bias) 
    '''
        Model weright: Parameter containing:
            tensor([[1.3196, 0.8403],
                    [1.6509, 1.6746],
                    [2.9375, 2.0313]], requires_grad=True)
        Bias: Parameter containing:
            tensor([0.8400, 1.6741, 2.0308], requires_grad=True)
    '''
    print(f'Prediction after training: f(5) = {model(torch.tensor([[5.0, 9.0]]))}')