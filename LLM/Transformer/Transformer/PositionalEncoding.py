import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        # arange(): Generate a sequence from 0 to max_len with a step size of 1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Operate on all rows and on columns starting from 0 with a step size of 2.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Similar to the above
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.shape: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # The parameter values can be saved to the state_dict, but they cannot be trained.
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
    
if __name__ == '__main__':
    '''
        Broadcast Mechanism:
            Before broadcasting
                position =[                  
                    [0],
                    [1],
                    [2],
                    [3],
                    [4],
                    [5]
                ]
                div_term = [0, 2, 4]
            After broadcasting
                position =[                  
                    [0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5]
                ]
                div_term = [
                    [0, 2, 4],
                    [0, 2, 4],
                    [0, 2, 4],
                    [0, 2, 4],
                    [0, 2, 4],
                    [0, 2, 4],
                ]
        The result of position * div_term is:
            print(position * div_term)
            tensor([[ 0,  0,  0],
                [ 0,  2,  4],
                [ 0,  4,  8],
                [ 0,  6, 12],
                [ 0,  8, 16],
                [ 0, 10, 20]])
        
    '''
    position = torch.arange(0, 6).unsqueeze(1)
    div_term = torch.arange(0, 6, 2)
    print(position * div_term)
    