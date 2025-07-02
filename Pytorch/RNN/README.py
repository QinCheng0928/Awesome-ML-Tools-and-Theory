"""
Definition:
    nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
    where:
        input_size  : The input feature dimension at each time step (usually the same as embedding_dim)
        hidden_size : Hidden state dimension
        num_layers  : The number of layers in the RNN stack (default is 1)
        batch_first : Whether to take (batch, seq, feature) as input
    
Forward:
    input.shape  : (batch_size, seq_len, input_size)
    output       : 
        out : The output for all time steps.          Shape: (batch_size, seq_len, hidden_size)
        h_n : The hidden state of the last time step. Shape: (num_layers, batch_size, hidden_size)
"""

if __name__ == "__main__":
    print("This is the README document of RNN modol.")
