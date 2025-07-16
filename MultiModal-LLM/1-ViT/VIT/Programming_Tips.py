import torch
from einops import rearrange
from einops.layers.torch import Rearrange

'''
    1. The usage of torch.meshgrid
'''
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([1, 2, 3, 4])
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
print(grid_x)
print(grid_y)
# tensor([[1, 1, 1, 1],
#         [2, 2, 2, 2],
#         [3, 3, 3, 3],
#         [4, 4, 4, 4]])
# tensor([[1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4]])

grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
print(grid_x)
print(grid_y)
# tensor([[1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4]])
# tensor([[1, 1, 1, 1],
#         [2, 2, 2, 2],
#         [3, 3, 3, 3],
#         [4, 4, 4, 4]])

'''
    2. The usage of closures in Python
'''
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function
# x is fixed at 10
closure = outer_function(10)  
# output 15 (10 + 5)
print(closure(5))  

'''
    3. The usage of einops for tensor manipulation
'''
# Example tensor (batch, height, width, channels)
x = torch.randn(10, 32, 32, 3)

# Reorganize dimensions (convert HWC to CHW)
y = rearrange(x, 'b h w c -> b c h w')
print(y.shape)  
# output: [10, 3, 32, 32]

# Flatten spatial dimensions (merge height and width)
y = rearrange(x, 'b h w c -> b (h w) c')
print(y.shape)  
# output: [10, 1024, 3]

# Split dimensions (split height into 2 sub-dimensions)
y = rearrange(x, 'b (h1 h2) w c -> b h1 h2 w c', h1=16)
print(y.shape)  
# output: [10, 16, 2, 32, 3]

'''
    4. The usage of torch.chunk to split tensors
'''
x = torch.randn(2, 6).chunk(3, dim=-1) 
print("Chunked Tensors:", x[0].shape, x[1].shape, x[2].shape)