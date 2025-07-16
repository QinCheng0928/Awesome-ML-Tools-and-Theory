import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("The num of available GPUs:", torch.cuda.device_count()) 
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda:0")          
else:
    device = torch.device("cpu") 
