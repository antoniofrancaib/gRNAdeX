import torch

try:
    data = torch.load('/rds/user/ml2169/hpc-work/geometric-rna-design/data/processed.pt')
    print("File loaded successfully")
except Exception as e:
    print(f"Error: {e}")