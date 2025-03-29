import torch

print("Torch version:", torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")

# Simple operation
x = torch.randn(100, 100).to(device)
y = torch.matmul(x, x).sum()
print("Result:", y.item())
