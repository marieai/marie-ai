import torch

print("PyTorch version:", torch.__version__)

# Test basic tensor operation
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = x + y
print("Tensor operation successful:\n", z)

# Check for CUDA (GPU) support
if torch.cuda.is_available():
    print("CUDA is available!")
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is NOT available.")
