import torch
import time

# Set device to GPU if available, otherwise CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Define input tensors on the appropriate device
M = 1_000
N = 1_000

tensor10 = torch.rand(M, N, device=device)
tensor11 = torch.rand(N, M, device=device)

tensor20 = torch.rand(M, N, device=device)
tensor21 = torch.rand(N, M, device=device)

tensor30 = torch.rand(M, N, device=device)
tensor31 = torch.rand(N, M, device=device)

# Measure time for individual matrix multiplications
start_time = time.time()
result1 = torch.mm(tensor10, tensor11)
result2 = torch.mm(tensor20, tensor21)
result3 = torch.mm(tensor30, tensor31)
individual_time = time.time() - start_time

# Measure time for batched matrix multiplication
batched_tensors1 = torch.stack([tensor10, tensor20, tensor30])
batched_tensors2 = torch.stack([tensor11, tensor21, tensor31])

start_time = time.time()
batched_result = torch.bmm(batched_tensors1, batched_tensors2)
batched_time = time.time() - start_time

print(f"Time for individual matrix multiplications: {individual_time:.6f} seconds")
print(f"Time for batched matrix multiplication: {batched_time:.6f} seconds")