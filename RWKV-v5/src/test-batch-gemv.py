import torch
import time

# Set device to GPU if available, otherwise CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Define input tensors on the appropriate device
M = 1_000
N = 1_000

tensor1 = torch.rand(M, N, device=device)
tensor2 = torch.rand(M, N, device=device)
tensor3 = torch.rand(M, N, device=device)

# Vector of shape (1K)
vector = torch.rand(1_000, device=device)

# Batched multiplication using torch.bmm
def bmm_multiplication():
    # Stack matrices into shape (3, 4K, 1K)
    matrices = torch.stack([tensor1, tensor2, tensor3])  # Shape: (3, 4K, 1K)
    
    # Reshape vector to (3, 1K, 1) for broadcasting in bmm
    batched_vector = vector.unsqueeze(1).expand(3, -1, 1)  # Shape: (3, 1K, 1)
    
    start_time = time.time()
    # Perform batch matrix multiplication
    result = torch.bmm(matrices, batched_vector)  # Shape: (3, 4K, 1)

    bmm_time = time.time() - start_time
    print(f"bmm_time: {bmm_time}")

    return result.squeeze(2)  # Shape: (3, 4K)

# Non-batched multiplication
def separate_multiplications():
    result1 = tensor1 @ vector  # Shape: (4K)
    result2 = tensor2 @ vector  # Shape: (4K)
    result3 = tensor3 @ vector  # Shape: (4K)
    return torch.stack([result1, result2, result3])  # Shape: (3, 4K)

# Measure time for batched multiplication
start_time = time.time()
bmm_result = bmm_multiplication()
bmm_time = time.time() - start_time

# Measure time for non-batched multiplication
start_time = time.time()
separate_result = separate_multiplications()
separate_time = time.time() - start_time

# Return results and execution times
print(f"Batched multiplication time: {bmm_time} seconds")
print(f"Non-batched multiplication time: {separate_time} seconds")
