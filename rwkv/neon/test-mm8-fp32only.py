import torch
import mm8_neon
import inspect
import time

# below: basically x @ w, x-input, w-weights
#   ry,rx: scaling factors; 
#       rx: liekly applied to input matrix x
#       ry: likely applied to weight matrix w
#   my,mx: biases. (cf above)
#   it's common to scale input & weights separately 
# ex shape: w shape (768,64k) and mx shape (64k) ry shape (768)

# reference impl
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# x shape can be (batch,D)
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

############################################################
# mini test case
N = 2
M = 2
B = 2

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([1.0, 2.0], dtype=torch.float16)
# Input matrix xseq_fp16 of size B x N
xseq_fp16 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.tensor([[10, 20], [30, 40]], dtype=torch.uint8)

# mx_fp16 and my_fp16 of size M and N respectively
mx_fp16 = torch.tensor([0.1, 0.2], dtype=torch.float16)  # size M
rx_fp16 = torch.tensor([0.01, 0.02], dtype=torch.float16)  # size M

my_fp16 = torch.tensor([0.001, 0.002], dtype=torch.float16)  # size N
ry_fp16 = torch.tensor([0.0001, 0.0002], dtype=torch.float16)  # size N

# Expected output yy: tensor([0.3050, 0.6050], dtype=torch.float16)
'''
(Pdb) yy
tensor([0.3030, 0.6064], dtype=torch.float16)
(Pdb) y
tensor([0.3052, 0.6055], dtype=torch.float16)
'''
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

# yy = mm8_neon.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm8_neon.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
# seq
yseq = torch_mm8_seq(xseq_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
# breakpoint()
############################################################
# correctness test mm8_one
N = 10
M = 20

# b (N,M) (786,768*3.5) mx (M) rx (M) my (N,1) ry (N,1)

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.arange(1, N * M + 1, dtype=torch.uint8).reshape(N, M)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 0.2, M, dtype=torch.float16)
rx_fp16 = torch.linspace(0.01, 0.02, M, dtype=torch.float16)

# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.001, 0.002, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.linspace(0.0001, 0.0002, N, dtype=torch.float16).unsqueeze(1)

'''
expected:
tensor([ 5.6016,  5.8906,  6.1797,  6.4688,  6.7617,  7.0508,  7.3438,  7.6367,
         7.9258,  8.2109,  8.5000,  8.7969,  9.0859,  9.3750,  9.6641,  9.9609,
        10.2500, 10.5391, 10.8281, 11.1172], dtype=torch.float16)
'''
# dequant w to fp16, then compute all on fp16
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)

# dequant w to fp32, then compute all on fp32
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

# yy = mm8_neon.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)  # not impl

# neon: dequant w to fp32, then compute all on fp32. with neon
yyy = mm8_neon.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

# breakpoint()

############################################################
# correctness test mm8_one
N = 50
M = 100

# Input vector x_fp16 of size N
x_fp16 = torch.tensor([i + 1.0 for i in range(N)], dtype=torch.float16)

# Weight matrix w_uint8 of size N x M
w_uint8 = torch.arange(1, N * M + 1, dtype=torch.uint8).reshape(N, M)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 1.5, M, dtype=torch.float16)
rx_fp16 = torch.linspace(0.1, 1.5, M, dtype=torch.float16)

# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.1, 1.5, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.linspace(0.1, 1.5, N, dtype=torch.float16).unsqueeze(1)

'''
expected:
tensor([...], dtype=torch.float16)
'''

# cf above

y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

# yy = mm8_neon.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm8_neon.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
# breakpoint()

############################################################
# correctness (test case -- known bad), mm8_one
x_fp16 = torch.tensor([-8.7786e-04, -2.4207e-01, -1.0859e+00,  1.0625e+00, -3.5840e-01,
                       2.9316e+00,  1.3037e+00,  4.5337e-01,  5.3760e-01, -1.4478e-01],
                      dtype=torch.float16)

w_uint8 = torch.tensor([[149, 247, 110,   3,  53, 148,  33, 216,   2,  97],
                        [ 24, 180, 232,  84, 223,  56, 208, 190, 103, 163],
                        [185,  30,  85,  14, 143, 252,  98,  62, 143, 112],
                        [ 19, 185,   4, 176, 115, 195,  32,  87, 132, 202],
                        [ 85,  74, 221, 113, 137,  74, 235, 150,  15, 116],
                        [ 64, 195, 176, 245,  51, 190,  25, 254,  86, 106],
                        [162, 109,  63,  68,  56, 243, 120,  94, 192,   9],
                        [ 80, 121, 178,  22,  23, 148,  35,  56, 153, 103],
                        [ 27,  24, 233,  49,  91,  10, 144, 129,  82, 183],
                        [123,  12, 105, 143, 236, 201, 138, 253, 161,  85]],
                       dtype=torch.uint8)

mx_fp16 = torch.tensor([0.1000, 0.2000, 0.2998, 0.3999, 0.5000, 0.6001, 0.7002, 0.7998, 0.8999,
                        1.0000], dtype=torch.float16)

rx_fp16 = torch.tensor([0.3914, 0.0447, 0.1252, 0.2034, 0.3882, 0.1665, 0.1735, 0.4670, 0.3103,
                        0.3884], dtype=torch.float16)

my_fp16 = torch.tensor([[0.0010],
                        [0.0011],
                        [0.0012],
                        [0.0013],
                        [0.0014],
                        [0.0016],
                        [0.0017],
                        [0.0018],
                        [0.0019],
                        [0.0020]], dtype=torch.float16)

ry_fp16 = torch.tensor([[0.0581],
                        [0.3372],
                        [0.1445],
                        [0.2220],
                        [0.0409],
                        [0.4751],
                        [0.3643],
                        [0.0629],
                        [0.0181],
                        [0.2498]], dtype=torch.float16)

y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
y1 = torch_mm8_one(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)

# yy = mm8_neon.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16, 1)
yyy = mm8_neon.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
print(">here")
# breakpoint()

############################################################
# time benchmark ... mm8_one
# x (N) w (N,M) (768,768*3.5)  mx (M) rx (M) my (N,1) ry (N,1)

# Example data
# N = 1024
# M = int(N * 3.5)

# N = 768
# M = int(N * 3.5)

# cls head 
N = 768
M = 65536


# N = 768
# M = 768

x_fp16 = torch.randn(N, dtype=torch.float16)
w_uint8 = torch.randint(0, 256, (N, M), dtype=torch.uint8)
# mx_fp16 = torch.randn(M, dtype=torch.float16)
# rx_fp16 = torch.randn(M, dtype=torch.float16)
# my_fp16 = torch.randn((N,1), dtype=torch.float16)
# ry_fp16 = torch.randn((N,1), dtype=torch.float16)

# mx_fp16 and rx_fp16 of size M
mx_fp16 = torch.linspace(0.1, 1.0, M, dtype=torch.float16)
rx_fp16 = torch.rand(M, dtype=torch.float16) * 0.49 + 0.01
# my_fp16 and ry_fp16 of size Nx1
my_fp16 = torch.linspace(0.001, 0.002, N, dtype=torch.float16).unsqueeze(1)
ry_fp16 = torch.rand((N,1), dtype=torch.float16) * 0.49 + 0.01


# y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)

# y1 = torch_mm8_one(
#     x_fp16.to(torch.float), 
#     w_uint8, 
#     mx_fp16.to(torch.float), 
#     rx_fp16.to(torch.float), 
#     my_fp16.to(torch.float), 
#     ry_fp16.to(torch.float)
# )

# yy = mm8_neon.mm_one_fp16i8(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
# yyy = mm8_neon.mm_one_fp32i8(
#     x_fp16.to(torch.float), 
#     w_uint8, 
#     mx_fp16.to(torch.float), 
#     rx_fp16.to(torch.float), 
#     my_fp16.to(torch.float), 
#     ry_fp16.to(torch.float)
# )
# print(f"y: {y[:10]}")
# print(f"yy: {yy[:10]}")
# print(f"yyy: {yyy[:10]}")

# breakpoint()


print(f"N = {N}, M = {M}")
# Measure execution time for torch_mm8_one
start_time = time.time()
y = torch_mm8_one(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
end_time = time.time()
print(f"Execution time for torch_mm8_one: {(end_time - start_time) * 1000:.3f} ms")

# fp32
start_time = time.time()
yyy = mm8_neon.mm_one_fp32i8(
    x_fp16.to(torch.float), 
    w_uint8, 
    mx_fp16.to(torch.float), 
    rx_fp16.to(torch.float), 
    my_fp16.to(torch.float), 
    ry_fp16.to(torch.float)
)
end_time = time.time()
print(f"Execution time for mm_one_fp32i8: {(end_time - start_time) * 1000:.3f} ms")

# all fp16, no quant
# Generate random tensors x and w
x = torch.randn(N, dtype=torch.float16)
w = torch.randn(N, M, dtype=torch.float16)
start_time = time.time()
result = x @ w
end_time = time.time()
print(f">>> Execution time for fp16 noquant: {(end_time - start_time) * 1000:.3f} ms")
print("\n")

if False:
    print(f"torch y: {y[:10]}")
    print(f"fp16i8 v1 yy1: {yy1[:10]}")
    print(f"fp16i8 v2 yy2: {yy2[:10]}")
    print(f"fp16i8 v3 yy3: {yy3[:10]}")
    print(f"fp32i8 yyy: {yyy[:10]}")


y_torch_f32 = y.to(torch.float32)

# Find the maximum relative difference

# Find the index of the item in y_torch_f32 that results in the maximum difference

# Print the values at max_diff_index

# breakpoint()



############################################################
#       correctness check for mm8_seq
# Prepare test inputs
# B = 16
B=4
N = 1024
M = int(N * 3.5)

print(f"B={B}, N = {N}, M = {M}")

x_fp16 = torch.randn(B, N, dtype=torch.float16)
w_uint8 = torch.randint(0, 256, (N, M), dtype=torch.uint8)
mx_fp16 = torch.randn(M, dtype=torch.float16)
rx_fp16 = torch.randn(M, dtype=torch.float16)
my_fp16 = torch.randn(N, 1, dtype=torch.float16)
ry_fp16 = torch.randn(N, 1, dtype=torch.float16)

# Measure execution time for torch_mm8_seq
start_time = time.time()
y_torch = torch_mm8_seq(x_fp16, w_uint8, mx_fp16, rx_fp16, my_fp16, ry_fp16)
end_time = time.time()
print(f"Execution time for torch_mm8_seq: {(end_time - start_time) * 1000:.3f} ms")

# Measure execution time for mm_seq_fp32i8
start_time = time.time()
y_cpp32 = mm8_neon.mm_seq_fp32i8(
    x_fp16.to(torch.float),
    w_uint8,
    mx_fp16.to(torch.float),
    rx_fp16.to(torch.float),
    my_fp16.to(torch.float),
    ry_fp16.to(torch.float)
)
end_time = time.time()
print(f"Execution time for mm_seq_fp32i8: {(end_time - start_time) * 1000:.3f} ms")

# Measure time for fp16 no quant (problematic?? so slow
# w = torch.randn(N, M, dtype=torch.float16)
w = w_uint8.to(torch.float16)
start_time = time.time()
result = x_fp16 @ w
end_time = time.time()
print(f">>> Execution time for fp16 noquant: {(end_time - start_time) * 1000:.3f} ms")
print("\n")

y_torch_f32 = y_torch.to(torch.float32)
# Compute the relative differences between y_cpp32 and y_torch_f32
relative_differences_cpp32 = torch.abs(y_torch_f32 - y_cpp32) / torch.abs(y_torch_f32)

# Find the maximum relative difference
max_relative_difference_cpp32 = torch.max(relative_differences_cpp32).item()
print(f"Maximum Relative Difference (y_cpp32 vs y_torch_f32): {max_relative_difference_cpp32}")

# # Get the top 5 values with the highest relative differences
# top_k_cpp32 = 5
# top_k_indices_cpp32 = torch.topk(relative_differences_cpp32.view(-1), top_k_cpp32).indices

# print(f"Top {top_k_cpp32} values with the highest relative differences (y_cpp32 vs y_torch_f32):")
# for idx in top_k_indices_cpp32:
#     print(f"Index: {idx.item()}, Relative Difference: {relative_differences_cpp32.view(-1)[idx].item()}")
#     print(f"Value in y_torch_f32: {y_torch_f32.view(-1)[idx].item()}")
#     print(f"Value in y_cpp32: {y_cpp32.view(-1)[idx].item()}")

# # Find the index of the item in y_torch_f32 that results in the maximum difference
max_diff_index_cpp32 = torch.argmax(relative_differences_cpp32).item()
print(f"Index of the item with the maximum difference (y_cpp32 vs y_torch_f32): {max_diff_index_cpp32}")

# # Print the values at max_diff_index
print(f"Value in y_torch_f32 at max_diff_index: {y_torch_f32.view(-1)[max_diff_index_cpp32]}")
print(f"Value in y_cpp32 at max_diff_index: {y_cpp32.view(-1)[max_diff_index_cpp32]}")

# print(f"torch y: {y_torch[:10]}")
# print(f"fp16i8 : {y_cpp[:10]}")

# Compare the outputs
print(torch.allclose(y_torch.to(torch.float32), y_cpp32, rtol=1e-1))
# breakpoint()

'''
---------------------------------------------
orpi zero 2w
---------------------------------------------
(myenv) root@orangepizero2w:/home/orangepi/workspace-rwkv/RWKV-LM/rwkv/neon# python3 test-mm8-fp32only.py
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
>here
N = 768, M = 2688
Execution time for torch_mm8_one: 41.178 ms
Execution time for mm_one_fp32i8: 5.436 ms
>>> Execution time for fp16 noquant: 6.371 ms


B=4, N = 1024, M = 3584
Execution time for torch_mm8_seq: 521.603 ms
Execution time for mm_seq_fp32i8: 54.305 ms
>>> Execution time for fp16 noquant: 467.376 ms

N = 768, M = 65536 (v3) 
Execution time for torch_mm8_one: 1365.946 ms
Execution time for mm_one_fp32i8: 337.632 ms <<<< still slow???
>>> Execution time for fp16 noquant: 180.270 ms

---------------------------------------------
rpi5, 8GB. cortexa76 has fp16 native support 
---------------------------------------------

N = 1024, M = 3584
    mm_one_fp32i8 V1
        Execution time for torch_mm8_one: 27.992 ms
        Execution time for mm_one_fp32i8: 3.611 ms
        >>> Execution time for fp16 noquant: 1.786 ms    
        ^^^^ ONLY CLEAR FOR LARGER INPUT ??
    mm_one_fp32i8 V3
        Execution time for torch_mm8_one: 28.183 ms
        Execution time for mm_one_fp32i8: 1.819 ms  <<<< fixed, great
        >>> Execution time for fp16 noquant: 1.770 ms

N = 768, M = 2688
    mm_one_fp32i8 V1
        Execution time for torch_mm8_one: 16.009 ms
        Execution time for mm_one_fp32i8: 1.412 ms
        >>> Execution time for fp16 noquant: 1.449 ms
        ^^^^ GAP IS SMALL 
    mm_one_fp32i8 V3
        Execution time for torch_mm8_one: 15.174 ms
        Execution time for mm_one_fp32i8: 1.198 ms  <<<< fixed, great
        >>> Execution time for fp16 noquant: 0.913 ms

N = 1024, M = 65536

    mm_one_fp32i8 V3
        Execution time for torch_mm8_one: 561.809 ms
        Execution time for mm_one_fp32i8: 29.913 ms <<<<<<< fixed, great
        >>> Execution time for fp16 noquant: 26.757 ms        

N = 768, M = 65536
    Execution time for torch_mm8_one: 415.750 ms
    Execution time for mm_one_fp32i8: 28.011 ms
    >>> Execution time for fp16 noquant: 19.179 ms

B=4
N = 1024
M = int(N * 3.5)
'''    