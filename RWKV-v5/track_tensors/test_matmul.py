import torch
from torch.profiler import profile, ProfilerActivity

@torch.jit.script
def my_func(a, b):
    #return torch.matmula(a, b)
    matmul_sparse()
    return a @ b

a = torch.randn(5, 768)
b = torch.randn(768, 768)
#traced = torch.jit.trace(my_func, (a, b))
#print(traced.graph)

#with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#c = a @ b
c = my_func(a, b)
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

#c = my_func(a, b)

#print(traced.graph)
#c = traced(a, b)
