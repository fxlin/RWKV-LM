'''
pytorch's tensor interface difficult to extend. thus, only iplement function 
for change a mmap tensor's mapping relation. 

cf test-load-sparse-tensors.py
'''

import sys
import torch
import time 
import random
import mmap
import os
import torch.autograd.profiler as profiler

'''
In Python, you can use the mmap module to map memory, but by default, the module
doesn’t allow you to directly specify a memory address or set flags like
MAP_FIXED. However, you can use the ctypes or ctypes.util module to interface
with the mmap system call at a lower level and pass custom flags.
'''
import ctypes

# Constants (these can vary between systems, check your system headers)
PROT_READ = 0x1
PROT_WRITE = 0x2
MAP_PRIVATE = 0x02
MAP_ANONYMOUS = 0x20
MAP_FIXED = 0x10

# Get libc
libc = ctypes.CDLL("libc.so.6")

# mmap syscall: void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
def custom_mmap(addr, length, prot, flags, fd=-1, offset=0):
    libc.mmap.restype = ctypes.c_void_p  # return type: void*
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    
    result = libc.mmap(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                       ctypes.c_int(prot), ctypes.c_int(flags),
                       ctypes.c_int(fd), ctypes.c_size_t(offset))
    
    if result == -1:
        raise OSError("mmap failed")
    return result

# same as above, but return <Python-buf, raw-buf>. Python buf can be used 
# as torch.from_buffer() input
# mmap syscall: void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
def custom_mmap_python(addr, length, prot, flags, fd=-1, offset=0):
    libc.mmap.restype = ctypes.c_void_p  # return type: void*
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_size_t]
    
    result = libc.mmap(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                       ctypes.c_int(prot), ctypes.c_int(flags),
                       ctypes.c_int(fd), ctypes.c_size_t(offset))
    
    if result == -1:
        raise OSError("mmap failed")
    
    # Create a ctypes type that matches the buffer length and map to the result address
    ctypes_array_type = (ctypes.c_char * length)
    ctypes_array = ctypes_array_type.from_address(result)
    
    # Return a memoryview, which is a buffer-like object
    return memoryview(ctypes_array), result 

###############################################################################

libhelper = ctypes.CDLL('./mmap_helper.so')
# Define the function argument and return types
libhelper.mmap_addresses.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # addresses
    ctypes.c_size_t,                  # num_addresses
    ctypes.c_size_t,                  # length
    ctypes.c_int,                     # prot
    ctypes.c_int,                     # flags
    ctypes.POINTER(ctypes.c_long),    # offsets
    ctypes.c_int                      # fd
]
libhelper.mmap_addresses.restype = ctypes.c_int  # Return int (0 for success, -1 for failure)

# map a list of addresses to a file
# addrlist: list of addresses to map
# size: size of each address (same for all addresses)
# offsetlist: list of offsets within the file
def custom_mmap_batch(addrlist, size, prot, flags, offsetlist, fd):
    # breakpoint()
    # Convert the list of addresses to a ctypes array
    addr_array = (ctypes.c_void_p * len(addrlist))(*addrlist)
    offset_array = (ctypes.c_long * len(offsetlist))(*offsetlist)

    # Call the helper function
    libhelper.mmap_addresses(addr_array, len(addrlist), size, prot, flags, offset_array, fd)

libhelper.munmap_addresses.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # addresses
    ctypes.c_size_t,                  # num_addresses
    ctypes.c_size_t                   # length
]
# return type
libhelper.munmap_addresses.restype = ctypes.c_int  # Return int (0 for success, -1 for failure)

# unmap a list of addresses
# addrlist: list of addresses to unmap
# size: size of each address (same for all addresses)
def custom_unmap_batch(addrlist, size):
    # Convert the list of addresses to a ctypes array
    addr_array = (ctypes.c_void_p * len(addrlist))(*addrlist)

    # Call the helper function
    libhelper.munmap_addresses(addr_array, len(addrlist), size)

###############################################################################

'''
create a big anon mapping, atop it, mmap the tensor file
first mmap the big anon region, then mmap regions from the tensor file 
initially, this is no rows are in core. 

cf load_tensor_with_anonymous_mapping_bench2()
'''
def sparsetensor_init(file_path, tensor_shape, dtype):
    assert len(tensor_shape) == 2   #only 2D
    length = tensor_shape[0] * tensor_shape[1] * torch._utils._element_size(dtype)

    with open(file_path, "r+b") as f:
        prot = PROT_READ | PROT_WRITE
        flags = MAP_PRIVATE | MAP_ANONYMOUS
        # Create one large anonymous mmap area
        anon_mmap_py, anon_mmap = custom_mmap_python(-1, length, prot, flags)

        # Create a tensor from the anonymous mmap area
        mapped_tensor = torch.frombuffer(anon_mmap_py, dtype=dtype, count=tensor_shape[0] * tensor_shape[1])
        mapped_tensor = mapped_tensor.view(tensor_shape)  # Reshape to the original tensor shape

        # mapped_tensor.data_ptr() should be the same as anon_mmap
        return mapped_tensor, anon_mmap

'''
take a tensor, unmap and remap its rows 
old_mask, new_mask: each shape (D,) with 1s and 0s. 1 means in core. D=tensor_shape[0]
do_unmap: if False, do not unmap rows not in new_mask (i.e keep all mapped rows)

return: updated old_mask
cf: load_tensor_with_anonymous_mapping_bench3()
'''
def sparsetensor_remap(tensor, file_path, dtype, old_mask, new_mask, do_unmap=True):
    tensor_shape = tensor.shape
    anon_mmap = tensor.data_ptr()
    row_size_bytes = torch._utils._element_size(dtype) * tensor_shape[1]

    with open(file_path, "r+b") as f:
        prot = PROT_READ
        flags = MAP_PRIVATE | MAP_FIXED

        # Prepare lists for addresses and offsets
        addrlist = []       # mem addr within the tensor
        offsetlist = []     # offsets within the file

        # 1 if new_mask has 1 (to map) on that idx but old_mask has 0 (unmapped)
        tomap_mask = new_mask & ~old_mask
        # 1 if new_mask has 0 (to unmap) on that idx but old_mask has 1 (mapped)
        tounmap_mask = old_mask & ~new_mask

        for i in range(tensor_shape[0]):
            if tomap_mask[i] == 1:
                addrlist.append(anon_mmap + i * row_size_bytes)
                offsetlist.append(i * row_size_bytes)
        custom_mmap_batch(addrlist, row_size_bytes, prot, flags, offsetlist, f.fileno())

        # Unmap the rows that are no longer needed
        if do_unmap:
            addrlist = []
            for i in range(tensor_shape[0]):
                if tounmap_mask[i] == 1:
                    addrlist.append(anon_mmap + i * row_size_bytes)
            custom_unmap_batch(addrlist, row_size_bytes)
            updated_mask = new_mask
        else:
            updated_mask = old_mask | new_mask

    return updated_mask

'''
revert a tensor to its initial state (a contig anon mapping)
'''
def sparsetensor_clearmap(tensor):
    tensor_shape = tensor.shape
    anon_mmap = tensor.data_ptr()
    row_size_bytes = torch._utils._element_size(tensor.dtype) * tensor_shape[1]

    addrlist = []
    for i in range(tensor_shape[0]):
        addrlist.append(anon_mmap + i * row_size_bytes)
    custom_unmap_batch(addrlist, row_size_bytes)


###############################################################################
# test code

def create_random_tensor_and_save(file_path, shape, dtype=torch.float32):
    """
    Creates a tensor with random data and writes it to a binary file.
    
    Args:
    - file_path (str): Path to the file where the tensor will be saved.
    - shape (tuple): Shape of the tensor to create.
    - dtype (torch.dtype): Data type of the tensor (default: torch.float32).
    
    Returns:
    - None
    """
    # Create a tensor with the specified shape, data type, and fill it with a specific value
    specific_value = 1.0  # You can change this value to whatever you need
    random_tensor = torch.full(shape, specific_value, dtype=dtype)
    
    # Open the file in binary write mode and save the raw tensor data
    with open(file_path, 'wb') as f:
        # Write the raw data (tensor as bytes) to the file
        f.write(random_tensor.numpy().tobytes())

    print(f"Tensor with shape {shape} and dtype {dtype} saved to {file_path}")

if __name__ == "__main__":

    D=1024
    # dtype = torch.float32  # Data type of the tensor elements
    dtype = torch.float16  # Data type of the tensor elements
    # dtype = torch.bfloat16  # need some special treatment ... TBD
    tensor_shape = (4*D, D)  # Shape of the 2D tensor (rows, columns)
    file_path = '/tmp/large_tensor_file.bin'  # Path to the tensor file

    # if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    # Create a random tensor and save it to a binary file
    create_random_tensor_and_save(file_path, tensor_shape, dtype=dtype)
    print("created tensor file")

    # Initialize the sparse tensor
    tensor, anon_mmap = sparsetensor_init(file_path, tensor_shape, dtype)
    print(f"tensor: {tensor.shape}, anon_mmap: {hex(anon_mmap)}")

    # Randomly generate a mask for the rows to map
    mask = torch.randint(0, 2, (tensor_shape[0],), dtype=torch.bool)
    print(f"mask: {mask}")

    # Remap the tensor rows according to the mask
    mask = sparsetensor_remap(tensor, file_path, dtype, torch.zeros(tensor_shape[0], dtype=torch.bool), mask)
    print(f"mask: {mask}")

    # Clear the mapping and revert the tensor to its initial state
    sparsetensor_clearmap(tensor)
