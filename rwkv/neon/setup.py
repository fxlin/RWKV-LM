from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

import subprocess

mcpu = None
march = None

extra_compile_args=[
                '-O3',
                '-g',
                '-std=c++17',
                '-fopenmp'    # Enable OpenMP
            ]

# Run 'lscpu' command to check for the CPU model
cpu_info = subprocess.check_output("lscpu", shell=True).decode()
if "A53" in cpu_info:   # opi0
    extra_compile_args.append("-mcpu=cortex-a53")
    extra_compile_args.append("-march=armv8-a")
elif "A76" in cpu_info: # rpi5
    extra_compile_args.append("-mcpu=cortex-a76")
    extra_compile_args.append("-march=armv8.2-a+fp16")
    extra_compile_args.append("-DHAS_NEON_FP16")
elif "A72" in cpu_info:  # rpi4
    extra_compile_args.append("-mcpu=cortex-a72")
    extra_compile_args.append("-march=armv8-a")
else: 
    print(f"Could not detect CPU model: exit")
    sys.exit(1)
    
setup(
    name='mm8_neon',
    ext_modules=[
        CppExtension(
            'mm8_neon',
            ['mm8_neon.cpp'],
            extra_compile_args=extra_compile_args,
            extra_link_args=['-fopenmp'],  # Link against OpenMP library
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
