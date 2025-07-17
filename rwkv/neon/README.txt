# build and isntall 
pip3 install -e .

++++++++++++++++++++++++++++++++++++++++

python3 setup.py build_ext --inplace

# to show raw commands 
# VERBOSE=1 python3 setup.py build_ext --inplace
# or 
python3 setup.py build_ext --inplace --verbose

# install 
python3 setup.py develop

++++++++++++++++++++++++++++++++++++++++

mm8_neon.cpp    the neon kenrels 
setup.py     python wrappers for makign the kernels c++ extension of python. 
    also include compoilation flags for neon.
test-mm8.py   py code that test the neon kernels.

rwkv/neon/matmul_1bit_fp16.c
    1bit weight, gemm code 