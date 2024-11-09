# these need: 
# sudo apt install libblas-dev
# sudo apt install libopenblas-dev
# sudo apt-get install libexplain-dev
gcc -o sgemv_example sgemv_example.c -lopenblas -O2 -g
# -rt for high precision timer
gcc -o test-mmap-overlay test-mmap-overlay.c -lrt  -O2 -g
# ./mmap_test <file_path> <N>

gcc -o test-unmap test-unmap.c -O2 -g
gcc -o test-unmap-bench test-unmap-bench.c -O2 -g

gcc -o test-madvise test-madvise.c -O2 -g

# gcc -shared -o mmap_helper.so -fPIC mmap_helper.c -O2 -g



gcc -shared -o mmap_helper.so -fPIC mmap_helper.c -O2 -g -lexplain