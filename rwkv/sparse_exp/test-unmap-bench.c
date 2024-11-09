// like test-unmap.c, but focusing on test memory consumption, speed, ertc. 
// therefore, deal with larger fiels, and dont print out contents
// cf the EOF for output 

// this shows that as soon as overlapped anonymous mapping is created, the pages are freed immmediately. 
// see EOF 

// dd if=/dev/random of=/tmp/existing_file.bin bs=1M count=16

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

const char *file_path = "/tmp/existing_file.bin";

void print_firstbytes(char *map) {
#ifdef DEBUG 
    // Print the first 32 bytes for each of the first 4 pages
    for (int i = 0; i < 4; i++) {
        printf("Page %d: ", i);
        for (int j = 0; j < 32; j++) {
            printf("%02x ", ((unsigned char *)map)[i * 4096 + j]);
        }
        printf("....\n");
    }
#endif    
}

// return the number of pages in the current process's memory
int get_curret_mem_pages() {
    FILE *file = fopen("/proc/self/statm", "r");
    if (!file) {
        perror("Error opening /proc/self/statm");
        return -1;
    }

    int mem_usage;
    if (fscanf(file, "%*s %d", &mem_usage) != 1) {  // Read the 2nd field.. rss??
        perror("Error reading memory usage");
        fclose(file);
        return -1;
    }
    fclose(file);
    // return mem_usage * getpagesize(); // Convert pages to bytes
    return mem_usage; 
}

int main(int argc, char *argv[]) {

    int is_unmap = 1; 
    
    int N;    // # of anonymous mappings to create

    int fd;
    struct stat sb;
    void *map;
    size_t file_size;
    struct timespec start, end;

    // Open the file
    fd = open(file_path, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get the file size
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }
    file_size = sb.st_size;

    printf("before map the file. current pages: %d\n", get_curret_mem_pages());

    // Memory-map the file
    map = mmap(NULL, file_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    printf("map the whole file %s. total %lu pages \n", file_path, (file_size + 4095)/4096);
    printf("after map the file. current pages: %d\n", get_curret_mem_pages());

    N = (file_size + 4095)/4096/2; //50% of pages

    // Touch the 1st byte of every page
    for (size_t i = 0; i < file_size; i += 4096) {
        // ((char *)map)[i] = ((char *)map)[i];
        ((char *)map)[i] ++; 
    }
    printf("after touch all mem \n");
    printf("current pages: %d\n", get_curret_mem_pages());
    
    // Create N anonymous mappings atop the file mmap region... everey other page 
    N = (N < file_size / 4096 / 2) ? N : file_size / 4096 / 2;  // Limit N to the number of 4KB pages in the file

    for (int i = 0; i < N; i++) {
        void *new_mapping = mmap((char *)map + i * 2 * 4096, 4096, PROT_READ,
                                 MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_mapping == MAP_FAILED) {
            perror("anonymous mmap failed");
            munmap(map, file_size);
            close(fd);
            exit(EXIT_FAILURE);
        }
        // printf("map page %d to anonymous\n", i*2);
    }

    printf("after create anonymous regions\n");
    printf("current pages: %d\n", get_curret_mem_pages());

    if (is_unmap) {
        // unmappe N page 
        // this deletes both the file and the anonymous mappings :(
        for (int i = 0; i < N; i++) {

            // before unmap, we advise the kernel to free the page... (needed???
            if (madvise((char *)map + i * 2 * 4096, 4096, MADV_DONTNEED) == -1) {
                perror("Error with madvise");
                munmap(map, file_size);
                close(fd);
                return 1;
            } // else printf("advise ok on page %d\n", i*2);

            // printf("unmap page %d\n", i*2);
            if (munmap((char *)map + i * 2 * 4096, 4096) == -1)
                perror("Error unmapping the file");
        }
        printf("after unmap...(+MADV_DONTNEED)\n");
        printf("current pages: %d\n", get_curret_mem_pages());
    }    

    close(fd);
}



/*

// this shows that as soon as overlapped anonymous mapping is created, the pages are freed immmediately. 

Ubuntu 22.04, rpi4
Linux rpi4 5.15.98-rt62-raspi #1 SMP PREEMPT_RT Sun May 7 10:39:42 UTC 2023 aarch64 aarch64 aarch64 GNU/Linux

dd if=/dev/random of=/tmp/existing_file.bin bs=1M count=16

(myenv) robot@rpi4:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-unmap-bench --unmap
before map the file. current pages: 194   (^^^ could be shared libs, etc 
map the whole file /tmp/existing_file.bin. total 4096 pages 
after map the file. current pages: 194
after touch all mem 
current pages: 4433
after create anonymous regions
current pages: 2385
after unmap...(+MADV_DONTNEED)
current pages: 2385

also rpi5, ubuntu 24

xl6yq@rpi5:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-unmap-bench
before map the file. current pages: 288
map the whole file /tmp/existing_file.bin. total 4096 pages
after map the file. current pages: 320
after touch all mem
current pages: 4416
after create anonymous regions
current pages: 2400
after unmap...(+MADV_DONTNEED)
current pages: 2400

*/

