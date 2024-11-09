// test the unmap behaivor() on overlapped mappings
// showing that unmap() will delete both overlapped  mappings (file and the anonymous) at the given addr
// cf the EOF for output 


// dd if=/dev/random of=/tmp/existing_file.bin bs=4K count=4


#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

const char *file_path = "/tmp/existing_file.bin";

int main(int argc, char *argv[]) {

    int is_patch = 0;
    int is_unmap = 0; 
    if (argc > 1 && strcmp(argv[1], "--patch-anonymous") == 0) {
        is_patch = 1;
    }
    if (argc > 1 && strcmp(argv[1], "--unmap") == 0) {
        is_unmap = 1;
    }

    int N = 128;    // # of anonymous mappings to create

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

    // Memory-map the file
    map = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        exit(EXIT_FAILURE);
    }
    printf("map the whole file. total %lu pages \n", (file_size + 4095)/4096);

    printf("after map\n");
    // Print the first 32 bytes for each of the first 4 pages
    for (int i = 0; i < 4; i++) {
        printf("Page %d: ", i);
        for (int j = 0; j < 32; j++) {
            printf("%02x ", ((unsigned char *)map)[i * 4096 + j]);
        }
        printf("....\n");
    }    

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
        printf("map page %d to anonymous\n", i*2);
    }

    printf("after map\n");
    for (int i = 0; i < 4; i++) {
        printf("Page %d: ", i);
        for (int j = 0; j < 32; j++) {
            printf("%02x ", ((unsigned char *)map)[i * 4096 + j]);
        }
        printf("....\n");
    }

    if (is_unmap) {
        // unmappe the first N page 
        // this deletes both the file and the anonymous mappings :(
        N=2;    
        for (int i = 0; i < N; i++) {
            // before unmap, we advise the kernel to free the page... (needed???
            if (madvise((char *)map + i * 2 * 4096, 4096, MADV_DONTNEED) == -1) {
                perror("Error with madvise");
                munmap(map, file_size);
                close(fd);
                return 1;
            } else printf("advise ok on page %d\n", i*2);

            printf("unmap page %d\n", i*2);
            if (munmap((char *)map + i * 2 * 4096, 4096) == -1)
                perror("Error unmapping the file");
        }

        printf("after unmap...\n");
    }

    int i = 1;      // this shows that the unmmaped pages are still good... showing file contents
    printf("touch page %d\n", i);
    printf("Page %d: ", i);
    for (int j = 0; j < 32; j++) {
        printf("%02x ", ((unsigned char *)map)[i * 4096 + j]);
    }
    printf("\n");

    // next: will touch an unmapped page. 
    // without "patch", this will segfault. i.e. unmap above deletes both the file and the anonymous mappings
    i = 0; 

    // here, we show "patch in" an anonymous mapping here, to avoid the segfault
    if (is_patch) { 
        printf(" create anonymous mapping for page %d ... \n", i);
        void *new_mapping = mmap((char *)map + i * 2 * 4096, 4096, PROT_READ,
                                 MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_mapping == MAP_FAILED) {
            perror("anonymous mmap failed");
            munmap(map, file_size);
            close(fd);
            exit(EXIT_FAILURE);
        }
        printf("map page %d to anonymous\n", i*2);
    }

    printf("touch page %d\n", i);
    printf("Page %d: ", i);
    for (int j = 0; j < 32; j++) {
        printf("%02x ", ((unsigned char *)map)[i * 4096 + j]);
    }
    printf("\n");

    close(fd);
}


/* 


(myenv) robot@rpi4:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-unmap 
map the whole file. total 4 pages 
after map
Page 0: 62 1c b8 0d 75 35 6b f7 1f 09 b8 ed 9a 20 59 d1 8a 2a 66 91 a1 b7 5b 63 de dd 64 8b d6 a4 59 93 ....
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 ....
Page 2: 9b e3 44 0f 55 24 cc 90 a1 92 ac 7e f7 a9 af 24 1e 9f a1 ca 28 8c 2d 76 28 6e 77 ae cb 76 29 34 ....
Page 3: bd 6a 65 b3 68 08 8e 47 7c 1d 9a ef 38 d3 0a cf 3b 2d be a2 49 3b 8e 6a f0 d6 6e 05 40 b2 33 81 ....
map page 0 to anonymous
map page 2 to anonymous
after map
Page 0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ....
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 ....
Page 2: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ....
Page 3: bd 6a 65 b3 68 08 8e 47 7c 1d 9a ef 38 d3 0a cf 3b 2d be a2 49 3b 8e 6a f0 d6 6e 05 40 b2 33 81 ....
unmap page 0
unmap page 2
after unmap...
touch page 1
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 
^^^^  page 1 still good (the file mappings)
touch page 0
Segmentation fault (core dumped)


 can be patched in by adding the --patch-anonymous option


 (myenv) robot@rpi4:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-unmap --patch-anonymous
map the whole file. total 4 pages 
after map
Page 0: 62 1c b8 0d 75 35 6b f7 1f 09 b8 ed 9a 20 59 d1 8a 2a 66 91 a1 b7 5b 63 de dd 64 8b d6 a4 59 93 ....
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 ....
Page 2: 9b e3 44 0f 55 24 cc 90 a1 92 ac 7e f7 a9 af 24 1e 9f a1 ca 28 8c 2d 76 28 6e 77 ae cb 76 29 34 ....
Page 3: bd 6a 65 b3 68 08 8e 47 7c 1d 9a ef 38 d3 0a cf 3b 2d be a2 49 3b 8e 6a f0 d6 6e 05 40 b2 33 81 ....
map page 0 to anonymous
map page 2 to anonymous
after map
Page 0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ....
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 ....
Page 2: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ....
Page 3: bd 6a 65 b3 68 08 8e 47 7c 1d 9a ef 38 d3 0a cf 3b 2d be a2 49 3b 8e 6a f0 d6 6e 05 40 b2 33 81 ....
unmap page 0
unmap page 2
after unmap...
touch page 1
Page 1: 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 58 f6 18 a7 56 b8 b0 e0 6d 6b aa 8b f2 58 a1 90 
map page 0 to anonymous
touch page 0
Page 0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 

^^^^ no segfault. the anonymous mapping is patched in. 

*/