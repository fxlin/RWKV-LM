// dd if=/dev/random of=/tmp/existing_file.bin bs=4K count=2

// https://man7.org/linux/man-pages/man2/madvise.2.html

// upate: MADV_REMOVE works. but thta's likely the same as overlapped anonymous mapping?

// WONT WORK -- B/C madvise(..MADV_DONTNEED) only returns 0 on anonymous mapping.
// it will still load contents from disk 

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int test_MADV_DONTNEED() {
    const char *filename = "/tmp/existing_file.bin";
    size_t file_size = 4096*2; 
    int fd;
    char *map;

    // 1. Open the existing file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }

    // 2. Get the size of the file (using fstat or assume fixed size)
    // In this case, we assume the file size is known (4KB), but you can also use fstat()
    
    // 3. Memory-map the file
    map = (char *)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return 1;
    }

    // 5. Use madvise() to mark a part of the region as unneeded (MADV_DONTNEED)
    size_t advise_offset = 4096;  // Start from 1024 bytes offset (within the 4KB range)
    size_t advise_length = 4096;  // A 1KB region

    printf("Accessing the advised region BEFORE ... : ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }    

    if (madvise(map + advise_offset, advise_length, MADV_DONTNEED) == -1) {
        perror("Error with madvise");
        munmap(map, file_size);
        close(fd);
        return 1;
    }

    printf("Marked region from byte %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)advise_offset, (unsigned long)(advise_offset + advise_length));

    // 6. Accessing the region after madvise() with MADV_DONTNEED
    printf("Accessing the advised region (should return zeros or cleared data): ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }
    printf("\n");

    // 7. Write new data to the region after MADV_DONTNEED (which re-allocates pages)
    strcpy(map + advise_offset, "New data after MADV_DONTNEED.");

    // 8. Read back the new data
    printf("New data in the advised region: %s\n", map + advise_offset);

    // 9. Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);

    return 0;
}


/* 
    this works 

(myenv) robot@rpi4:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-madvise 
Accessing the advised region BEFORE ... : 2a df ee 27 70 94 e3 eb 5d 6b 7a 47 ae 36 2f 87 Marked region from byte 4096 to 8192 as unneeded using MADV_REMOVE .
Accessing the advised region (should return zeros or cleared data): 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 
New data in the advised region: New data after MADV_REMOVE .


        https://man7.org/linux/man-pages/man2/madvise.2.html
Free up a given range of pages and its associated backing
    store.  This is equivalent to punching a hole in the
    corresponding range of the backing store (see
    fallocate(2)).  Subsequent accesses in the specified
    address range will see data with a value of zero.

    The specified address range must be mapped shared and
    writable.  */

int test_MADV_REMOVE() {
    const char *filename = "/tmp/existing_file.bin";
    size_t file_size = 4096*2; 
    int fd;
    char *map;

    // 1. Open the existing file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }

    // 2. Get the size of the file (using fstat or assume fixed size)
    // In this case, we assume the file size is known (4KB), but you can also use fstat()
    
    // 3. Memory-map the file
    map = (char *)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return 1;
    }

    // 5. Use madvise() to mark a part of the region as unneeded (MADV_DONTNEED)
    size_t advise_offset = 4096;  // Start from 1024 bytes offset (within the 4KB range)
    size_t advise_length = 4096;  // A 1KB region

    printf("Accessing the advised region BEFORE ... : ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }    

    if (madvise(map + advise_offset, advise_length, MADV_REMOVE) == -1) {
        perror("Error with madvise");
        munmap(map, file_size);
        close(fd);
        return 1;
    }

    printf("Marked region from byte %lu to %lu as unneeded using MADV_REMOVE .\n", 
           (unsigned long)advise_offset, (unsigned long)(advise_offset + advise_length));

    // 6. Accessing the region after madvise() with MADV_REMOVE 
    printf("Accessing the advised region (should return zeros or cleared data): ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }
    printf("\n");

    // 7. Write new data to the region after MADV_REMOVE  (which re-allocates pages)
    strcpy(map + advise_offset, "New data after MADV_REMOVE .");

    // 8. Read back the new data
    printf("New data in the advised region: %s\n", map + advise_offset);

    // 9. Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);

    return 0;
}

int main() {
    test_MADV_REMOVE();
    return 0; 
}

