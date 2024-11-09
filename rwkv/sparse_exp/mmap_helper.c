
// gcc -shared -o mmap_helper.so -fPIC mmap_helper.c

// https://linux.die.net/man/2/munmap

#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>

#include <libexplain/mmap.h>
#include <libexplain/munmap.h>


// batch mmap(), to be called from python code 

// Function to mmap each address from the passed list

int mmap_addresses0(uintptr_t *addresses, size_t num_addresses, size_t length, int prot, int flags, long *offsets, int fd) {
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));
        long offset = offsets[i] & ~(0xFFF);

        void *mapped_addr = mmap(addr, length, prot, flags, fd, offset);
        
        if (mapped_addr == MAP_FAILED) {
            perror("mmap_addresses: mmap failed");
            printf("i %d, addr %p, length %ld, prot %d, flags %d, fd %d, offset %ld\n", i, addr, length, prot, flags, fd, offsets[i]);
            return -1; 
        } else {
            // printf("Memory mapped at: %p for requested address: %p with offset: %ld\n", mapped_addr, addr, offsets[i]);
            ;
        }
    }
    return 0; 
}

// Function to mmap each address from the passed list
//   assuming addresses are sorted in ascending order.
// baesdon mmap_addresses0, but add check to avoid remapping the same page consecutively. 
//      b/c caller may pass tensor rows belonging to the same page. 
int mmap_addresses(uintptr_t *addresses, size_t num_addresses, size_t length, int prot, int flags, long *offsets, int fd) {
    void *last_mapped_addr = NULL;
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));     // page mask 
        long offset = offsets[i] & ~(0xFFF);

        // Calculate the end address aligned to the next page boundary
        void *end_addr = (void *)((addresses[i] + length + 0xFFF) & ~(0xFFF)); 
        // Calculate the actual length to map
        size_t actual_length = (uintptr_t)end_addr - (uintptr_t)addr;

        if (addr != last_mapped_addr) {
            void *mapped_addr = mmap(addr, actual_length, prot, flags, fd, offset);
            
            if (mapped_addr == MAP_FAILED) {
                perror("mmap_addresses: mmap failed");
                printf("i %d, addr %p, actual_length %ld, prot %d, flags %d, fd %d, offset %ld\n", 
                    i, addr, actual_length, prot, flags, fd, offset);
                fprintf(stderr, "%s\n",  explain_mmap(addr, actual_length, prot, flags, fd, offset));
                return -1; 
            } else {
                // printf("mmap_addresses: OK\n");
                // printf("Memory mapped at: %p for requested address: %p with offset: %ld\n", mapped_addr, addr, offsets[i]);
                // printf("i %d, addr %p, length %ld, prot %d, flags %d, fd %d, offset %ld\n", i, addr, length, prot, flags, fd, offset);
                last_mapped_addr = addr;
            }
        }
    }
    return 0; 
}

// Function to mmap each address from the passed list with anonymous mapping
//   assuming addresses are sorted in ascending order.
//   has check to avoid remapping the same page consecutively.
int mmap_anon_addresses(uintptr_t *addresses, size_t num_addresses, size_t length, int prot, int flags) {
    
    void *last_mapped_addr = NULL;
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));     // page mask 

        // Calculate the end address aligned to the next page boundary
        void *end_addr = (void *)((addresses[i] + length + 0xFFF) & ~(0xFFF)); 
        // Calculate the actual length to map
        size_t actual_length = (uintptr_t)end_addr - (uintptr_t)addr;

        if (addr != last_mapped_addr) {
            void *mapped_addr = mmap(addr, actual_length, prot, flags | MAP_ANONYMOUS, -1, 0);
            
            if (mapped_addr == MAP_FAILED) {
                perror("mmap_addresses_anon: mmap failed");
                printf("i %d, addr %p, actual_length %ld, prot %d, flags %d\n", 
                    i, addr, actual_length, prot, flags);
                fprintf(stderr, "%s\n", explain_mmap(addr, actual_length, prot, flags | MAP_ANONYMOUS, -1, 0));
                return -1; 
            } else {
                // printf(">>>>> mmap_addresses_anon: OK\n");
                // printf("Memory mapped at: %p for requested address: %p\n", mapped_addr, addr);
                last_mapped_addr = addr;
            }
        }
    }
    return 0; 
}

// Function to munmap each address from the passed list
//   assuming addresses are sorted in ascending order.
//   has check to avoid unmapping the same page consecutively.
int munmap_addresses(uintptr_t *addresses, size_t num_addresses, size_t length) {
    void *last_unmapped_addr = NULL;
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));
        // Calculate the end address aligned to the next page boundary
        void *end_addr = (void *)((addresses[i] + length + 0xFFF) & ~(0xFFF)); 
        // Calculate the actual length to unmap
        size_t actual_length = (uintptr_t)end_addr - (uintptr_t)addr; 

        if (addr != last_unmapped_addr) {
            int ret = munmap(addr, actual_length);
            if (ret == -1) {
                perror("munmap_addresses: munmap failed");
                fprintf(stderr, "%s\n", explain_munmap(addr, actual_length));
                return -1; 
            } else {
                printf("Memory unmapped at: %p, length: %zu\n", addr, actual_length);
                last_unmapped_addr = addr;
            }
        }
    }
    return 0; 
}


// idea1: assuing addresses are sorted ascending; check if the page is already mapped. if so, 
// skip the mmap() call.

// idea2: don't pass in both addresses & offsets, just pass in the base address, and the offsets