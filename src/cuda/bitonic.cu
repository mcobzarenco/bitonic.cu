#include "struct.h"
#include <cstdio>
 
extern "C" __global__ void bitonic_sort(Struct *structs, const size_t n) {
    /* std::printf("%d %lu\n", index, n); */

    __syncthreads();

    uint left_index = 2 * threadIdx.x;
    float left_value = structs[left_index].value;

    uint right_index = 2 * threadIdx.x + 1;
    float right_value = structs[right_index].value; 
    
    if (right_value < left_value) {
        structs[left_index].value = right_value;
        structs[right_index].value = left_value;
    }
}

extern "C" __global__ void bitonic_sort(Struct *structs, const size_t n) {
    /* std::printf("%d %lu\n", index, n); */

    __syncthreads();

    uint left_index = 2 * threadIdx.x;
    float left_value = structs[left_index].value;

    uint right_index = 2 * threadIdx.x + 1;
    float right_value = structs[right_index].value; 
    
    if (right_value < left_value) {
        structs[left_index].value = right_value;
        structs[right_index].value = left_value;
    }
}


__device__ void compare_and_swap(Struct *structs, uint i, uint j) {
    if (structs[i].value < structs[j].value) {
        uint aux = structs[i].value;
	structs[i].value = structs[j].value;
	structs[j].value = aux;
    }
}

/* void do_flip(Struct *structs, int h) { */
/*     uint t = gl_LocalInvocationID.x; */
/*     int q = ((2 * t) / h) * h; */
/*     ivec2 indices = q + ivec2( t % h, h - (t % h) ); */
/*     local_compare_and_swap(indices); */
/* } */

/* void do_disperse(int h){ */
/* 	uint t = gl_LocalInvocationID.x; */
/* 	int q = ((2 * t) / h) * h; */
/* 	ivec2 indices = q + ivec2( t % h, (t % h) + (h / 2) ); */
/* 	local_compare_and_swap(indices); */
/* } */

__device__ void flip() {
  
}
