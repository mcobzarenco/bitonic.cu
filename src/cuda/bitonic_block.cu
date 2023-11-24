#include "struct.h"

__device__ void compare_and_swap(Struct *structs, uint i, uint j) {
  if (structs[i].value > structs[j].value) {
    uint aux = structs[i].value;
    structs[i].value = structs[j].value;
    structs[j].value = aux;
  }
}

__device__ void do_flip(Struct *structs, uint height) {
  uint t = threadIdx.x;

  uint half_height = height / 2;
  uint q = ((2 * t) / height) * height;
  uint i = q + t % half_height;
  uint j = q + height - (t % half_height) - 1;

  compare_and_swap(structs, i, j);
}

__device__ void do_disperse(Struct *structs, uint height) {
  uint t = threadIdx.x;

  uint half_height = height / 2;
  uint q = ((2 * t) / height) * height;
  uint i = q + t % half_height;
  uint j = q + (t % half_height) + half_height;

  compare_and_swap(structs, i, j);
}

extern "C" __global__ void bitonic_sort(Struct *structs, const size_t length) {
  uint thread_id = threadIdx.x;
  if (thread_id >= length / 2) {
    return;
  }

  // Each thread must copy two elements to shared block memory, as there
  // are twice as many elments as threads in a block.
  __shared__ Struct local_structs[1024];
  local_structs[2 * thread_id] = structs[2 * thread_id];
  local_structs[2 * thread_id + 1] = structs[2 * thread_id + 1];

  for (uint height = 2; height <= length; height *= 2) {
    __syncthreads();
    do_flip(local_structs, height);

    for (uint dheight = height / 2; dheight > 1; dheight /= 2) {
      __syncthreads();
      do_disperse(local_structs, dheight);
    }
  }

  __syncthreads();

  // Write local memory back to buffer
  structs[2 * thread_id] = local_structs[2 * thread_id];
  structs[2 * thread_id + 1] = local_structs[2 * thread_id + 1];
}
