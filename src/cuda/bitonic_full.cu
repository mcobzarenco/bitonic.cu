#include "struct.h"
#include <cstdint>
#include <cstdio>

extern __shared__ float shared[];

constexpr size_t BLOCK_SIZE = 1024;

__device__ Struct * load_shared_memory(Struct const *const global_structs) {
  uint32_t const offset{2 * blockDim.x * blockIdx.x};
  uint32_t const position{threadIdx.x};
  /* std::printf("offset: %u | position: %u\n", offset, position);  */

  // Each thread must copy two elements to shared block memory, as there
  // are twice as many elments as threads in a block.
  __shared__ Struct block_structs[BLOCK_SIZE];
  block_structs[2 * position] = global_structs[offset + 2 * position];
  block_structs[2 * position + 1] = global_structs[offset + 2 * position + 1];

  return block_structs;
}

__device__ void save_shared_memory(Struct *const global_structs,
                                   __shared__ Struct const *const block_structs) {
  uint32_t const offset{2 * blockDim.x * blockIdx.x};
  uint32_t const position{threadIdx.x};

  global_structs[offset + 2 * position] = block_structs[2 * position];
  global_structs[offset + 2 * position + 1] = block_structs[2 * position + 1];
}

__forceinline__
__device__ void compare_and_swap(Struct *const structs, uint const i,
                                 uint const j) {
  if (structs[i].value > structs[j].value) {
    const auto aux{structs[i].value};
    structs[i].value = structs[j].value;
    structs[j].value = aux;
  }
}

__forceinline__
__device__ uint32_t fast_modulo(uint32_t const value, uint32_t const modulo) {
    return value & (modulo - 1);
}

__forceinline__
__device__ void apply_local_flip(Struct *const structs, uint32_t const height) {
  uint32_t const t = threadIdx.x;

  uint32_t const half_height = height >> 1;
  uint32_t const q = ((t << 1) / height) * height;
  uint32_t const i = q + fast_modulo(t, half_height);
  uint32_t const j = q + height - fast_modulo(t, half_height) - 1;

  compare_and_swap(structs, i, j);
}

// Performs progressively diminishing disperse operations on indices available
// in the same block: E.g. height == 8 -> 8 : 4 : 2.
//
// One disperse operation for every time we can divide h by 2.
__device__ void apply_local_disperse(__shared__ Struct *structs, uint32_t height) {
  uint32_t const t = threadIdx.x;

  /* uint32_t const originalh = height; */

  for (; height > 1; height /= 2) {
    __syncthreads();
    uint32_t const half_height{height >> 1};
    uint32_t const q{((t << 1) / height) * height};
    uint32_t const i{q + fast_modulo(t, half_height)};
    uint32_t const j{i + half_height};

    /* std::printf("[apply_local_disperse(height=%u)] h=%u i=%u j=%u s[i]=%f
     * s[j]=%f\n", originalh, */
    /*             height, i, j, structs[i].value, structs[j].value); */

    compare_and_swap(structs, i, j);
  }
}

extern "C" __global__ void local_disperse(Struct *const global_structs,
                                          uint32_t const height) {
  Struct * const block_structs = load_shared_memory(global_structs);
  __syncthreads();
  apply_local_disperse(block_structs, height);
  __syncthreads();
  save_shared_memory(global_structs, block_structs);
}

// Perform binary merge sort for local elements, up to a maximum
// number of elements h
extern "C" __global__ void local_binary_merge_sort(Struct *const global_structs,
                                                   uint32_t const height) {
  Struct *const block_structs = load_shared_memory(global_structs);
  for (uint op_height{2}; op_height <= height; op_height <<= 1) {
    __syncthreads();
    apply_local_flip(block_structs, op_height);
    __syncthreads();
    apply_local_disperse(block_structs, op_height / 2);
  }
  __syncthreads();
  save_shared_memory(global_structs, block_structs);
}

extern "C" __global__ void global_flip(Struct *const structs,
                                       uint32_t const height) {
  auto const t{blockDim.x * blockIdx.x + threadIdx.x};

  auto const half_height{height >> 1};
  auto const q{((t << 1) / height) * height};
  auto const i{q + fast_modulo(t, half_height)};
  auto const j{q + height - fast_modulo(t, half_height) - 1};

  compare_and_swap(structs, i, j);
}

extern "C" __global__ void global_disperse(Struct *const structs,
                                           uint32_t const height) {
  auto const t{blockDim.x * blockIdx.x + threadIdx.x};

  auto const half_height{height >> 1};
  auto const q{((t << 1) / height) * height};
  auto const i{q + fast_modulo(t, half_height)};
  auto const j{i + half_height};

  compare_and_swap(structs, i, j);
}
