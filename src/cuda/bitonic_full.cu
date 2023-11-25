#include "struct.h"

extern __shared__ Struct block_structs[];

__device__ Struct *load_shared_memory(Struct const *const global_structs) {
  auto const offset{2 * blockDim.x * blockIdx.x};
  auto const position{threadIdx.x};

  // Each thread must copy two elements to shared block memory, as there
  // are twice as many elments as threads in a block.
  auto const even_position = 2 * position;
  auto const odd_position = even_position + 1;

  block_structs[even_position] = global_structs[offset + even_position];
  block_structs[odd_position] = global_structs[offset + odd_position];

  return block_structs;
}

__device__ void save_shared_memory(Struct *const global_structs) {
  auto const offset{2 * blockDim.x * blockIdx.x};
  auto const position{threadIdx.x};

  // Each thread must copy two elements from shared block memory, as there
  // are twice as many elments as threads in a block.
  auto const even_position = 2 * position;
  auto const odd_position = even_position + 1;

  global_structs[offset + even_position] = block_structs[even_position];
  global_structs[offset + odd_position] = block_structs[odd_position];
}

__forceinline__ __device__ void compare_and_swap(Struct *const structs,
                                                 uint const i, uint const j) {
  auto const i_value = structs[i].value;
  auto const j_value = structs[j].value;

  if (i_value > j_value) {
    structs[i].value = j_value;
    structs[j].value = i_value;
  }
}

__forceinline__ __device__ uint32_t modulo_pow2(uint32_t const value,
                                                uint32_t const power_of_two) {
  return value & (power_of_two - 1);
}

__forceinline__ __device__ void apply_local_flip(Struct *const structs,
                                                 uint32_t const height) {
  auto const thread_id{threadIdx.x};

  auto const half_height{height >> 1};
  auto const offset{
      (~(height - 1)) &
      (thread_id << 1)}; // equivalent to ((2 * t) / height) * height
  auto const i{offset + modulo_pow2(thread_id, half_height)};
  auto const j{offset + height - modulo_pow2(thread_id, half_height) - 1};

  compare_and_swap(structs, i, j);
}

// Performs progressively diminishing disperse operations on indices available
// in the same block: E.g. height == 8 -> 8 : 4 : 2.
//
// One disperse operation for every time we can divide h by 2.
__device__ void apply_local_disperse(__shared__ Struct *structs,
                                     uint32_t height) {
  auto const thread_id = threadIdx.x;
  for (; height > 1; height >>= 1) {
    auto const half_height{height >> 1};
    auto const offset{
        (~(height - 1)) &
        (thread_id << 1)}; // equivalent to ((2 * t) / height) * height
    auto const i{offset + modulo_pow2(thread_id, half_height)};
    auto const j{i + half_height};

    compare_and_swap(structs, i, j);
    __syncthreads();
  }
}

extern "C" __global__ void local_disperse(Struct *const global_structs,
                                          uint32_t height) {
  Struct *const block_structs = load_shared_memory(global_structs);
  __syncthreads();
  apply_local_disperse(block_structs, height);
  save_shared_memory(global_structs);
}

// Perform binary merge sort for local elements, up to a maximum
// number of elements h
extern "C" __global__ void local_binary_merge_sort(Struct *const global_structs,
                                                   uint32_t const height) {
  Struct *const block_structs = load_shared_memory(global_structs);
  for (uint32_t op_height{2}; op_height <= height; op_height <<= 1) {
    __syncthreads();
    apply_local_flip(block_structs, op_height);
    __syncthreads();
    apply_local_disperse(block_structs, op_height >> 1);
  }
  save_shared_memory(global_structs);
}

extern "C" __global__ void global_flip(Struct *const structs,
                                       uint32_t const height) {
  auto const t{blockDim.x * blockIdx.x + threadIdx.x};

  auto const half_height{height >> 1};
  // offset is equivalent to ((2 * t) / height) * height (height is a
  // power of two)
  auto const offset{(~(height - 1)) & (t << 1)};
  auto const i{offset + modulo_pow2(t, half_height)};
  auto const j{offset + height - modulo_pow2(t, half_height) - 1};

  compare_and_swap(structs, i, j);
}

extern "C" __global__ void global_disperse(Struct *const structs,
                                           uint32_t const height) {
  auto const t{blockDim.x * blockIdx.x + threadIdx.x};

  auto const half_height{height >> 1};
  // offset is equivalent to ((2 * t) / height) * height (height is a
  // power of two)
  auto const offset{(~(height - 1)) & (t << 1)};
  auto const i{offset + modulo_pow2(t, half_height)};
  auto const j{i + half_height};

  compare_and_swap(structs, i, j);
}
