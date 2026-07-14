#pragma once

#ifndef CUDA_BLOCK_SIZE
#define CUDA_BLOCK_SIZE 256
#endif

#if CUDA_BLOCK_SIZE <= 0
#error "CUDA_BLOCK_SIZE must be positive"
#endif

constexpr int kCudaBlockSize = CUDA_BLOCK_SIZE;
