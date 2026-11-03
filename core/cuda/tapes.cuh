// Tape layout shared by every jam kernel.
//
// A tape is a sequence of 0/1 symbols packed LSB first into 32-bit words (bit
// j of word w is symbol 32w + j). In a heap, every tape starts on a word
// boundary, its bits past its length are zero, and it is followed by a zero
// word; the heap itself starts with a zero word. So a 32-bit window may start
// up to 32 bits before a tape or end up to 32 bits after it.

#pragma once

#define FULL 0xffffffffu

// Words of a tape of n bits, without the trailing zero word.
__device__ __forceinline__ long long tape_words(long long n) { return (n + 31) >> 5; }

// 32 bits of A starting at bit offset `off` (off may be as low as -32).
__device__ __forceinline__ unsigned window(const unsigned *A, long long off)
{
    const long long w = off >> 5;  // arithmetic shift: floor division
    return __funnelshift_r(A[w], A[w + 1], (unsigned)(off & 31));
}

// Mask of the valid bits in the last word of an n-bit tape (n > 0).
__device__ __forceinline__ unsigned last_mask(long long n)
{
    const int r = (int)(n & 31);
    return r ? (FULL >> (32 - r)) : FULL;
}

__device__ __forceinline__ int bit_at(const unsigned *A, long long i)
{
    return (A[i >> 5] >> (i & 31)) & 1u;
}
