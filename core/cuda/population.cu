// Kernels behind core/population.py.
#include "tapes.cuh"

// Copy tape k (its words and the trailing zero word) from src at src_off[k]
// to dst at dst_off[k]. One warp per tape.
extern "C" __global__ void copy_tapes(const unsigned *src, const long long *src_off, const int *len,
                                      unsigned *dst, const long long *dst_off, const int n)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= n) return;
    const long long words = tape_words(len[job]) + 1;
    const unsigned *s = src + src_off[job];
    unsigned *d = dst + dst_off[job];
    for (long long w = lane; w < words; w += 32) d[w] = s[w];
}

// 64-bit FNV-1a of each tape's words, with its length mixed in first. One thread per tape.
extern "C" __global__ void hash_tapes(const unsigned *heap, const long long *off, const int *len,
                                      unsigned long long *out, const int n)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned long long h = 0xcbf29ce484222325ull;
    const unsigned long long prime = 0x100000001b3ull;
    h = (h ^ (unsigned long long)(unsigned)len[k]) * prime;
    const unsigned *t = heap + off[k];
    const long long words = tape_words(len[k]);
    for (long long w = 0; w < words; ++w) {
        h = (h ^ t[w]) * prime;
        h ^= h >> 29;
    }
    out[k] = h;
}

// 1 where tapes x[k] and y[k] are equal. One warp per pair.
extern "C" __global__ void equal_tapes(const unsigned *heap, const long long *off, const int *len,
                                       const long long *x, const long long *y, unsigned char *out, const int n)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= n) return;
    const long long a = x[job], b = y[job];
    int differ = len[a] != len[b];
    if (!differ) {
        const unsigned *ta = heap + off[a], *tb = heap + off[b];
        const long long words = tape_words(len[a]);
        for (long long w = lane; w < words && !differ; w += 32) differ = ta[w] != tb[w];
    }
    differ = __any_sync(FULL, differ);
    if (lane == 0) out[job] = !differ;
}
