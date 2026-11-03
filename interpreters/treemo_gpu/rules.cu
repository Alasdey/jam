// Kernels behind interpreters/treemo_gpu/rules.py.
#include "tapes.cuh"

// Tape k's bits, one byte each, at out[dst[k]...]. One warp per tape.
extern "C" __global__ void unpack_tapes(const unsigned *heap, const long long *off, const int *len,
                                        const long long *dst, unsigned char *out, const int n)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= n) return;
    const unsigned *t = heap + off[job];
    unsigned char *o = out + dst[job];
    const long long bits = len[job];
    for (long long i = lane; i < bits; i += 32) o[i] = (t[i >> 5] >> (i & 31)) & 1u;
}

// 1 where rule r's pattern equals its replacement. rules[r] = (pattern bit,
// pattern length, replacement bit, replacement length) relative to the
// program's first bit, base[r] (in bits). One thread per rule.
extern "C" __global__ void rule_ident(const unsigned *heap, const int4 *rules, const long long *base,
                                      unsigned char *ident, const int n)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const int4 q = rules[r];
    int same = q.y == q.w;
    for (int k = 0; same && k < q.y; k += 32) {
        const int m = min(32, q.y - k);
        const unsigned mask = m == 32 ? FULL : (FULL >> (32 - m));
        same = ((window(heap, base[r] + q.x + k) ^ window(heap, base[r] + q.z + k)) & mask) == 0u;
    }
    ident[r] = same;
}
