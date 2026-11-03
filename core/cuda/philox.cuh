// Philox4x32-10 (Salmon et al. 2011), the counter-based generator behind
// every random draw in jam. A draw is a pure function of its counter and the
// run's key, so nothing needs saving: a resumed generation draws exactly what
// the original did. Counters are (element, draw, generation, call): `element`
// is a thread's job (an offspring, a row), `draw` counts its draws, `call`
// separates the launches of one generation (see core/rng.py).

#pragma once

struct Philox {
    unsigned c0, c1, c2, c3;  // counter; c1 advances by one per block of 4 words
    unsigned k0, k1;
    unsigned out[4];
    int left;                 // unused words in out
};

__device__ __forceinline__ void philox_round(unsigned &c0, unsigned &c1, unsigned &c2, unsigned &c3,
                                             unsigned k0, unsigned k1)
{
    const unsigned long long p0 = 0xD2511F53ull * c0;
    const unsigned long long p1 = 0xCD9E8D57ull * c2;
    const unsigned hi0 = (unsigned)(p0 >> 32), lo0 = (unsigned)p0;
    const unsigned hi1 = (unsigned)(p1 >> 32), lo1 = (unsigned)p1;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
}

__device__ __forceinline__ void philox4x32_10(unsigned c[4], unsigned k0, unsigned k1)
{
    for (int r = 0; r < 10; ++r) {
        if (r) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
        philox_round(c[0], c[1], c[2], c[3], k0, k1);
    }
}

__device__ __forceinline__ void rng_init(Philox &r, unsigned k0, unsigned k1,
                                         unsigned element, unsigned gen, unsigned call)
{
    r.c0 = element; r.c1 = 0; r.c2 = gen; r.c3 = call;
    r.k0 = k0; r.k1 = k1;
    r.left = 0;
}

__device__ __forceinline__ unsigned rng_u32(Philox &r)
{
    if (r.left == 0) {
        unsigned c[4] = {r.c0, r.c1, r.c2, r.c3};
        philox4x32_10(c, r.k0, r.k1);
        r.out[0] = c[0]; r.out[1] = c[1]; r.out[2] = c[2]; r.out[3] = c[3];
        r.c1 += 1;
        r.left = 4;
    }
    return r.out[4 - r.left--];
}

// Uniform in [0, 1) with 53 random bits.
__device__ __forceinline__ double rng_uniform(Philox &r)
{
    const unsigned a = rng_u32(r) >> 5, b = rng_u32(r) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// Uniform integer in [0, n), n >= 1, unbiased (Lemire's rejection).
__device__ __forceinline__ unsigned rng_below(Philox &r, unsigned n)
{
    unsigned long long m = (unsigned long long)rng_u32(r) * n;
    unsigned lo = (unsigned)m;
    if (lo < n) {
        const unsigned threshold = (0u - n) % n;
        while (lo < threshold) {
            m = (unsigned long long)rng_u32(r) * n;
            lo = (unsigned)m;
        }
    }
    return (unsigned)(m >> 32);
}

// True with probability p.
__device__ __forceinline__ bool rng_bernoulli(Philox &r, double p)
{
    return rng_uniform(r) < p;
}
