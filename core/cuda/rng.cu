// Array draws for core/rng.py: element i of a call uses counter (i, 0, gen, call).
#include "philox.cuh"

extern "C" __global__ void rng_uniform_fill(double *out, const long long n, const unsigned k0, const unsigned k1,
                                            const unsigned gen, const unsigned call)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Philox r;
    rng_init(r, k0, k1, (unsigned)i, gen, call);
    out[i] = rng_uniform(r);
}

// out[i] uniform in [0, high[i]) (high[0] for every i when scalar); high >= 1.
extern "C" __global__ void rng_below_fill(long long *out, const long long *high, const long long n,
                                          const int scalar, const unsigned k0, const unsigned k1,
                                          const unsigned gen, const unsigned call)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Philox r;
    rng_init(r, k0, k1, (unsigned)i, gen, call);
    out[i] = rng_below(r, (unsigned)high[scalar ? 0 : i]);
}
