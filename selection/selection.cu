// Kernels behind the selection steps.

// dominated[i] = 1 if some row j strictly dominates row i of the k x k
// matrix M, given as its transpose MT (MT[c * k + r] = M[r, c], so that the
// threads of a warp, which share j and take consecutive i, read consecutive
// words). j dominates i when M[j, c] >= M[i, c] for every c and > for some c.
extern "C" __global__ void dominated_rows(const int *MT, const int k, unsigned char *dominated)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)k * k) return;
    const int j = (int)(t / k), i = (int)(t % k);
    if (i == j) return;
    int strict = 0;
    for (int c = 0; c < k; ++c) {
        const int a = MT[(long long)c * k + j], b = MT[(long long)c * k + i];
        if (a < b) return;
        strict |= a > b;
    }
    if (strict) dominated[i] = 1;
}

// 64-bit FNV-1a over each row of a row-major int32 matrix. One thread per row.
extern "C" __global__ void row_hash(const int *A, const int n_rows, const int n_cols, unsigned long long *out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    unsigned long long h = 0xcbf29ce484222325ull;
    const int *row = A + (long long)r * n_cols;
    for (int c = 0; c < n_cols; ++c) {
        h = (h ^ (unsigned)row[c]) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    out[r] = h;
}

// 1 where rows x[k] and y[k] of A are equal. One warp per pair.
extern "C" __global__ void rows_equal(const int *A, const int n_cols, const long long *x, const long long *y,
                                      unsigned char *out, const int n)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= n) return;
    const int *a = A + x[job] * n_cols, *b = A + y[job] * n_cols;
    int differ = 0;
    for (int c = lane; c < n_cols && !differ; c += 32) differ = a[c] != b[c];
    differ = __any_sync(0xffffffffu, differ);
    if (lane == 0) out[job] = !differ;
}

// Block-wide reductions over blockDim.x (a power of two) threads.
__device__ int block_max(int v, int *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) scratch[threadIdx.x] = max(scratch[threadIdx.x], scratch[threadIdx.x + s]);
        __syncthreads();
    }
    const int out = scratch[0];
    __syncthreads();
    return out;
}

__device__ long long block_sum(long long v, long long *scratch)
{
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
        __syncthreads();
    }
    const long long out = scratch[0];
    __syncthreads();
    return out;
}

// Lexicase survivor selection, n_accepted picks without replacement, in one
// block. Rows with identical payoff vectors form groups (rep[g] is one of
// them, group_of[r] the group of row r, left[g] its unselected members):
// cases cannot separate them, so filtering runs on groups and stops when one
// group remains. Pick p walks the cases in order[p * n_cases ...], keeping
// the candidate groups with the exact maximum payoff on each case, then takes
// the floor(tie[p] * m)-th of the m unselected rows of the remaining groups,
// in row order. PT is the payoff transposed: PT[c * n_rows + r] = payoff[r, c].
extern "C" __global__ void lexicase(const int *PT, const int n_rows, const int n_cases,
                                    const int n_groups, const long long *rep, const long long *group_of,
                                    int *left, unsigned char *cand, unsigned char *taken,
                                    const int *order, const double *tie, const int n_accepted,
                                    long long *selected)
{
    __shared__ int smax[1024];
    __shared__ long long ssum[1024];
    __shared__ long long winner;
    for (int p = 0; p < n_accepted; ++p) {
        long long count = 0;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            cand[g] = left[g] > 0;
            count += cand[g];
        }
        count = block_sum(count, ssum);
        for (int ci = 0; ci < n_cases && count > 1; ++ci) {
            const int *col = PT + (long long)order[(long long)p * n_cases + ci] * n_rows;
            int best = -2147483647 - 1;
            for (int g = threadIdx.x; g < n_groups; g += blockDim.x)
                if (cand[g]) best = max(best, col[rep[g]]);
            best = block_max(best, smax);
            count = 0;
            for (int g = threadIdx.x; g < n_groups; g += blockDim.x)
                if (cand[g]) {
                    cand[g] = col[rep[g]] == best;
                    count += cand[g];
                }
            count = block_sum(count, ssum);
        }
        long long m = 0;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x)
            if (cand[g]) m += left[g];
        m = block_sum(m, ssum);
        if (threadIdx.x == 0) {
            long long k = (long long)(tie[p] * (double)m);
            if (k >= m) k = m - 1;
            winner = -1;
            for (int r = 0; r < n_rows; ++r)
                if (!taken[r] && cand[group_of[r]] && k-- == 0) { winner = r; break; }
            taken[winner] = 1;
            left[group_of[winner]] -= 1;
            selected[p] = winner;
        }
        __syncthreads();
    }
}
