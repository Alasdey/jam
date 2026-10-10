// Batched Treemo on CUDA: one warp per job, tapes bit-packed LSB-first into
// 32-bit words (bit j of word w is symbol 32*w + j).
//
// All tapes of a batch (programs, inputs, outputs) live in one heap H. Each
// starts on a word boundary and is followed by a zero word, and H starts with
// a zero word, so a 32-bit window may begin up to 32 bits before or end up to
// 32 bits after any tape. Jobs address tapes by bit offset into H (a multiple
// of 32) and length. Rules are not copied: a pattern or replacement is a bit
// range of its program. Each execution owns a state buffer with one guard
// word on each side, sized to the exact growth bound computed on the host.

#define FULL 0xffffffffu

// 32 bits of A starting at bit offset `off`. Requires a readable word on each
// side of the window, which is what the guard words provide; off may be as low
// as -32 when A points one word past a guard.
__device__ __forceinline__ unsigned window(const unsigned *A, long long off)
{
    const long long w = off >> 5;  // arithmetic shift: floor division
    return __funnelshift_r(A[w], A[w + 1], (unsigned)(off & 31));
}

// Leftmost start s in [from, n - m] where T[s, s + m) equals the m-bit pattern
// at H bit `pat`, or -1. Lane L tests the 32 starts of word (blk >> 5) + L,
// keeping one survivor bit per start and dropping them a pattern bit at a time.
__device__ int warp_find(const unsigned *T, int n, const unsigned *H,
                         long long pat, int m, int from, int lane)
{
    const int last = n - m;
    for (int blk = from & ~31; blk <= last; blk += 1024) {
        const int s0 = blk + (lane << 5);
        unsigned alive = 0u;
        if (s0 <= last) {
            alive = FULL;
            if (s0 < from) alive <<= (from - s0);
            if (last - s0 < 31) alive &= FULL >> (31 - (last - s0));
            const unsigned *Tw = T + (s0 >> 5);
            unsigned lo = Tw[0], hi = Tw[1];
            for (int kw = 0; alive && kw < m; kw += 32) {
                if (kw) { lo = hi; hi = Tw[(kw >> 5) + 1]; }
                const unsigned P = window(H, pat + kw);
                const int kend = min(32, m - kw);
                for (int k = 0; k < kend; ++k) {
                    const unsigned W = __funnelshift_r(lo, hi, k);
                    alive &= ((P >> k) & 1u) ? W : ~W;
                    if (!alive) break;
                }
            }
        }
        const unsigned any = __ballot_sync(FULL, alive != 0u);
        if (any) {
            const int L = __ffs(any) - 1;
            return blk + (L << 5) + __ffs(__shfl_sync(FULL, alive, L)) - 1;
        }
    }
    return -1;
}

// Replace T[pos, pos + m) by the r-bit replacement at H bit `rep`, in place.
// Destination words are rewritten 32 at a time, every lane reading before any
// lane writes; growing tapes are walked top-down and shrinking ones bottom-up,
// so no read ever sees a word that an earlier round already moved.
__device__ void warp_splice(unsigned *T, int n, int pos, int m,
                            const unsigned *H, long long rep, int r, int lane)
{
    const int delta = r - m;
    const int w0 = pos >> 5;
    const int wend = delta == 0 ? (pos + r - 1) >> 5 : (n + delta - 1) >> 5;
    const int count = wend - w0 + 1;
    const int tail_start = pos + r;
    for (int c = 0; c < count; c += 32) {
        const int idx = c + lane;
        const bool active = idx < count;
        const int d = delta > 0 ? wend - idx : w0 + idx;
        unsigned val = 0u;
        if (active) {
            const int b0 = d << 5;
            const int a = pos - b0;         // bits below a keep the prefix
            const int t = tail_start - b0;  // bits from t on take the shifted tail
            const unsigned pre = a <= 0 ? 0u : (a >= 32 ? FULL : FULL >> (32 - a));
            const unsigned tail = t >= 32 ? 0u : (t <= 0 ? FULL : FULL << t);
            const unsigned mid = ~(pre | tail);
            if (pre) val |= T[d] & pre;
            if (mid) val |= window(H, rep + (b0 - pos)) & mid;
            if (tail) val |= window(T, (long long)b0 - delta) & tail;
        }
        __syncwarp();
        if (active) T[d] = val;
        __syncwarp();
    }
}

// Mirrors run() in interpreters/treemo_c/treemo.c step for step, including
// its dirty[] scan watermarks, so results are identical by construction.
// rules[k] = (pattern bit offset, pattern length, replacement bit offset,
// replacement length), offsets relative to the program's first bit; a job's
// rules are rules[job_rule[job] .. job_rule[job] + job_nrules[job]).
extern "C" __global__ void treemo_run(
    const unsigned *H, const int4 *rules, const unsigned char *ident,
    const long long *job_code, const int *job_rule, const int *job_nrules,
    const long long *job_input, const int *job_input_len,
    const long long *job_state, const int *job_cap, const long long *job_dirty,
    unsigned *pool, int *dpool, int *out_len,
    const int njobs, const int max_step, const int pass_mode, const int first_mode)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;

    unsigned *T = pool + job_state[job] + 1;
    const int cap = job_cap[job];
    int n = job_input_len[job];
    const unsigned *I = H + (job_input[job] >> 5);
    for (int w = lane; w <= (n + 31) >> 5; w += 32) T[w] = I[w];
    if (lane == 0) T[-1] = 0u;

    const long long base = job_code[job];
    const int nr = job_nrules[job];
    const int4 *R = rules + job_rule[job];
    const unsigned char *Id = ident + job_rule[job];
    int *D = dpool + job_dirty[job];
    for (int j = lane; j < nr; j += 32) D[j] = 0;
    __syncwarp();

    int rule = 0, no_fire = 0, fired = 0, overflow = 0;
    for (int step = 0; nr && step < max_step; ) {
        const int4 r = R[rule];
        const int pos = warp_find(T, n, H, base + r.x, r.y, D[rule], lane);
        if (pos >= 0) {
            if (Id[rule]) break;
            if (n + r.w - r.y > cap) { overflow = 1; break; }
            warp_splice(T, n, pos, r.y, H, base + r.z, r.w, lane);
            n += r.w - r.y;
            ++step;
            no_fire = 0;
            fired = 1;
            for (int j = lane; j < nr; j += 32) {
                int d = pos + 1 - R[j].y;
                if (d < 0) d = 0;
                if (j == rule || (pass_mode && first_mode && j < rule) || d < D[j]) D[j] = d;
            }
            __syncwarp();
            if (pass_mode) {
                rule = first_mode ? 0 : (rule + 1 == nr ? 0 : rule + 1);
                fired = 0;
            }
        } else {
            if (lane == 0) D[rule] = n;
            __syncwarp();
            if (!fired) {
                if (++no_fire >= nr) break;
                rule = rule + 1 == nr ? 0 : rule + 1;
            } else {
                no_fire = 0;
                fired = 0;
                rule = first_mode ? 0 : (rule + 1 == nr ? 0 : rule + 1);
            }
        }
    }
    if (lane == 0) out_len[job] = overflow ? -1 : n;
}

// Exact LCS length of each final state against its own program, bit-parallel
// over the program (Crochemore et al. 2001): V' = (V + (V & M[c])) | (V & ~M[c]),
// LCS = zeros of V. The program's words are spread over the warp, K per lane;
// the multiword add resolves inter-lane carries with one carry-lookahead
// over ballots. V only loses bits, so once every program bit is matched
// (LCS = m) the rest of the output cannot change the result and is skipped.
// Programs longer than 1024 * 16 bits are left to the host.
template <int K>
__device__ void lcs_job(const unsigned *T, int n, const unsigned *Pw, int m,
                        int lane, int *out)
{
    const int nw = (m + 31) >> 5;
    unsigned V[K], M1[K], M0[K], valid[K];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int w = lane * K + i;
        valid[i] = w >= nw ? 0u : (w == nw - 1 && (m & 31)) ? FULL >> (32 - (m & 31)) : FULL;
        M1[i] = w < nw ? Pw[w] & valid[i] : 0u;
        M0[i] = ~M1[i] & valid[i];
        V[i] = valid[i];
    }
    for (int t0 = 0; t0 < n; t0 += 32) {
        const unsigned tw = T[t0 >> 5];
        const int tend = min(32, n - t0);
        for (int b = 0; b < tend; ++b) {
            const bool one = (tw >> b) & 1u;
            unsigned S[K];
            unsigned carry = 0u, ones = FULL;
#pragma unroll
            for (int i = 0; i < K; ++i) {
                const unsigned M = one ? M1[i] : M0[i];
                const unsigned long long s = (unsigned long long)V[i] + (V[i] & M) + carry;
                S[i] = (unsigned)s;
                carry = (unsigned)(s >> 32);
                ones &= S[i];
            }
            const unsigned gen = __ballot_sync(FULL, carry);
            const unsigned prop = __ballot_sync(FULL, ones == FULL);
            if ((((gen + (gen | prop)) ^ prop) >> lane) & 1u) {
#pragma unroll
                for (int i = 0; i < K; ++i)
                    if (++S[i]) break;
            }
#pragma unroll
            for (int i = 0; i < K; ++i) V[i] = (S[i] | (V[i] & ~(one ? M1[i] : M0[i]))) & valid[i];
        }
        unsigned left = 0u;
#pragma unroll
        for (int i = 0; i < K; ++i) left |= V[i];
        if (!__any_sync(FULL, left)) break;
    }
    int zeros = 0;
#pragma unroll
    for (int i = 0; i < K; ++i) zeros += __popc(~V[i] & valid[i]);
    zeros = __reduce_add_sync(FULL, zeros);
    if (lane == 0) *out = zeros;
}

template <int K>
__device__ void lcs_dispatch(const unsigned *T, int n, const unsigned *Pw, int m,
                             int lane, int *out, int per_lane)
{
    if (per_lane == K) lcs_job<K>(T, n, Pw, m, lane, out);
    else lcs_dispatch<K + 1>(T, n, Pw, m, lane, out, per_lane);
}

template <>
__device__ void lcs_dispatch<17>(const unsigned *, int, const unsigned *, int,
                                 int lane, int *out, int)
{
    if (lane == 0) *out = -2;
}

// LCS length of each (text, ref) pair of heap tapes; ref is the shorter one.
extern "C" __global__ void treemo_lcs(
    const unsigned *H, const long long *text_bit, const int *text_len,
    const long long *ref_bit, const int *ref_len, int *lcs, const int njobs)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;
    const int n = text_len[job], m = ref_len[job];
    if (n == 0 || m == 0) { if (lane == 0) lcs[job] = 0; return; }
    const int per_lane = (((m + 31) >> 5) + 31) >> 5;
    lcs_dispatch<1>(H + (text_bit[job] >> 5), n, H + (ref_bit[job] >> 5), m,
                    lane, lcs + job, per_lane);
}

__device__ __forceinline__ unsigned last_mask(int n)
{
    return (n & 31) ? FULL >> (32 - (n & 31)) : FULL;
}

// Copy each final state into the heap at dst_word, with its trailing zero word.
extern "C" __global__ void treemo_gather(
    const long long *job_state, const int *out_len, const long long *dst_word,
    const unsigned *pool, unsigned *H, const int njobs)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;
    const int n = out_len[job];
    if (n < 0) return;
    const unsigned *T = pool + job_state[job] + 1;
    unsigned *O = H + dst_word[job];
    const int nw = (n + 31) >> 5;
    for (int w = lane; w <= nw; w += 32)
        O[w] = w < nw ? (w == nw - 1 ? T[w] & last_mask(n) : T[w]) : 0u;
}

// 1 where tapes x and y are equal, else 0.
extern "C" __global__ void treemo_equal(
    const unsigned *H, const long long *x_bit, const int *x_len,
    const long long *y_bit, const int *y_len, int *out, const int njobs)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;
    const int n = x_len[job];
    if (n != y_len[job]) { if (lane == 0) out[job] = 0; return; }
    const unsigned *X = H + (x_bit[job] >> 5), *Y = H + (y_bit[job] >> 5);
    const int nw = (n + 31) >> 5;
    unsigned diff = 0u;
    for (int w = lane; w < nw; w += 32)
        diff |= (X[w] ^ Y[w]) & (w == nw - 1 ? last_mask(n) : FULL);
    const int differ = __any_sync(FULL, diff != 0u);
    if (lane == 0) out[job] = !differ;
}

// Number of 1 symbols in each tape.
extern "C" __global__ void treemo_ones(
    const unsigned *H, const long long *x_bit, const int *x_len, int *out, const int njobs)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;
    const int n = x_len[job];
    const unsigned *X = H + (x_bit[job] >> 5);
    const int nw = (n + 31) >> 5;
    int ones = 0;
    for (int w = lane; w < nw; w += 32)
        ones += __popc(X[w] & (w == nw - 1 ? last_mask(n) : FULL));
    ones = __reduce_add_sync(FULL, ones);
    if (lane == 0) out[job] = ones;
}

// x followed by y, written into the heap at dst_word with a trailing zero word.
extern "C" __global__ void treemo_concat(
    unsigned *H, const long long *x_bit, const int *x_len,
    const long long *y_bit, const int *y_len, const long long *dst_word, const int njobs)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= njobs) return;
    const int nx = x_len[job], n = nx + y_len[job];
    const unsigned *X = H + (x_bit[job] >> 5), *Y = H + (y_bit[job] >> 5);
    unsigned *D = H + dst_word[job];
    const int nw = (n + 31) >> 5;
    for (int d = lane; d <= nw; d += 32) {
        unsigned v = 0u;
        if (d < nw) {
            const int a = nx - (d << 5);  // bits below a come from x
            const unsigned from_x = a <= 0 ? 0u : (a >= 32 ? FULL : FULL >> (32 - a));
            if (from_x) v |= X[d] & from_x;
            if (~from_x) v |= window(Y, (long long)(d << 5) - nx) & ~from_x;
            if (d == nw - 1) v &= last_mask(n);
        }
        D[d] = v;
    }
}

// The same LCS one job per thread, for a ref of at most 32 * K bits: no
// inter-lane carries, so far less work per symbol than the warp version. V
// lives in registers; the ref's words too while they fit (MREG), else they are
// re-read from the (L1-cached) heap. Words above the ref collect carries but
// never feed back into it (carries only move up), so they are masked only
// when checked.
template <int K, bool MREG>
__device__ int lcs_thread(const unsigned *T, int n, const unsigned *P, int m)
{
    const int nw = (m + 31) >> 5;
    unsigned V[K], M1[MREG ? K : 1];
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const unsigned valid = i < nw - 1 ? FULL : (i == nw - 1 ? last_mask(m) : 0u);
        if (MREG) M1[i] = i < nw ? P[i] & valid : 0u;
        V[i] = valid;
    }
    for (int t0 = 0; t0 < n; t0 += 32) {
        const unsigned tw = T[t0 >> 5];
        const int tend = min(32, n - t0);
        for (int b = 0; b < tend; ++b) {
            const unsigned flip = ((tw >> b) & 1u) - 1u;  // 0 for a 1 symbol, all ones for a 0
            unsigned carry = 0u;
#pragma unroll
            for (int i = 0; i < K; ++i) {
                const unsigned M = (MREG ? M1[i] : (i < nw ? P[i] : 0u)) ^ flip;
                const unsigned long long s = (unsigned long long)V[i] + (V[i] & M) + carry;
                carry = (unsigned)(s >> 32);
                V[i] = (unsigned)s | (V[i] & ~M);
            }
        }
        unsigned left = 0u;
#pragma unroll
        for (int i = 0; i < K; ++i)
            left |= V[i] & (i < nw - 1 ? FULL : (i == nw - 1 ? last_mask(m) : 0u));
        if (!left) break;
    }
    int zeros = 0;
#pragma unroll
    for (int i = 0; i < K; ++i)
        zeros += __popc(~V[i] & (i < nw - 1 ? FULL : (i == nw - 1 ? last_mask(m) : 0u)));
    return zeros;
}

// One launch per K (uniform across the launch, so the switch never diverges);
// every ref in it has at most 32 * K bits.
extern "C" __global__ void treemo_lcs_small(
    const unsigned *H, const long long *text_bit, const int *text_len,
    const long long *ref_bit, const int *ref_len, int *lcs, const int njobs, const int K)
{
    const int job = blockIdx.x * blockDim.x + threadIdx.x;
    if (job >= njobs) return;
    const int n = text_len[job], m = ref_len[job];
    if (n == 0 || m == 0) { lcs[job] = 0; return; }
    const unsigned *T = H + (text_bit[job] >> 5), *P = H + (ref_bit[job] >> 5);
    int r;
    switch (K) {
        case 4: r = lcs_thread<4, true>(T, n, P, m); break;
        case 8: r = lcs_thread<8, true>(T, n, P, m); break;
        case 16: r = lcs_thread<16, true>(T, n, P, m); break;
        case 24: r = lcs_thread<24, true>(T, n, P, m); break;
        case 32: r = lcs_thread<32, true>(T, n, P, m); break;
        case 48: r = lcs_thread<48, true>(T, n, P, m); break;
        case 64: r = lcs_thread<64, true>(T, n, P, m); break;
        case 96: r = lcs_thread<96, true>(T, n, P, m); break;
        default: r = lcs_thread<128, true>(T, n, P, m); break;
    }
    lcs[job] = r;
}
