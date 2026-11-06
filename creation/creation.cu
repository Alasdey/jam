// Treemo genetic operators, one warp per child (creation/treemo.py drives them).
//
// Trees are Dyck words, 1 = open a node, 0 = close it; the root is implicit.
// Each warp works on its child as bytes (one symbol per byte) in its own
// scratch region and follows the CPU operators (tests/reference/creation_cpu.py)
// choice for choice. The 32 lanes share every scan, shift and copy, so a
// child's latency no longer grows with one thread walking its whole tree for
// every mutation event (small online batches wait for their slowest child).
//
// Every lane makes the same random draws, from the child's Philox stream read
// by word position (Stream below), in the order the one-thread-per-child
// kernel made them: given the same parents and counters, the children are
// exactly those of tests/reference/creation_thread.cu (the tests compare them).
//
// Control flow is warp-uniform: every lane runs every loop that holds a
// ballot or a shuffle. Lanes write only between __syncwarp()s.
//
// Adding an operator: a __device__ function over (t, &n, cap, Stream&) that
// returns false when the child would exceed cap, a new op code dispatched in
// create(), and its name in creation/treemo.py.

#include "philox.cuh"
#include "tapes.cuh"

typedef unsigned char u8;

enum { OP_RANDOM = 0, OP_MUTATE_SUBTREE = 1, OP_MUTATE_LEAF = 2, OP_CROSS_RANDOM_DEPTH = 3, OP_CROSS_DEPTH1 = 4 };

#define LANE ((int)(threadIdx.x & 31))
#define NONE (1 << 30)

// ── random draws ────────────────────────────────────────────────────────────

// A child's stream: word w is word w % 4 of Philox block (element, w / 4, gen,
// call), the words rng_u32 (philox.cuh) hands out one at a time. Reading by
// position lets lanes draw ahead in parallel.
struct Stream {
    unsigned k0, k1, element, gen, call;
    unsigned long long pos;  // next word
};

__device__ __forceinline__ unsigned word_at(const Stream &s, unsigned long long w)
{
    unsigned c[4] = {s.element, (unsigned)(w >> 2), s.gen, s.call};
    philox4x32_10(c, s.k0, s.k1);
    return c[w & 3];
}

__device__ __forceinline__ unsigned next_u32(Stream &s) { return word_at(s, s.pos++); }

__device__ __forceinline__ double to_uniform(unsigned x, unsigned y)  // rng_uniform's 53 bits
{
    return ((x >> 5) * 67108864.0 + (y >> 6)) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ double next_uniform(Stream &s)
{
    const unsigned x = next_u32(s);
    return to_uniform(x, next_u32(s));
}

// rng_below: uniform in [0, n), Lemire's rejection.
__device__ __forceinline__ unsigned below(Stream &s, unsigned n)
{
    unsigned long long m = (unsigned long long)next_u32(s) * n;
    unsigned lo = (unsigned)m;
    if (lo < n) {
        const unsigned threshold = (0u - n) % n;
        while (lo < threshold) {
            m = (unsigned long long)next_u32(s) * n;
            lo = (unsigned)m;
        }
    }
    return (unsigned)(m >> 32);
}

// ── warp primitives ─────────────────────────────────────────────────────────

__device__ __forceinline__ int incl_sum(int v)
{
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, v, o);
        if (LANE >= o) v += y;
    }
    return v;
}

__device__ __forceinline__ int incl_min(int v)
{
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, v, o);
        if (LANE >= o) v = min(v, y);
    }
    return v;
}

__device__ __forceinline__ int warp_max(int v)
{
#pragma unroll
    for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

__device__ __forceinline__ int warp_min(int v)
{
#pragma unroll
    for (int o = 16; o; o >>= 1) v = min(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

// The lane of the k-th (from 0) set bit of mask; mask has more than k bits.
__device__ __forceinline__ int lane_of_kth(unsigned mask, int k)
{
    const bool hit = ((mask >> LANE) & 1u) && __popc(mask & ((1u << LANE) - 1u)) == k;
    return __ffs(__ballot_sync(FULL, hit)) - 1;
}

// ── scans (each lane takes one symbol of every 32) ─────────────────────────

// Position of the k-th 0 in t[from, to), or -1.
__device__ int kth_zero(const u8 *t, int from, int to, int k)
{
    for (int base = from; base < to; base += 32) {
        const int i = base + LANE;
        const unsigned m = __ballot_sync(FULL, i < to && !t[i]);
        const int c = __popc(m);
        if (k < c) return base + lane_of_kth(m, k);
        k -= c;
    }
    return -1;
}

__device__ int count_zeros(const u8 *t, int from, int to)
{
    int c = 0;
    for (int base = from; base < to; base += 32) {
        const int i = base + LANE;
        c += __popc(__ballot_sync(FULL, i < to && !t[i]));
    }
    return c;
}

// Matching open of the close at q.
__device__ int open_of(const u8 *t, int q)
{
    int d = 0;
    for (int top = q; top >= 0; top -= 32) {
        const int i = top - LANE;
        const int v = i >= 0 ? (t[i] ? -1 : 1) : 0;
        const int s = d + incl_sum(v);
        const unsigned m = __ballot_sync(FULL, i >= 0 && s == 0);
        if (m) return top - (__ffs(m) - 1);
        d = __shfl_sync(FULL, s, 31);
    }
    return -1;
}

// Matching close of the open at p, in t[0, n).
__device__ int close_of(const u8 *t, int p, int n)
{
    int d = 0;
    for (int base = p; base < n; base += 32) {
        const int i = base + LANE;
        const int v = i < n ? (t[i] ? 1 : -1) : 0;
        const int s = d + incl_sum(v);
        const unsigned m = __ballot_sync(FULL, i < n && s == 0);
        if (m) return base + __ffs(m) - 1;
        d = __shfl_sync(FULL, s, 31);
    }
    return -1;
}

// Opens in t[from, to) with `depth` nodes open before them (counted from
// `from`): how many when k < 0, else the position of the k-th (-1 if none).
__device__ int opens_at(const u8 *t, int from, int to, int depth, int k)
{
    int d = 0, c = 0;
    for (int base = from; base < to; base += 32) {
        const int i = base + LANE;
        const bool in = i < to;
        const bool one = in && t[i];
        const int v = in ? (one ? 1 : -1) : 0;
        const int s = d + incl_sum(v);  // depth after i
        const unsigned m = __ballot_sync(FULL, one && s - v == depth);
        const int cnt = __popc(m);
        if (k >= 0) {
            if (k < cnt) return base + lane_of_kth(m, k);
            k -= cnt;
        }
        c += cnt;
        d = __shfl_sync(FULL, s, 31);
    }
    return k >= 0 ? -1 : c;
}

// Children of the node [start, end): subtrees opening at relative depth 0 inside it.
__device__ int count_children(const u8 *t, int start, int end) { return opens_at(t, start + 1, end - 1, 0, -1); }
__device__ int kth_child(const u8 *t, int start, int end, int k) { return opens_at(t, start + 1, end - 1, 0, k); }

// Subtrees whose open has `depth` nodes already open before it.
__device__ int count_at_depth(const u8 *t, int n, int depth) { return opens_at(t, 0, n, depth, -1); }
__device__ int kth_at_depth(const u8 *t, int n, int depth, int k) { return opens_at(t, 0, n, depth, k); }

__device__ int max_depth(const u8 *t, int n)
{
    int d = 0, m = 0;
    for (int base = 0; base < n; base += 32) {
        const int i = base + LANE;
        const int v = i < n ? (t[i] ? 1 : -1) : 0;
        const int s = d + incl_sum(v);
        m = max(m, warp_max(i < n ? s : 0));
        d = __shfl_sync(FULL, s, 31);
    }
    return m;
}

// The closes of t[from, n) whose node opened at or after `from`: a depth d
// starts at 0, an open raises it, a close counts and lowers it only while
// d > 0. With H the running sum before a symbol and M the least H so far
// (0 included), that d is H - M, so a close counts where H > M. How many when
// k < 0, else the position of the k-th (-1 if none).
__device__ int closes_after(const u8 *t, int from, int n, int k)
{
    int h = 0, lo = 0, c = 0;
    for (int base = from; base < n; base += 32) {
        const int i = base + LANE;
        const bool in = i < n;
        const bool one = in && t[i];
        const int v = in ? (one ? 1 : -1) : 0;
        const int s = h + incl_sum(v);
        const int H = s - v;
        const int M = min(lo, incl_min(H));
        const unsigned m = __ballot_sync(FULL, in && !one && H > M);
        const int cnt = __popc(m);
        if (k >= 0) {
            if (k < cnt) return base + lane_of_kth(m, k);
            k -= cnt;
        }
        c += cnt;
        h = __shfl_sync(FULL, s, 31);
        lo = __shfl_sync(FULL, M, 31);
    }
    return k >= 0 ? -1 : c;
}

// Leaves [1, 0] of t[0, n): how many when k < 0, else the position of the k-th.
__device__ int leaves(const u8 *t, int n, int k)
{
    int c = 0;
    for (int base = 0; base < n - 1; base += 32) {
        const int i = base + LANE;
        const unsigned m = __ballot_sync(FULL, i < n - 1 && t[i] && !t[i + 1]);
        const int cnt = __popc(m);
        if (k >= 0) {
            if (k < cnt) return base + lane_of_kth(m, k);
            k -= cnt;
        }
        c += cnt;
    }
    return k >= 0 ? -1 : c;
}

// ── moves ───────────────────────────────────────────────────────────────────

// t[i - by] = t[i] for i in [from, to), by > 0: a chunk is read whole before
// it is written, and writes stay below the next chunk.
__device__ void move_left(u8 *t, int from, int to, int by)
{
    for (int base = from; base < to; base += 32) {
        const int i = base + LANE;
        const u8 x = i < to ? t[i] : 0;
        __syncwarp();
        if (i < to) t[i - by] = x;
        __syncwarp();
    }
}

// t[i + by] = t[i] for i in [from, to), by > 0, from the top down.
__device__ void move_right(u8 *t, int from, int to, int by)
{
    for (int top = to; top > from; top -= 32) {
        const int i = top - 1 - LANE;
        const u8 x = i >= from ? t[i] : 0;
        __syncwarp();
        if (i >= from) t[i + by] = x;
        __syncwarp();
    }
}

__device__ void copy(u8 *dst, const u8 *src, int n)
{
    for (int i = LANE; i < n; i += 32) dst[i] = src[i];
    __syncwarp();
}

__device__ void reverse(u8 *t, int a, int b)  // t[a, b)
{
    const int half = (b - a) >> 1;
    for (int j = LANE; j < half; j += 32) {
        const u8 x = t[a + j];
        t[a + j] = t[b - 1 - j];
        t[b - 1 - j] = x;
    }
    __syncwarp();
}

// t[a, d) = t[c, d) + t[b, c) + t[a, b), in place.
__device__ void block_swap(u8 *t, int a, int b, int c, int d)
{
    reverse(t, a, b); reverse(t, b, c); reverse(t, c, d); reverse(t, a, d);
}

// ── generation ──────────────────────────────────────────────────────────────

// gen_tree(size): a random tree of `size` nodes, root implicit, as
// 2 * (size - 1) symbols at t. Shuffle size - 1 opens and closes
// (Fisher-Yates, as random.shuffle), then rotate to start just after the last
// minimum of the running height. Same distribution as the CPU gen_tree, which
// is not uniform over plane trees: a tree's weight grows with the length of
// its first top-level subtree.
__device__ void gen_tree(u8 *t, int size, Stream &s)
{
    const int L = 2 * (size - 1);
    if (L <= 0) return;
    for (int i = LANE; i < L; i += 32) t[i] = i < size - 1;
    __syncwarp();
    // Swap i with below(i + 1) for i = L - 1 down to 1. Lane l drafts the draw
    // of swap top - l from word pos + l; the drafts hold up to the first one
    // Lemire's rule rejects, which is then drawn again in full, as below() does.
    int i = L - 1;
    while (i > 0) {
        const int mine = i - LANE;
        unsigned j = 0;
        bool stop = mine <= 0;
        if (!stop) {
            const unsigned n = mine + 1;
            const unsigned long long m = (unsigned long long)word_at(s, s.pos + LANE) * n;
            const unsigned lo = (unsigned)m;
            stop = lo < n && lo < (0u - n) % n;
            j = (unsigned)(m >> 32);
        }
        const unsigned stops = __ballot_sync(FULL, stop);
        const int f = stops ? __ffs(stops) - 1 : 32;
        for (int l = 0; l < f; ++l) {
            const int jj = __shfl_sync(FULL, (int)j, l);
            if (LANE == 0) {
                const int a = i - l;
                const u8 x = t[a]; t[a] = t[jj]; t[jj] = x;
            }
        }
        __syncwarp();
        s.pos += f;
        i -= f;
        if (f < 32 && i > 0) {  // lane f's draft was rejected
            const int jj = below(s, i + 1);
            if (LANE == 0) { const u8 x = t[i]; t[i] = t[jj]; t[jj] = x; }
            __syncwarp();
            --i;
        }
    }
    // The last position of the lowest running height (the height ends at 0,
    // so the lowest is at most 0, as the CPU loop's starting `lowest`).
    int h = 0, lowest = NONE, at = 0;
    for (int base = 0; base < L; base += 32) {
        const int p = base + LANE;
        const int v = p < L ? (t[p] ? 1 : -1) : 0;
        const int hp = h + incl_sum(v);
        const int cmin = warp_min(p < L ? hp : NONE);
        if (cmin <= lowest) {
            const unsigned m = __ballot_sync(FULL, p < L && hp == cmin);
            lowest = cmin;
            at = base + 31 - __clz(m);
        }
        h = __shfl_sync(FULL, hp, 31);
    }
    const int r = (at + 1) % L;  // rotate left by r
    reverse(t, 0, r); reverse(t, r, L); reverse(t, 0, L);
}

// [1] + gen_tree(size) + [0]: a branch of `size` nodes, 2 * size symbols.
__device__ void gen_branch(u8 *t, int size, Stream &s)
{
    if (LANE == 0) t[0] = 1;
    __syncwarp();
    gen_tree(t + 1, size, s);
    if (LANE == 0) t[2 * size - 1] = 0;
    __syncwarp();
}

// ── mutation ────────────────────────────────────────────────────────────────

// One subtree mutation event (creation/genetics.py _mutate_tree_once).
// op: 0 delete, 1 swap, 2 insert, 3 regenerate.
__device__ bool mutate_once(u8 *t, int &n, int cap, int op, Stream &s)
{
    const int nodes = n / 2;
    if (nodes == 0) return true;
    const int q = kth_zero(t, 0, n, below(s, nodes));  // nodes in close order, uniformly
    const int start = open_of(t, q), end = q + 1;
    if (op == 0) {  // delete
        move_left(t, end, n, end - start);
        n -= end - start;
    } else if (op == 2) {  // insert
        // A size from the empirical subtree sizes, the implicit root included.
        const int j = below(s, nodes + 1);
        int size = nodes + 1;
        if (j < nodes) {
            const int qj = kth_zero(t, 0, n, j);
            size = (qj + 1 - open_of(t, qj)) / 2;
        }
        const int L = 2 * size;
        if (n + L > cap) return false;
        const int nc = count_children(t, start, end);
        const int c = below(s, nc + 1);
        const int pos = c < nc ? kth_child(t, start, end, c) : end - 1;
        move_right(t, pos, n, L);
        gen_branch(t + pos, size, s);
        n += L;
    } else if (op == 3) {  // regenerate: a random branch of the same node count
        gen_branch(t + start, (end - start) / 2, s);
    } else {  // swap a child branch of the node with one of a disjoint node
        // Eligible nodes lie entirely before or after the node: every close
        // before `start`, and the closes after `end` of nodes opened after it.
        const int before = count_zeros(t, 0, start);
        const int after = closes_after(t, end, n, -1);
        if (before + after == 0) return true;
        int r = below(s, before + after);
        const int q2 = r < before ? kth_zero(t, 0, start, r) : closes_after(t, end, n, r - before);
        const int o2 = open_of(t, q2), e2 = q2 + 1;
        int la, lb, ra, rb;
        const int nc = count_children(t, start, end);
        const int c1 = below(s, nc ? nc : 1);
        if (nc) { la = kth_child(t, start, end, c1); lb = close_of(t, la, n) + 1; }
        else { la = lb = end - 1; }
        const int nc2 = count_children(t, o2, e2);
        const int c2 = below(s, nc2 ? nc2 : 1);
        if (nc2) { ra = kth_child(t, o2, e2, c2); rb = close_of(t, ra, n) + 1; }
        else { ra = rb = e2 - 1; }
        if (ra < la || (ra == la && rb < lb)) {  // sorted([left, right])
            int x = la; la = ra; ra = x;
            x = lb; lb = rb; rb = x;
        }
        block_swap(t, la, lb, ra, rb);
    }
    return true;
}

// mutate_tree_subtree: one Bernoulli(rate) trial per original node sets the
// number of events; each picks one of the four operations uniformly.
__device__ bool mutate_subtree(u8 *t, int &n, int cap, double rate, Stream &s)
{
    const int nodes = n / 2;
    int events = 0;
    for (int base = 0; base < nodes; base += 32) {  // trial i reads words pos + 2i, pos + 2i + 1
        const int i = base + LANE;
        bool hit = false;
        if (i < nodes) {
            const unsigned long long w = s.pos + 2ull * i;
            hit = to_uniform(word_at(s, w), word_at(s, w + 1)) < rate;
        }
        events += __popc(__ballot_sync(FULL, hit));
    }
    s.pos += 2ull * nodes;
    for (int e = 0; e < events; ++e)
        if (!mutate_once(t, n, cap, below(s, 4), s)) return false;
    return true;
}

// mutate_tree_leaf: with probability rate, contract a random leaf [1, 0]
// (half the time, if the tree has more than one node) or expand it to [1, 1, 0, 0].
__device__ bool mutate_leaf(u8 *t, int &n, int cap, double rate, Stream &s)
{
    if (n == 0 || next_uniform(s) > rate) return true;
    const int count = leaves(t, n, -1);
    if (!count) return true;
    const int idx = leaves(t, n, below(s, count));
    if (next_uniform(s) < 0.5 && n > 2) {
        move_left(t, idx + 2, n, 2);
        n -= 2;
    } else {
        if (n + 2 > cap) return false;
        move_right(t, idx + 1, n, 2);
        if (LANE == 0) { t[idx + 1] = 1; t[idx + 2] = 0; }
        __syncwarp();
        n += 2;
    }
    return true;
}

// ── crossover ───────────────────────────────────────────────────────────────

// child = a[:sa0] + b[sb0:sb1] + a[sa1:].
__device__ int splice(u8 *child, const u8 *a, int na, const u8 *b, int sa0, int sa1, int sb0, int sb1)
{
    copy(child, a, sa0);
    copy(child + sa0, b + sb0, sb1 - sb0);
    copy(child + sa0 + sb1 - sb0, a + sa1, na - sa1);
    return sa0 + (sb1 - sb0) + (na - sa1);
}

__device__ int copy_of(u8 *child, const u8 *a, int na)
{
    copy(child, a, na);
    return na;
}

// crossover_tree_random_depth: up to 5 draws of a depth in [1, min max depth];
// the first with subtrees at that depth in both parents swaps one of b's in for one of a's.
__device__ int cross_random_depth(u8 *child, const u8 *a, int na, const u8 *b, int nb, Stream &s)
{
    const int da = max_depth(a, na), db = max_depth(b, nb);
    if (da < 1 || db < 1) return copy_of(child, a, na);
    for (int attempt = 0; attempt < 5; ++attempt) {
        const int depth = 1 + below(s, min(da, db));
        const int ca = count_at_depth(a, na, depth), cb = count_at_depth(b, nb, depth);
        if (ca && cb) {
            const int sa = kth_at_depth(a, na, depth, below(s, ca));
            const int sb = kth_at_depth(b, nb, depth, below(s, cb));
            return splice(child, a, na, b, sa, close_of(a, sa, na) + 1, sb, close_of(b, sb, nb) + 1);
        }
    }
    return copy_of(child, a, na);
}

// crossover_tree_depth1: one of b's depth-1 subtrees in for one of a's.
__device__ int cross_depth1(u8 *child, const u8 *a, int na, const u8 *b, int nb, Stream &s)
{
    const int ca = count_at_depth(a, na, 1), cb = count_at_depth(b, nb, 1);
    if (!ca || !cb) return copy_of(child, a, na);
    const int sa = kth_at_depth(a, na, 1, below(s, ca));
    const int sb = kth_at_depth(b, nb, 1, below(s, cb));
    return splice(child, a, na, b, sa, close_of(a, sa, na) + 1, sb, close_of(b, sb, nb) + 1);
}

// ── kernels ─────────────────────────────────────────────────────────────────

__device__ void unpack(u8 *dst, const unsigned *heap, long long word, int n)
{
    for (int i = LANE; i < n; i += 32) dst[i] = (heap[word + (i >> 5)] >> (i & 31)) & 1u;
    __syncwarp();
}

// Child k (one warp): op[k] on parents a[k], b[k] (survivor indices; unused
// ones -1), written as bytes at scratch[at[k]] within cap[k] bytes; the
// parents' bytes go after the child's region. out_len[k] = the child's
// length, or -1 when it would exceed cap[k] (the caller retries with more
// room: element[k] fixes the draws, so the retry makes the same child).
extern "C" __global__ void create(
    const unsigned *heap, const long long *off, const int *len,
    const int n, const int *op, const long long *a, const long long *b, const unsigned *element,
    u8 *scratch, const long long *at, const int *cap, int *out_len,
    const int tree_size, const double rate,
    const unsigned k0, const unsigned k1, const unsigned gen, const unsigned call)
{
    const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (k >= n) return;  // whole warps
    Stream s = {k0, k1, element[k], gen, call, 0ull};
    u8 *t = scratch + at[k];
    int m = 0;
    bool fits = true;
    switch (op[k]) {
    case OP_RANDOM:
        gen_tree(t, tree_size, s);
        m = tree_size > 1 ? 2 * (tree_size - 1) : 0;
        break;
    case OP_MUTATE_SUBTREE:
    case OP_MUTATE_LEAF:
        m = len[a[k]];
        unpack(t, heap, off[a[k]], m);
        fits = op[k] == OP_MUTATE_SUBTREE ? mutate_subtree(t, m, cap[k], rate, s)
                                          : mutate_leaf(t, m, cap[k], rate, s);
        break;
    default: {
        const int na = len[a[k]], nb = len[b[k]];
        u8 *pa = t + cap[k], *pb = pa + na;
        unpack(pa, heap, off[a[k]], na);
        unpack(pb, heap, off[b[k]], nb);
        m = op[k] == OP_CROSS_RANDOM_DEPTH ? cross_random_depth(t, pa, na, pb, nb, s)
                                           : cross_depth1(t, pa, na, pb, nb, s);
    }
    }
    if (LANE == 0) out_len[k] = fits ? m : -1;
}

// Pack child k's bytes into a heap at word dst[k], with its trailing zero word. One warp per child.
extern "C" __global__ void pack_children(const u8 *scratch, const long long *at, const int *len,
                                         unsigned *heap, const long long *dst, const int n)
{
    const int job = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (job >= n) return;
    const u8 *t = scratch + at[job];
    const int m = len[job];
    const int words = (m + 31) >> 5;
    for (int w = 0; w <= words; ++w) {
        const int i = (w << 5) + lane;
        const unsigned word = __ballot_sync(FULL, i < m && t[i]);
        if (lane == 0) heap[dst[job] + w] = word;
    }
}
