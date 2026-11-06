// The one-thread-per-child creation kernel that creation/creation.cu replaced,
// kept as its reference: given the same parents, ops and Philox counters, the
// warp kernel must make exactly these children (tests/test_gpu_creation.py).
//
// Treemo genetic operators, one thread per child.
//
// Trees are Dyck words, 1 = open a node, 0 = close it; the root is implicit.
// Each thread works on its child as bytes (one symbol per byte) in its own
// scratch region and follows creation/genetics.py operator for operator: the
// same choices with the same distributions (draws come from Philox, so not
// the same values as Python's random). Parents are read from the packed heap.

#include "philox.cuh"
#include "tapes.cuh"

typedef unsigned char u8;

enum { OP_RANDOM = 0, OP_MUTATE_SUBTREE = 1, OP_MUTATE_LEAF = 2, OP_CROSS_RANDOM_DEPTH = 3, OP_CROSS_DEPTH1 = 4 };

// ── scans ───────────────────────────────────────────────────────────────────

// Position of the k-th 0 in t[from, to), or -1.
__device__ int kth_zero(const u8 *t, int from, int to, int k)
{
    for (int i = from; i < to; ++i)
        if (!t[i] && k-- == 0) return i;
    return -1;
}

__device__ int count_zeros(const u8 *t, int from, int to)
{
    int c = 0;
    for (int i = from; i < to; ++i) c += !t[i];
    return c;
}

// Matching open of the close at q.
__device__ int open_of(const u8 *t, int q)
{
    int d = 0;
    for (int i = q; i >= 0; --i) {
        d += t[i] ? -1 : 1;
        if (d == 0) return i;
    }
    return -1;
}

// Matching close of the open at p.
__device__ int close_of(const u8 *t, int p)
{
    int d = 0;
    for (int i = p;; ++i) {
        d += t[i] ? 1 : -1;
        if (d == 0) return i;
    }
}

// Children of the node [start, end): subtrees opening at relative depth 0 inside it.
__device__ int count_children(const u8 *t, int start, int end)
{
    int d = 0, c = 0;
    for (int i = start + 1; i < end - 1; ++i) {
        if (t[i]) { c += d == 0; ++d; }
        else --d;
    }
    return c;
}

__device__ int kth_child(const u8 *t, int start, int end, int k)
{
    int d = 0;
    for (int i = start + 1; i < end - 1; ++i) {
        if (t[i]) { if (d == 0 && k-- == 0) return i; ++d; }
        else --d;
    }
    return -1;
}

// Subtrees whose open has `depth` nodes already open before it.
__device__ int count_at_depth(const u8 *t, int n, int depth)
{
    int d = 0, c = 0;
    for (int i = 0; i < n; ++i) {
        if (t[i]) { c += d == depth; ++d; }
        else --d;
    }
    return c;
}

__device__ int kth_at_depth(const u8 *t, int n, int depth, int k)
{
    int d = 0;
    for (int i = 0; i < n; ++i) {
        if (t[i]) { if (d == depth && k-- == 0) return i; ++d; }
        else --d;
    }
    return -1;
}

__device__ int max_depth(const u8 *t, int n)
{
    int d = 0, m = 0;
    for (int i = 0; i < n; ++i) {
        if (t[i]) { if (++d > m) m = d; }
        else --d;
    }
    return m;
}

__device__ void reverse(u8 *t, int a, int b)  // t[a, b)
{
    for (--b; a < b; ++a, --b) { const u8 x = t[a]; t[a] = t[b]; t[b] = x; }
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
__device__ void gen_tree(u8 *t, int size, Philox &rng)
{
    const int L = 2 * (size - 1);
    if (L <= 0) return;
    for (int i = 0; i < L; ++i) t[i] = i < size - 1;
    for (int i = L - 1; i > 0; --i) {
        const int j = rng_below(rng, i + 1);
        const u8 x = t[i]; t[i] = t[j]; t[j] = x;
    }
    int h = 0, lowest = 0, at = 0;
    for (int i = 0; i < L; ++i) {
        h += t[i] ? 1 : -1;
        if (h <= lowest) { lowest = h; at = i; }
    }
    const int r = (at + 1) % L;  // rotate left by r
    reverse(t, 0, r); reverse(t, r, L); reverse(t, 0, L);
}

// [1] + gen_tree(size) + [0]: a branch of `size` nodes, 2 * size symbols.
__device__ void gen_branch(u8 *t, int size, Philox &rng)
{
    t[0] = 1;
    gen_tree(t + 1, size, rng);
    t[2 * size - 1] = 0;
}

// ── mutation ────────────────────────────────────────────────────────────────

// One subtree mutation event (creation/genetics.py _mutate_tree_once).
// op: 0 delete, 1 swap, 2 insert, 3 regenerate.
__device__ bool mutate_once(u8 *t, int &n, int cap, int op, Philox &rng)
{
    const int nodes = n / 2;
    if (nodes == 0) return true;
    const int q = kth_zero(t, 0, n, rng_below(rng, nodes));  // nodes in close order, uniformly
    const int start = open_of(t, q), end = q + 1;
    if (op == 0) {  // delete
        for (int i = end; i < n; ++i) t[i - (end - start)] = t[i];
        n -= end - start;
    } else if (op == 2) {  // insert
        // A size from the empirical subtree sizes, the implicit root included.
        const int j = rng_below(rng, nodes + 1);
        int size = nodes + 1;
        if (j < nodes) {
            const int qj = kth_zero(t, 0, n, j);
            size = (qj + 1 - open_of(t, qj)) / 2;
        }
        const int L = 2 * size;
        if (n + L > cap) return false;
        const int nc = count_children(t, start, end);
        const int c = rng_below(rng, nc + 1);
        const int pos = c < nc ? kth_child(t, start, end, c) : end - 1;
        for (int i = n - 1; i >= pos; --i) t[i + L] = t[i];
        gen_branch(t + pos, size, rng);
        n += L;
    } else if (op == 3) {  // regenerate: a random branch of the same node count
        gen_branch(t + start, (end - start) / 2, rng);
    } else {  // swap a child branch of the node with one of a disjoint node
        // Eligible nodes lie entirely before or after the node: every close
        // before `start`, and the closes after `end` of nodes opened after it.
        const int before = count_zeros(t, 0, start);
        int after = 0;
        for (int i = end, d = 0; i < n; ++i) {
            if (t[i]) ++d;
            else if (d > 0) { ++after; --d; }
        }
        if (before + after == 0) return true;
        int r = rng_below(rng, before + after);
        int q2 = -1;
        if (r < before) q2 = kth_zero(t, 0, start, r);
        else {
            r -= before;
            for (int i = end, d = 0; i < n && q2 < 0; ++i) {
                if (t[i]) ++d;
                else if (d > 0) { if (r-- == 0) q2 = i; --d; }
            }
        }
        const int o2 = open_of(t, q2), e2 = q2 + 1;
        int la, lb, ra, rb;
        const int nc = count_children(t, start, end);
        const int c1 = rng_below(rng, nc ? nc : 1);
        if (nc) { la = kth_child(t, start, end, c1); lb = close_of(t, la) + 1; }
        else { la = lb = end - 1; }
        const int nc2 = count_children(t, o2, e2);
        const int c2 = rng_below(rng, nc2 ? nc2 : 1);
        if (nc2) { ra = kth_child(t, o2, e2, c2); rb = close_of(t, ra) + 1; }
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
__device__ bool mutate_subtree(u8 *t, int &n, int cap, double rate, Philox &rng)
{
    int events = 0;
    for (int i = 0, nodes = n / 2; i < nodes; ++i) events += rng_uniform(rng) < rate;
    for (int e = 0; e < events; ++e)
        if (!mutate_once(t, n, cap, rng_below(rng, 4), rng)) return false;
    return true;
}

// mutate_tree_leaf: with probability rate, contract a random leaf [1, 0]
// (half the time, if the tree has more than one node) or expand it to [1, 1, 0, 0].
__device__ bool mutate_leaf(u8 *t, int &n, int cap, double rate, Philox &rng)
{
    if (n == 0 || rng_uniform(rng) > rate) return true;
    int leaves = 0;
    for (int i = 0; i + 1 < n; ++i) leaves += t[i] && !t[i + 1];
    if (!leaves) return true;
    int k = rng_below(rng, leaves), idx = -1;
    for (int i = 0; i + 1 < n && idx < 0; ++i)
        if (t[i] && !t[i + 1] && k-- == 0) idx = i;
    if (rng_uniform(rng) < 0.5 && n > 2) {
        for (int i = idx + 2; i < n; ++i) t[i - 2] = t[i];
        n -= 2;
    } else {
        if (n + 2 > cap) return false;
        for (int i = n - 1; i > idx; --i) t[i + 2] = t[i];
        t[idx + 1] = 1;
        t[idx + 2] = 0;
        n += 2;
    }
    return true;
}

// ── crossover ───────────────────────────────────────────────────────────────

// child = a[:sa0] + b[sb0:sb1] + a[sa1:], or a when no subtree qualifies.
__device__ int splice(u8 *child, const u8 *a, int na, const u8 *b, int sa0, int sa1, int sb0, int sb1)
{
    int n = 0;
    for (int i = 0; i < sa0; ++i) child[n++] = a[i];
    for (int i = sb0; i < sb1; ++i) child[n++] = b[i];
    for (int i = sa1; i < na; ++i) child[n++] = a[i];
    return n;
}

__device__ int copy_of(u8 *child, const u8 *a, int na)
{
    for (int i = 0; i < na; ++i) child[i] = a[i];
    return na;
}

// crossover_tree_random_depth: up to 5 draws of a depth in [1, min max depth];
// the first with subtrees at that depth in both parents swaps one of b's in for one of a's.
__device__ int cross_random_depth(u8 *child, const u8 *a, int na, const u8 *b, int nb, Philox &rng)
{
    const int da = max_depth(a, na), db = max_depth(b, nb);
    if (da < 1 || db < 1) return copy_of(child, a, na);
    for (int attempt = 0; attempt < 5; ++attempt) {
        const int depth = 1 + rng_below(rng, min(da, db));
        const int ca = count_at_depth(a, na, depth), cb = count_at_depth(b, nb, depth);
        if (ca && cb) {
            const int sa = kth_at_depth(a, na, depth, rng_below(rng, ca));
            const int sb = kth_at_depth(b, nb, depth, rng_below(rng, cb));
            return splice(child, a, na, b, sa, close_of(a, sa) + 1, sb, close_of(b, sb) + 1);
        }
    }
    return copy_of(child, a, na);
}

// crossover_tree_depth1: one of b's depth-1 subtrees in for one of a's.
__device__ int cross_depth1(u8 *child, const u8 *a, int na, const u8 *b, int nb, Philox &rng)
{
    const int ca = count_at_depth(a, na, 1), cb = count_at_depth(b, nb, 1);
    if (!ca || !cb) return copy_of(child, a, na);
    const int sa = kth_at_depth(a, na, 1, rng_below(rng, ca));
    const int sb = kth_at_depth(b, nb, 1, rng_below(rng, cb));
    return splice(child, a, na, b, sa, close_of(a, sa) + 1, sb, close_of(b, sb) + 1);
}

// ── kernels ─────────────────────────────────────────────────────────────────

__device__ void unpack(u8 *dst, const unsigned *heap, long long word, int n)
{
    for (int i = 0; i < n; ++i) dst[i] = (heap[word + (i >> 5)] >> (i & 31)) & 1u;
}

// Child k: op[k] on parents a[k], b[k] (survivor indices; unused ones -1),
// written as bytes at scratch[at[k]] within cap[k] bytes; the parents' bytes
// go after the child's region. out_len[k] = the child's length, or -1 when it
// would exceed cap[k] (the caller retries with more room: element[k] fixes
// the draws, so the retry makes the same child).
extern "C" __global__ void create(
    const unsigned *heap, const long long *off, const int *len,
    const int n, const int *op, const long long *a, const long long *b, const unsigned *element,
    u8 *scratch, const long long *at, const int *cap, int *out_len,
    const int tree_size, const double rate,
    const unsigned k0, const unsigned k1, const unsigned gen, const unsigned call)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    Philox rng;
    rng_init(rng, k0, k1, element[k], gen, call);
    u8 *t = scratch + at[k];
    int m = 0;
    bool fits = true;
    switch (op[k]) {
    case OP_RANDOM:
        gen_tree(t, tree_size, rng);
        m = tree_size > 1 ? 2 * (tree_size - 1) : 0;
        break;
    case OP_MUTATE_SUBTREE:
    case OP_MUTATE_LEAF:
        m = len[a[k]];
        unpack(t, heap, off[a[k]], m);
        fits = op[k] == OP_MUTATE_SUBTREE ? mutate_subtree(t, m, cap[k], rate, rng)
                                          : mutate_leaf(t, m, cap[k], rate, rng);
        break;
    default: {
        const int na = len[a[k]], nb = len[b[k]];
        u8 *pa = t + cap[k], *pb = pa + na;
        unpack(pa, heap, off[a[k]], na);
        unpack(pb, heap, off[b[k]], nb);
        m = op[k] == OP_CROSS_RANDOM_DEPTH ? cross_random_depth(t, pa, na, pb, nb, rng)
                                           : cross_depth1(t, pa, na, pb, nb, rng);
    }
    }
    out_len[k] = fits ? m : -1;
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
