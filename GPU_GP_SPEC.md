# Fully-GPU Treemo genetic programming — prototype spec

Standalone prototype, outside jam: one fixed configuration of jam's evolution
loop (Treemo, current creation operators, dedupe + lexicase, zero-sum
reward built from Treemo runs and equality tests) running entirely on one
NVIDIA GPU. Only genomes, births and parentage come back to the host. The goal
is to measure how fast a fully GPU-resident generation is; it bypasses jam's
code, conventions and file formats.

Everything needed is in this file: jam's GPU code will be rolled back.
Measurements cited are from an RTX 5060 Laptop GPU (26 SMs, 8 GB, sm_120) on
2026-10-05/06.

---

## 1. Fixed configuration

| Parameter | Value (jam default) |
|---|---|
| Interpreter | Treemo, `max_step = 50`, `pass_mode = False`, `first_mode = False` |
| Random tree size | `tree_size = 100` nodes (Dyck word of 198 bits) |
| Per generation | `n_random = 100` random trees, `n_offspring = 900` bred |
| Genetics | `mutation_rate = 0.05`, `crossover_prob = 0.8`, `homoiconic_prob = 0.3`; tree mutation `subtree`, tree crossover `random_depth`, homoiconic `output` |
| Pre-selection | max length 10,000 bits, then exact dedupe |
| Selection | lexicase, `n_accepted = 200` (also try 1,000) |
| Reward | a zero-sum expression of the DSL in §6 |
| Generations | `n_iter` (e.g. 10,000) |
| RNG | one seed; GPU counter-based RNG (Philox). Runs are reproducible within the prototype, **not** draw-for-draw identical to jam's Python RNG |

Every value is a constant or a command-line argument of the prototype.

---

## 2. Treemo semantics (must be bit-exact with `interpreters/treemo_c/treemo.c`)

Tapes are sequences of 0/1 symbols (Dyck words: 1 = open, 0 = close).

**Rule extraction.** Programs are forests under an implicit root. Visit nodes
in pre-order; a node with **exactly two children** is a rule node: its rule is
(pattern = first child's subtree, replacement = second child's subtree), and
its descendants are not visited. The implicit root counts: if the program has
exactly two top-level trees, the only rule is (first tree, second tree).
Equivalently: rules are the two-child nodes with no two-child ancestor, in
pre-order of their opening bit. A rule is an *identity* rule when pattern ==
replacement. An empty program has no rules. Genomes are always balanced 0/1
words here (all operators preserve that), so no invalid-program path is needed.

Sequential extraction (one lane per program is fine):
1. One pass with a stack: for each open position `o`, its matching close
   `match[o]` and its number of children.
2. Count top-level trees. If exactly 2: one rule `([0, m0], [m0+1, end))`
   where `m0 = match[0]`; done.
3. Else scan positions left to right: at an open `o` with exactly 2 children,
   emit rule (pattern = `[o+1, match[o+1]]`, replacement =
   `[match[o+1]+1, match[o]-1]`) and jump to `match[o]+1`; at any other open,
   step to `o+1`; at a close, step on.

Store each rule as (pattern bit offset, pattern length, replacement bit
offset, replacement length) relative to the program's first bit, plus the
identity flag. Patterns are read straight out of the packed program; nothing
is copied.

**Execution** (`run(code, input)`), with rules `0..nr-1`:

```
state = input; rule = 0; no_fire = 0; fired = False; dirty[j] = 0 for all j
while nr > 0 and step < max_step:
    pos = leftmost occurrence of pattern[rule] in state starting at >= dirty[rule]
    if pos found:
        if identity[rule]: stop
        state[pos : pos + plen] = replacement      # splice, length may change
        step += 1; no_fire = 0; fired = True
        for every rule j:
            d = max(0, pos + 1 - plen_j)
            if j == rule or (pass_mode and first_mode and j < rule) or d < dirty[j]:
                dirty[j] = d
        if pass_mode:
            rule = 0 if first_mode else (rule + 1) % nr; fired = False
    else:
        dirty[rule] = len(state)
        if not fired:
            no_fire += 1
            if no_fire >= nr: stop
            rule = (rule + 1) % nr
        else:
            no_fire = 0; fired = False
            rule = 0 if first_mode else (rule + 1) % nr
return state
```

Matching is plain substring search on the flat bit sequence. `dirty[]` is
an optimisation that does not change results, but it is essential for speed:
on evolved genomes it cuts scanning 40–60×, and most scans hit within a few
dozen positions.

**Exact state bound.** No firing grows the tape by more than the program's
largest rule growth `g = max(0, max(rlen - plen))`, so
`len(output) <= len(input) + max_step * g`. Allocate exactly that: nothing is
ever truncated or retried. Nested runs compose the bound.

---

## 3. One generation (exact order)

State on device: survivors (genomes, ids), their survivor × survivor payoff
matrix (int8, zero-sum), the RNG counter.

1. **Create** 100 random trees, then 900 offspring from the survivors (§4).
   Generation 0 (no survivors): 100 + 900 random trees.
2. **Pre-select** on `survivors + new`, in that order:
   a. drop genomes longer than 10,000 bits;
   b. dedupe: keep the first occurrence of each distinct genome (so an
      existing survivor beats an identical newborn, and earlier newborns
      beat later ones).
3. **Payoff extension** (zero-sum): keep `old × old`; compute `old × new`
   and the upper triangle of `new × new`; set the transposes as negatives;
   the diagonal is 0.
4. **Lexicase** on the full `(old + new)` square matrix down to `n_accepted`
   survivors (§8). Reindex the kept payoff rows and columns.
5. **Record** newborns that survived (id, birth generation, method, parent
   ids, genome) to the device archive (§9).

---

## 4. Creation (statistically identical to `creation/`)

**Random tree** `gen_tree(n)`: shuffle `n-1` ups and `n-1` downs uniformly;
rotate the sequence to start just after the **last** position where the
running height reaches its minimum (cycle lemma); map up to 1 and down to 0.
`gen_tree(1)` is the empty word. It produces a uniform plane tree with `n`
nodes, root implicit.

**Offspring**, each one independently:

```
if U < 0.8 and n_survivors >= 2:
    pa, pb = two distinct survivors, uniformly
    child = None
    if U < 0.3: child = run(pa, pb)                     # homoiconic "output"
                if child is empty: child = None
    if child is None: child = crossover(pa, pb); method = "crossover"
    else: method = "homoiconic"
    parents = [pa, pb]
else:
    p = one survivor, uniformly; child = mutate(p); method = "mutate"; parents = [p]
```

The homoiconic child is the **output tape** of the existing Treemo kernel,
so it never leaves the device.

**Crossover `random_depth`(a, b).** Let `d_a`, `d_b` be the maximum nesting
(the largest number of simultaneously open nodes). If either is < 1, return
`a`. Up to 5 attempts: pick `d` uniformly in `[1, min(d_a, d_b)]`; collect
the subtree spans in `a` and in `b` whose opening bit has exactly `d` nodes
already open before it (so top-level trees are never picked, and when `d`
equals the maximum nesting no span qualifies and the attempt fails); if both
are non-empty, pick one of each uniformly and return
`a[:sa.start] + b[sb] + a[sa.end:]`. After 5 failed attempts, return `a`.

**Mutation `subtree`(t, rate).** The number of events is
`Binomial(len(t)//2, rate)`, drawn up front from the original tree. Each
event acts on the current tree:

- `spans` = every explicit node as `(open, close + 1)`, **ordered by close
  position** (the order of a stack-based scan). If empty, the event does nothing.
- Pick a node uniformly from `spans`; operation uniform over
  {delete, swap, insert, regenerate}:
  - **delete**: remove the node's span.
  - **regenerate**: replace the span by `[1] + gen_tree(size) + [0]` with
    `size = (end - start) // 2`, a random branch of the same node count.
  - **insert**: `size` uniform over the multiset `{(b - a)//2 for spans} ∪
    {len(t)//2 + 1}` (empirical subtree sizes plus the implicit root); build a
    branch of that size; choose the insertion position uniformly among the
    start of each child of the picked node and its last position (`end - 1`,
    i.e. as last child); insert there.
  - **swap**: `eligible` = spans entirely before or after the picked node
    (no overlap). If none, nothing happens. Otherwise pick `other` uniformly;
    pick one child branch of the node (or the empty branch at `end - 1` if it
    has none) and one of `other` (same rule); exchange the two branches.
- Inserted nodes may become targets but do not add events. The implicit root
  is never edited (an empty tree stays empty).

One warp per offspring is plenty (≈1,000 offspring per generation); all
operators are O(length) scans plus one splice, as in the Treemo kernel.

---

## 5. Pre-selection on device

- Max length: a flag per genome.
- Dedupe: 64-bit hash per genome (e.g. xxHash/FNV over packed words, with
  the length mixed in), sort by (hash, index), and confirm equality word by
  word inside equal-hash runs, so the result is exact. Keep the smallest index
  of each class (old before new).

---

## 6. Reward DSL, evaluated in situ

**Terms:** `A`, `B`, `run(t, u)` with `t`, `u` terms, nested freely; an
output can be a program or an input.
**Values:** `eq(t, u)` ∈ {0, 1} (tapes exactly equal), integer constants,
`+`, `-`, `*`.
**Reward:** an integer expression `f(A, B)`; the payoff of row `a` against
column `b` is `f(a, b)`. The user guarantees zero-sum
(`f(a, b) == -f(b, a)`); the safest way to write one is antisymmetric by
construction: `f(A, B) = g(A, B) - g(B, A)`.

Examples:

```
self_on_other   = eq(run(A, B), A) - eq(run(B, A), B)                 # A survives being run on B
keep_opponent   = eq(run(A, B), B) - eq(run(B, A), A)
second_order    = eq(run(run(A, B), A), A) - eq(run(run(B, A), B), B)
```

(`blind`, `1 - eq(run(B, A), run(A, B))`, is symmetric, not zero-sum, so it
is out of scope here.)

**Compilation (host, once):** parse the expression; substitute `A` and `B`
for the swapped operand order; deduplicate common sub-terms; get a straight
list of run instructions in dependency order (`r_k = run(x, y)`, with `x`,
`y` ∈ {A, B, r_j for j < k}), a list of equality tests `(x, y)`, and an
integer formula over the tests. Generate CUDA source with these constants
baked in (NVRTC or CuPy `RawModule` compile per reward): no interpretation
overhead on the device.

**Evaluation:** one warp per unordered pair `{i, j}` (i < j, plus the
old × new rectangle). The warp evaluates `f(A=i, B=j)` (its `-` gives
`f(j, i)`), running each `r_k` with the Treemo kernel into per-warp scratch
tapes, keeping them alive only while later instructions or tests need them,
then compares tapes word by word (equal lengths first). Outputs never leave
scratch; only the int8 payoff is written.

Programs that are outputs (`run(run(A, B), A)`) need rule extraction in the
warp (§2) before running; that is sequential but O(length).

**Memory:** a persistent kernel with a fixed number of resident warps
(e.g. 4 × SMs × warps per SM), each pulling pairs from an atomic counter
and owning one scratch area sized from the exact bounds of §2 for the
longest genome allowed (10,000 bits) and the compiled expression's nesting.
If a pair needs more than its scratch (a giant outlier), flag it and
re-evaluate flagged pairs in a second pass with larger scratch. Memory is
then fixed by the warp count, not by the population.

---

## 7. Payoff storage

`int8` square matrix over the current population; at most `(200 + 1000)^2`
bytes per generation (1.4 MB) at main sizes, `N^2` bytes for full-matrix
experiments (10,000 genomes: 100 MB). It stays on the device; only row sums
or nothing come back.

---

## 8. Lexicase (exact rules of `selection/lexicase.py`)

Select `n_accepted` distinct rows without replacement. For each pick:
candidates = all rows not yet selected; cases = all columns (selected rows
stay available as cases), in a fresh uniform random order; for each case in
order, stop if one candidate remains, else keep the candidates with the
**exact maximum** payoff on that case; finally pick uniformly among the
remaining candidates. Return the selected rows sorted.

GPU: picks are sequential, but each is a parallel filter. One thread block,
candidates as a bitmask in shared memory (1,200 rows = 38 words), a fresh
permutation per pick (random keys plus sort, or Fisher-Yates by one thread;
the filter usually stops after a few cases, so generating the permutation
lazily case by case is enough), a block-wide max and mask per case.
Payoffs are tiny integers, so rows with identical payoff vectors can be
grouped first, as jam does, if this ever shows up in the profile.

---

## 9. What comes back to the host

A device archive of every newborn that survived its birth generation (jam's
default logging): `id, birth generation, method (random|mutate|crossover|homoiconic),
parent ids, genome` (packed bits + length), plus the final survivor ids. Copy
it at the end of the run, or every K generations as a checkpoint. Optional:
survivor ids per generation (a few KB per generation), to rebuild the
population at any generation.

---

## 10. GPU design

Choices that were measured to work in jam's CUDA backend:

- **Bit-packed tapes**, LSB first: bit `j` of word `w` is symbol `32w + j`.
  Every tape is word-aligned, followed by a zero word, and the store begins
  with a zero word, so a 32-bit window may start up to 32 bits before or end
  32 bits after any tape (`window()` in Appendix A).
- **One warp per Treemo execution.** A pattern search tests 1,024 start
  positions per step: lane L owns 32 consecutive starts and drops them a
  pattern bit at a time; a ballot picks the leftmost survivor. A rewrite
  shifts the tail in place: destination words are rewritten 32 at a time,
  all lanes read before any writes, walking top-down when the tape grows and
  bottom-up when it shrinks (Appendix A).
- **Same `dirty[]` watermarks as C**, so results are identical by construction.
- **Exact capacity per execution** from §2 (never truncate, never retry).
- **Keep work program-major** (all jobs of one program adjacent); sorting
  jobs longest-first was measured slower.
- **CuPy `RawModule`** to compile and launch; kernels are cached on disk
  by CuPy after the first compile (about 2 s with many template instances).

What did **not** work:

- **One thread per execution** on byte tapes (another attempt): correct but
  slower than one CPU core on evolved genomes (250–450 µs vs 18 µs), because
  of warp divergence and serial byte moves over tapes thousands of symbols long.
- **Host-side per-matchup Python** (scoring scalar rewards matchup by matchup):
  materialising 10⁶ outputs as Python lists per generation costs more than
  the whole GPU payoff. This prototype keeps everything on the device for
  that reason.
- **A generic host-driven batch engine** (jam's last version: dedupe and
  memoisation in numpy around each batched operation): correct and
  reward-agnostic, but ~60% of its time at scale was host numpy, not GPU.
  The compiled DSL of §6 removes that layer.

Numbers to beat (laptop GPU, jam's generic engine, LCS rewards, evolved
genomes of ~1,000 bits):

| Measure | Value |
|---|---|
| Treemo execution, kernel time | 0.05–0.1 µs per execution (C: 18–70 µs on one core) |
| Payoff per generation, main preset | 0.4–1.3 s (CPU: 7–9 s for blind; quine_pressure out of reach) |
| Full N × N matrix | N = 3,000: 11–12 s · N = 5,000: 31 s · N = 10,000: 137 s |

With equality-only rewards and no host layer, expect payoff to be dominated
by executions: about `pairs × runs per pair × 0.05–0.1 µs`, e.g. ~0.1 s per
generation at main sizes and ~5–10 s for a 10,000 × 10,000 matrix with two
runs per pair (estimates, to be measured).

---

## 11. Validation

1. **Treemo kernel vs C**, bit-exact on random trees, forests and real evolved
   genomes, for all four `pass_mode`/`first_mode` combinations and
   `max_step ∈ {0, 7, 50}`; include programs whose rules grow tapes by
   hundreds of times (outputs reached 485,000 bits in jam).
2. **Rule extraction** vs `tree_to_rules` in `interpreters/treemo_py/treemo.py`
   (keep a copy) on thousands of programs, including forests of 1–4 trees and
   the empty program.
3. **Reward DSL** vs a CPU reference (the same expression evaluated with the C
   interpreter, one pair at a time) on random populations; also check
   `f(a, b) == -f(b, a)` on a sample and report violations.
4. **Dedupe** vs exact Python comparison.
5. **Lexicase** vs `selection/lexicase.py` with the same case permutations
   and tie draws fed to both.
6. **Creation**: every child is a balanced 0/1 word; method frequencies,
   child length and depth distributions match the Python operators over
   ~10⁵ draws (two-sample tests); spot-check each operator on hand-made trees.
7. **End to end**: no exact comparison with jam is possible (different RNG);
   compare trajectories statistically (population length, method survival
   rates, payoff row-sum distributions) over a few seeds.

---

## 12. Open choices

- Whether to keep `n_accepted = 200` or 1,000 (lexicase cost grows with it).
- Archive granularity: survivors per generation or only at the end.
- Hash width for dedupe (64-bit plus exact confirmation is enough).
- Scratch size per warp versus the overflow second pass (tune on real runs).

---

## Appendix A — validated CUDA for the Treemo core

Bit-exact against the C interpreter on ~10⁴ pairs in all modes.
`H` is the tape store, `T` an execution's state (with a guard word at `T[-1]`
and one after the end), `rules[k] = (pattern offset, pattern length,
replacement offset, replacement length)` relative to the program's bit
offset `base`.

```cuda
#define FULL 0xffffffffu

// 32 bits of A starting at bit offset `off` (off may be as low as -32).
__device__ __forceinline__ unsigned window(const unsigned *A, long long off)
{
    const long long w = off >> 5;  // arithmetic shift: floor division
    return __funnelshift_r(A[w], A[w + 1], (unsigned)(off & 31));
}

// Leftmost start s in [from, n - m] where T[s, s + m) equals the m-bit pattern
// at H bit `pat`, or -1. Lane L tests the 32 starts of word (blk >> 5) + L.
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
// All lanes read before any writes; growing tapes are walked top-down and
// shrinking ones bottom-up, so no read sees a word an earlier round moved.
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

// The execution loop of §2, one warp; returns the final length, or -1 if the
// state would exceed `cap` (cannot happen with the exact bound).
__device__ int warp_run(unsigned *T, int n, int cap, const unsigned *H, long long base,
                        const int4 *R, const unsigned char *Id, int nr, int *D,
                        int max_step, int pass_mode, int first_mode, int lane)
{
    for (int j = lane; j < nr; j += 32) D[j] = 0;
    __syncwarp();
    int rule = 0, no_fire = 0, fired = 0;
    for (int step = 0; nr && step < max_step; ) {
        const int4 r = R[rule];
        const int pos = warp_find(T, n, H, base + r.x, r.y, D[rule], lane);
        if (pos >= 0) {
            if (Id[rule]) break;
            if (n + r.w - r.y > cap) return -1;
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
    return n;
}
```

Equality of two tapes in a warp: lengths equal, then XOR word by word (mask
the last partial word) and `__any_sync` the differences.

## Appendix B — host-side rule extraction (vectorised numpy, validated)

Useful as the CPU reference for test 2: depth via cumulative sum of `2b - 1`;
each bit's level is `min(depth before, depth after)`; a stable sort by
(program, level) pairs every open with its close; a node's children are the
level+1 bits inside it, two per child; rule nodes are two-child nodes not
strictly inside another two-child node (a difference array over the
intervals), minus all of them when the implicit root has exactly two
children (then the root is the only rule).
