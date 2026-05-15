/* treemo.c — tree rewriting interpreter
 * Build:  cc -O3 -o treemo treemo.c
 * Usage:  ./treemo <program_bits> <input_bits> [max_step]
 *
 * Program and input are strings of '0'/'1' characters (Dyck words).
 * Max_step defaults to 50.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── resizable byte buffer ──────────────────────────────────────────── */

typedef struct { unsigned char *d; int len, cap; } Buf;

static void bgrow(Buf *b, int need) {
    if (need <= b->cap) return;
    if (!b->cap) b->cap = 128;
    while (b->cap < need) b->cap <<= 1;
    if (!(b->d = realloc(b->d, b->cap))) abort();
}

static void breplace(Buf *b, int pos, int old,
                     const unsigned char *r, int rlen) {
    int tail = b->len - pos - old;
    bgrow(b, pos + rlen + tail);
    memmove(b->d + pos + rlen, b->d + pos + old, tail);
    if (rlen) memcpy(b->d + pos, r, rlen);
    b->len = pos + rlen + tail;
}

/* ── KMP ────────────────────────────────────────────────────────────── */

static int *kmp_build(const unsigned char *p, int n) {
    int *t = malloc(n * sizeof *t);
    t[0] = 0;
    for (int i = 1, k = 0; i < n; i++) {
        while (k && p[k] != p[i]) k = t[k-1];
        t[i] = (p[k] == p[i]) ? ++k : 0;
    }
    return t;
}

/* First match starting at index >= from; returns index or -1. */
static int kfind(const unsigned char *h, int hn,
                 const unsigned char *n, int nn,
                 const int *t, int from) {
    for (int i = from, q = 0; i < hn; i++) {
        while (q && n[q] != h[i]) q = t[q-1];
        if (n[q] == h[i] && ++q == nn) return i + 1 - nn;
    }
    return -1;
}

/* ── Rule ───────────────────────────────────────────────────────────── */

typedef struct {
    unsigned char *pat, *rep;
    int *kmp;
    int plen, rlen, ident;
} Rule;

/* ── Rule extraction ────────────────────────────────────────────────── */

static Rule *extract(const unsigned char *code, int clen, int *nout) {
    /* Build parent list (length clen+1). */
    int plen = clen + 1;
    int *par = malloc(plen * sizeof *par);
    par[0] = -1;
    for (int i = 0, cur = 0, sz = 1; i < clen; i++) {
        if (code[i]) { par[sz] = cur; cur = sz++; }
        else          { cur = par[cur]; par[sz++] = cur; }
    }

    /* Max node id. */
    int mx = 0;
    for (int i = 0; i < plen; i++) if (par[i] > mx) mx = par[i];

    /* Occurrence counts per node; updated as segments are removed. */
    int *cnt = calloc(mx + 1, sizeof *cnt);
    for (int i = 0; i < plen; i++) if (par[i] >= 0) cnt[par[i]]++;

    Rule *rules = NULL;
    int nr = 0, os = 0;
    int *tree = malloc(plen * sizeof *tree);
    memcpy(tree, par, plen * sizeof *tree);
    int tlen = plen;

    for (int node = 0; node <= mx; node++) {
        if (cnt[node] != 4) continue;

        /* Collect the 4 positions in the current (shrinking) tree. */
        int idx[4], k = 0;
        for (int i = 0; i < tlen && k < 4; i++)
            if (tree[i] == node) idx[k++] = i;

        /* Map positions back to original code offsets. */
        int ms = os + idx[0] - 1, me = os + idx[1];
        int rs = os + idx[2] - 1, re = os + idx[3];

        Rule r;
        r.plen = me - ms; r.rlen = re - rs;
        r.pat = malloc(r.plen); memcpy(r.pat, code + ms, r.plen);
        r.rep = malloc(r.rlen); memcpy(r.rep, code + rs, r.rlen);
        r.kmp = kmp_build(r.pat, r.plen);
        r.ident = (r.plen == r.rlen && !memcmp(r.pat, r.rep, r.plen));

        rules = realloc(rules, (nr + 1) * sizeof *rules);
        rules[nr++] = r;

        /* Remove tree[idx[0]..idx[3]] and update counts. */
        for (int i = idx[0]; i <= idx[3]; i++)
            if (tree[i] >= 0) cnt[tree[i]]--;
        int rm = idx[3] + 1 - idx[0];
        memmove(tree + idx[0], tree + idx[3] + 1,
                (tlen - idx[3] - 1) * sizeof *tree);
        tlen -= rm;
        os   += rm;
    }

    free(tree); free(par); free(cnt);
    *nout = nr;
    return rules;
}

/* ── Interpreter ────────────────────────────────────────────────────── */

/*
 * dirty[i] = earliest position in the current state where rule i
 * could possibly match. Everything before dirty[i] has been verified
 * to contain no match for rule i.
 *
 * After firing rule i at position pos:
 *   dirty[j] = max(0, pos + 1 - plen_j)  for all j
 *
 * This is correct because:
 *  - rules 0..i-1 were fully scanned (no match found) before the
 *    replacement; positions before max(0,pos+1-plen_j) are unchanged
 *    and still match-free.
 *  - rules i..n-1 may have new matches only where the replacement
 *    overlaps with the pattern window, i.e. starting at pos+1-plen_j.
 */
static Buf run(Rule *rules, int nr,
               const unsigned char *inp, int ilen, int maxstep) {
    Buf st = {0};
    bgrow(&st, ilen > 64 ? ilen * 2 : 128);
    memcpy(st.d, inp, ilen);
    st.len = ilen;

    int *dirty = calloc(nr, sizeof *dirty);

    for (int step = 0; step < maxstep; step++) {
        int fired = 0;
        for (int i = 0; i < nr; i++) {
            Rule *r = &rules[i];
            int pos = kfind(st.d, st.len, r->pat, r->plen, r->kmp, dirty[i]);
            if (pos < 0) { dirty[i] = st.len; continue; }
            if (r->ident) goto done;

            breplace(&st, pos, r->plen, r->rep, r->rlen);

            for (int j = 0; j < nr; j++) {
                int d = pos + 1 - rules[j].plen;
                if (d < 0) d = 0;
                if (j <= i)
                    dirty[j] = d;          /* scanned this step: advance to dirty zone */
                else if (d < dirty[j])
                    dirty[j] = d;          /* not scanned: only pull back if needed    */
            }
            fired = 1;
            break;
        }
        if (!fired) break;
    }

done:
    free(dirty);
    return st;
}

/* ── Public library API ─────────────────────────────────────────────── */

typedef struct { Rule *rules; int nrules; } Prog;

Prog *treemo_compile(const unsigned char *prog, int plen) {
    Prog *p = malloc(sizeof *p);
    p->rules = extract(prog, plen, &p->nrules);
    return p;
}

/* Returns malloc'd result buffer; caller frees with treemo_free_buf.
 * *out_len is set to the number of bytes written. */
unsigned char *treemo_exec(const Prog *p,
                           const unsigned char *inp, int ilen,
                           int max_step, int *out_len) {
    Buf b = run(p->rules, p->nrules, inp, ilen, max_step);
    *out_len = b.len;
    return b.d;
}

void treemo_free_prog(Prog *p) {
    for (int i = 0; i < p->nrules; i++) {
        free(p->rules[i].pat);
        free(p->rules[i].rep);
        free(p->rules[i].kmp);
    }
    free(p->rules);
    free(p);
}

void treemo_free_buf(unsigned char *buf) { free(buf); }

/* ── CLI ────────────────────────────────────────────────────────────── */

static int parse(const char *s, unsigned char **out) {
    int n = strlen(s);
    *out = malloc(n);
    for (int i = 0; i < n; i++) {
        if      (s[i] == '0') (*out)[i] = 0;
        else if (s[i] == '1') (*out)[i] = 1;
        else { fprintf(stderr, "bad character '%c'\n", s[i]); exit(1); }
    }
    return n;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: treemo <program> <input> [max_step]\n");
        return 1;
    }
    unsigned char *code, *inp;
    int clen = parse(argv[1], &code);
    int ilen = parse(argv[2], &inp);
    int maxstep = argc >= 4 ? atoi(argv[3]) : 50;

    Prog *p = treemo_compile(code, clen);
    int rlen;
    unsigned char *result = treemo_exec(p, inp, ilen, maxstep, &rlen);

    for (int i = 0; i < rlen; i++) putchar('0' + result[i]);
    putchar('\n');

    treemo_free_buf(result);
    treemo_free_prog(p);
    free(code); free(inp);
    return 0;
}
