"""
Fast ancestry visualization: one column per generation, nodes stacked
vertically within their column, parent->child edges drawn as straight lines.

Unlike analysis.ancestry_graph (which feeds a DOT file to graphviz), this
script computes node positions directly with a fixed grid — no force-directed
or crossing-minimization layout — so it stays fast even at hundreds of
thousands of nodes (graphviz's sfdp/dot can take minutes to hours at that
scale; this is plain O(n + e) string formatting). Edges are drawn first and
nodes are drawn on top, so individuals are always visible above the arrows
pointing through their column.

Usage:
    python -m analysis.ancestry_columns <run_dir> [--out ancestry.svg] [--last-n-gens N]
    python -m analysis.ancestry_columns <run_dir> --split-components --out-dir analysis/out/components

    --split-components: the ancestry "graph" is rarely one connected blob —
        individuals with no surviving descendants, or whose entire lineage was
        skimmed away, end up in their own disconnected component. Splitting by
        connected component (union-find over parent/child edges) and rendering
        each one as its own SVG produces files proportional to that
        component's size instead of one SVG sized for the whole run.
"""

import argparse
import os

from analysis.ancestry_graph import METHOD_STYLE, load_individuals

COL_WIDTH = 24
ROW_HEIGHT = 16
NODE_R = 6
MARGIN = 40


def connected_components(individuals: dict[int, dict]) -> list[list[dict]]:
    parent = {i: i for i in individuals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for ind in individuals.values():
        for p in ind.get("parents", []):
            if p in parent:
                union(ind["id"], p)

    groups: dict[int, list[dict]] = {}
    for i, ind in individuals.items():
        groups.setdefault(find(i), []).append(ind)
    return sorted(groups.values(), key=len, reverse=True)


def build_svg(individuals: dict[int, dict]) -> str:
    by_gen: dict[int, list[dict]] = {}
    for ind in individuals.values():
        by_gen.setdefault(ind["first_gen"], []).append(ind)

    pos: dict[int, tuple[float, float]] = {}
    gens = sorted(by_gen)
    gen_col = {gen: i for i, gen in enumerate(gens)}
    max_rows = 0
    for gen, inds in by_gen.items():
        inds.sort(key=lambda ind: ind["id"])
        max_rows = max(max_rows, len(inds))
        x = MARGIN + gen_col[gen] * COL_WIDTH
        for row, ind in enumerate(inds):
            y = MARGIN + row * ROW_HEIGHT
            pos[ind["id"]] = (x, y)

    width = MARGIN * 2 + len(gens) * COL_WIDTH
    height = MARGIN * 2 + max_rows * ROW_HEIGHT

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]

    # edges first, so nodes render on top of (above) the arrows
    for ind in individuals.values():
        if ind["id"] not in pos:
            continue
        method = ind.get("method")
        style = METHOD_STYLE.get(method, METHOD_STYLE[None])
        cx, cy = pos[ind["id"]]
        dash = {"solid": "", "dashed": '5,3', "bold": "", "dotted": "2,2"}[style["edge_style"]]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        width_attr = ' stroke-width="2"' if style["edge_style"] == "bold" else ' stroke-width="1"'
        for parent_id in ind.get("parents", []):
            if parent_id not in pos:
                continue
            px, py = pos[parent_id]
            parts.append(
                f'<line x1="{px}" y1="{py}" x2="{cx}" y2="{cy}" '
                f'stroke="{style["color"]}"{width_attr}{dash_attr}/>'
            )

    # nodes on top
    for ind in individuals.values():
        if ind["id"] not in pos:
            continue
        method = ind.get("method")
        style = METHOD_STYLE.get(method, METHOD_STYLE[None])
        cx, cy = pos[ind["id"]]
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{NODE_R}" fill="{style["color"]}"/>')
        parts.append(
            f'<text x="{cx}" y="{cy + 3}" font-size="6" text-anchor="middle" fill="white">'
            f'{ind["id"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Experiment run dir, e.g. outputs/main/20260616_085157")
    parser.add_argument("--out", default="ancestry.svg")
    parser.add_argument("--last-n-gens", type=int, default=None)
    parser.add_argument("--split-components", action="store_true")
    parser.add_argument(
        "--out-dir", default=None,
        help="default: analysis/out/components_<run_dir basename>",
    )
    parser.add_argument(
        "--min-component-size", type=int, default=2,
        help="skip components smaller than this when splitting (default 2 — drops lone individuals with no edges)",
    )
    args = parser.parse_args()

    individuals = load_individuals(args.run_dir, args.last_n_gens)

    if not args.split_components:
        svg = build_svg(individuals)
        with open(args.out, "w") as f:
            f.write(svg)
        print(f"Wrote {len(individuals)} individuals to {args.out}")
        return

    out_dir = args.out_dir or os.path.join(
        "analysis/out", f"components_{os.path.basename(os.path.normpath(args.run_dir))}"
    )
    components = connected_components(individuals)
    os.makedirs(out_dir, exist_ok=True)
    n_written = n_skipped_individuals = n_skipped_components = 0
    for rank, comp in enumerate(components):
        if len(comp) < args.min_component_size:
            n_skipped_components += 1
            n_skipped_individuals += len(comp)
            continue
        comp_individuals = {ind["id"]: ind for ind in comp}
        svg = build_svg(comp_individuals)
        path = os.path.join(out_dir, f"component_{rank:05d}_n{len(comp)}.svg")
        with open(path, "w") as f:
            f.write(svg)
        n_written += 1
    print(
        f"Wrote {n_written} component SVGs to {out_dir} "
        f"(skipped {n_skipped_components} components / {n_skipped_individuals} individuals "
        f"below --min-component-size={args.min_component_size})"
    )


if __name__ == "__main__":
    main()
