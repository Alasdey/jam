"""
Build a Graphviz DOT ancestry graph for individuals in an experiment run.

Reads every populations/work_pop_*.json snapshot in a run dir (so individuals
that were later skimmed/capped out of the population still appear), unions
them by id, and emits one DOT digraph: a node per individual (colored by
creation method, labeled with its id and the number of generation snapshots
it survived in) and an edge per parent->child relationship (styled by the
child's creation method).

Usage:
    python -m analysis.ancestry_graph <run_dir> [--out ancestry.dot] [--last-n-gens N] [--render]

    <run_dir> is an experiment dir containing populations/work_pop_*.json
    --last-n-gens limits to individuals first appearing in the last N
        work_pop snapshots (full histories can be huge — millions of nodes).
    --render also invokes `sfdp -Tsvg` to produce ancestry.svg next to the .dot file.
        sfdp (not dot) is used because dot's crossing-minimization layout doesn't
        scale past a few thousand nodes. sfdp is still slow at real scale though
        (~41 min measured on a 770K-individual, 10000-generation run) since it's
        an iterative force-directed layout. For runs bigger than a few thousand
        individuals, use analysis.ancestry_columns instead — it lays nodes out
        directly on a fixed per-generation grid (no iterative layout) and
        finishes the same 770K-node run in under a minute.
"""

import argparse
import glob
import json
import os

METHOD_STYLE = {
    "random": {"color": "red", "edge_style": "solid"},
    "mutate": {"color": "green", "edge_style": "dashed"},
    "crossover": {"color": "orange", "edge_style": "bold"},
    "homoiconic": {"color": "blue", "edge_style": "dotted"},
    None: {"color": "gray", "edge_style": "solid"},
}


def load_individuals(run_dir: str, last_n_gens: int | None = None) -> dict[int, dict]:
    """Union of every individual seen across all work_pop snapshots, keyed by id."""
    pops_dir = os.path.join(run_dir, "populations")
    files = sorted(glob.glob(os.path.join(pops_dir, "work_pop_*.json")))
    if not files:
        raise FileNotFoundError(f"No work_pop_*.json files found in {pops_dir}")
    if last_n_gens is not None:
        files = files[-last_n_gens:]

    individuals: dict[int, dict] = {}
    for f in files:
        gen = int(os.path.basename(f).rsplit("_", 1)[-1].split(".")[0])
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list):
                    continue  # old raw-genome format, no ancestry info
                is_new = obj["id"] not in individuals
                ind = individuals.setdefault(obj["id"], obj)
                ind["n_gens_survived"] = ind.get("n_gens_survived", 0) + 1
                if is_new:
                    ind["first_gen"] = gen
    return individuals


def build_dot(individuals: dict[int, dict]) -> str:
    lines = ["digraph ancestry {", '  rankdir=LR;', '  node [style=filled, fontcolor=white];']

    for ind in individuals.values():
        method = ind.get("method")
        style = METHOD_STYLE.get(method, METHOD_STYLE[None])
        survived = ind.get("n_gens_survived", 1)
        lines.append(f'  {ind["id"]} [label="{ind["id"]}\\n{survived} gen", fillcolor={style["color"]}];')

    for ind in individuals.values():
        method = ind.get("method")
        style = METHOD_STYLE.get(method, METHOD_STYLE[None])
        for parent_id in ind.get("parents", []):
            if parent_id in individuals:
                lines.append(
                    f'  {parent_id} -> {ind["id"]} [color={style["color"]}, style={style["edge_style"]}];'
                )

    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Experiment run dir, e.g. outputs/main/20260616_085157")
    parser.add_argument("--out", default="ancestry.dot")
    parser.add_argument("--last-n-gens", type=int, default=None)
    parser.add_argument(
        "--render", action="store_true",
        help="also run `sfdp -Tsvg` on the output (slow at scale — see module docstring)",
    )
    args = parser.parse_args()

    individuals = load_individuals(args.run_dir, args.last_n_gens)
    dot = build_dot(individuals)

    with open(args.out, "w") as f:
        f.write(dot)
    print(f"Wrote {len(individuals)} individuals to {args.out}")

    if args.render:
        if len(individuals) > 5000:
            print(
                f"Warning: {len(individuals)} individuals — sfdp can take tens of "
                "minutes at this scale. Consider analysis.ancestry_columns instead."
            )
        svg_path = os.path.splitext(args.out)[0] + ".svg"
        os.system(f"sfdp -Tsvg {args.out} -o {svg_path}")
        print(f"Rendered {svg_path}")


if __name__ == "__main__":
    main()
