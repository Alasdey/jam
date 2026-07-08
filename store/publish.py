"""
Publish a run's population to the store.

Usage:
    python -m store.publish <run_dir> --label <label> [--gen N] [--store DIR]
"""

import argparse

from store.population_store import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run output dir, e.g. outputs/main/20260702_140000")
    parser.add_argument("--label", required=True, help="Human-readable methodology label")
    parser.add_argument("--gen", type=int, default=None, help="Generation to publish (default: latest)")
    parser.add_argument("--store", default="outputs/store/populations")
    args = parser.parse_args()

    pop_id = publish(args.store, args.label, args.run_dir, args.gen)
    print(pop_id)


if __name__ == "__main__":
    main()
