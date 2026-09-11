"""Headless CPP recommendation runner — no Streamlit, just run and get results.

Runs the same algorithm the web app uses (`recommend_for_algae`) and writes the
ranked results to a CSV you can open in VS Code, Excel, or anywhere. The UI is
irrelevant here: this is the algorithm spitting out results as fast as it can.

Usage (from the repo root, with the venv active or via .venv/bin/python):

    python run_recommend.py                         # defaults (pVEC anchor, top 25)
    python run_recommend.py --anchor pVEC-R6A       # different anchor preset
    python run_recommend.py --top-k 50              # more results
    python run_recommend.py --no-low-toxicity       # turn off the toxicity filter
    python run_recommend.py --out my_results.csv    # choose the output file

Press F5 in VS Code (with the "Run CPP recommendation" config) to run it there.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cpp_ai.pipeline import recommend_for_algae


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPP recommendation algorithm and save results.")
    parser.add_argument("--anchor", default="pVEC", help="Anchor preset (pVEC, pVEC-R6A, ClWOX) or a raw sequence.")
    parser.add_argument("--top-k", type=int, default=25, help="How many ranked peptides to keep.")
    parser.add_argument("--collapse-families", type=float, default=0.7,
                        help="Group near-duplicate peptides above this identity (0-1). Use 0 to disable.")
    parser.add_argument("--require-encodable", action="store_true", default=True,
                        help="Only keep peptides that can be cloned as an mCherry fusion (default: on).")
    parser.add_argument("--no-require-encodable", dest="require_encodable", action="store_false")
    parser.add_argument("--low-toxicity", action="store_true", default=True,
                        help="Filter out high predicted-toxicity peptides (default: on).")
    parser.add_argument("--no-low-toxicity", dest="low_toxicity", action="store_false")
    parser.add_argument("--rank-by", default="algae_v2",
                        choices=["blend", "algae_fit", "algae_v2"],
                        help="Ranking recipe: 'algae_v2' (new, data-derived, default), "
                             "'algae_fit' (older membrane blend), 'blend' (anchor resemblance).")
    parser.add_argument("--out", default="results.csv", help="Output CSV path.")
    args = parser.parse_args()

    print(f"Running recommendation for anchor={args.anchor!r} ...")
    t0 = time.time()
    rec = recommend_for_algae(
        anchor=args.anchor,
        top_k=args.top_k,
        rank_by=args.rank_by,
        require_encodable=args.require_encodable,
        low_toxicity=args.low_toxicity,
        collapse_families=(args.collapse_families or None),
    )
    elapsed = time.time() - t0

    df = rec.to_dataframe()
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    print(f"\nDone in {elapsed:.1f}s — {len(rec.profiles)} peptides.")
    print(f"Results written to: {out_path.resolve()}\n")
    # Print a readable summary table to the terminal too.
    print(rec.to_markdown(top_k=min(args.top_k, 15)))


if __name__ == "__main__":
    main()
