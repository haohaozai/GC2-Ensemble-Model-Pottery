"""Rebuild the golden results of the glaze parity test from the original notebook (after a new model arrives).

Runs the retrieval cells of the notebook - the code cells up to the one defining retrieve_glazes, without the matplotlib
gallery and refusing any cell that would start training (a top-level main() call) - on the artifacts of the model ZIP, for the
example queries of the notebook (query_args = dict(...)), and writes
multimodal_experiments/tests/fixtures/glaze_notebook_golden.json. Then:  python -m unittest tests.test_glaze_retrieval
This executes the notebook's code (its retrieval cells only): use it with a ZIP from the team, not from elsewhere.
usage: python scripts/build_glaze_golden.py "path\\to\\test_similarity.zip"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "multimodal_experiments"))
from mm.glaze_retrieval import FILES, model_version  # noqa: E402  (also points the model cache at <CG2>\model_cache)

OUT = CODE / "multimodal_experiments" / "tests" / "fixtures" / "glaze_notebook_golden.json"
GALLERY = "# Display retrieved glaze photographs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-per-cluster", type=int, default=2)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, zipfile.ZipFile(args.zip) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/") and not n.endswith("/")]
        wanted = list(FILES.values()) + [".ipynb"]
        for n in names:
            base = n.rsplit("/", 1)[-1]
            if base in FILES.values() or base.endswith(".ipynb"):
                (Path(tmp) / base).write_bytes(z.read(n))
        nbs = sorted(Path(tmp).glob("*.ipynb"))
        if len(nbs) > 1:                                    # e.g. training.ipynb + retrieval.ipynb: use the inference-only one
            nbs = [p for p in nbs if "retrieval" in p.stem.lower()] or nbs
        if len(nbs) != 1 or not all((Path(tmp) / f).is_file() for f in FILES.values()):
            print(f"ERROR: the ZIP needs one notebook (or one named retrieval*.ipynb) and {', '.join(FILES.values())} (looked for {wanted})")
            return 1
        cells = [("".join(c["source"])) for c in json.loads(nbs[0].read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        upto = next((i for i, s in enumerate(cells) if "def retrieve_glazes" in s), None)
        if upto is None:
            print("ERROR: no cell defines retrieve_glazes")
            return 1
        queries = [m.group(1) for s in cells[upto + 1:] for m in re.finditer(r"query_args\s*=\s*(dict\(.*?\))", s, re.S)]
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            ns: dict = {"__name__": "retrieval_notebook"}
            for i, src in enumerate(cells[:upto + 1]):
                if re.search(r"^\s*main\(\)", src, re.M):
                    print(f"ERROR: code cell {i} calls main() (training) - not run")
                    return 1
                if i == upto:
                    src = src.split(GALLERY)[0]
                exec(compile(src, f"{nbs[0].name}:cell{i}", "exec"), ns)  # noqa: S102  (the notebook's retrieval cells)
            out = []
            for qs in queries:
                q = eval(qs, {"dict": dict})  # noqa: S307  (a dict(...) literal of keyword constants from the notebook)
                q = {k: v for k, v in q.items() if k in ("color", "surface", "cone", "query_hex")}
                best = ns["retrieve_glazes"](**q, top_k=args.top_k, diversify=False)
                expl = ns["retrieve_glazes"](**q, top_k=args.top_k, diversify=True, max_per_cluster=args.max_per_cluster)
                r9 = lambda s: [round(float(x), 9) for x in s]  # noqa: E731
                out.append({"query": {**q, "top_k": args.top_k},
                            "best": {"recipe_id": [int(x) for x in best.recipe_id], "final_score": r9(best.final_score),
                                     "latent_similarity": r9(best.latent_similarity), "delta_e_2000": r9(best.delta_e_2000)},
                            "explore": {"recipe_id": [int(x) for x in expl.recipe_id], "final_score": r9(expl.final_score),
                                        "kmeans_cluster": [int(x) for x in expl.kmeans_cluster]}})
                print(f"{q['color']:>12}: best {out[-1]['best']['recipe_id'][:3]}... explore {out[-1]['explore']['recipe_id'][:3]}...")
        finally:
            os.chdir(cwd)
            if hasattr(ns.get("vectors"), "close"):            # the notebook keeps its .npz open (Windows cannot delete it)
                ns["vectors"].close()
        ver = model_version(Path(tmp))
    OUT.write_text(json.dumps({"_about": f"The original notebook ({nbs[0].name}, retrieval cells unmodified) run on the artifacts of "
                                         f"{Path(args.zip).name} by scripts/build_glaze_golden.py. Regenerate when a new model arrives.",
                               "model_version": ver, "top_k": args.top_k, "max_per_cluster": args.max_per_cluster, "queries": out}, indent=1),
                   encoding="utf-8")
    print(f"{len(out)} queries, model {ver} -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
