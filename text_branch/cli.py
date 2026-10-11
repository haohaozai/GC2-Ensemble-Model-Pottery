"""Command line: ``python -m text_branch <command> --help``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from .config import load_config


def _config(args: argparse.Namespace):
    config = load_config(args.config)
    if args.output_dir:
        config.paths.output_dir = str(Path(args.output_dir).resolve())
    return config


def _encoder(config, args: argparse.Namespace):
    from .encoder import TextEncoder

    if getattr(args, "device", None):
        config.encoder.device = args.device
    if getattr(args, "batch_size", None):
        config.encoder.batch_size = args.batch_size
    return TextEncoder(config.encoder)


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .pipeline import run_preprocess

    config = _config(args)
    recipes, report = run_preprocess(config, source=args.source)
    print(f"Cleaned {report['records_total']} rows ({report['records_in_scope']} in scope, "
          f"{report['text_available_records']} available for text retrieval).")
    print(f"Wrote {config.output_dir / 'recipes_clean.jsonl'}")
    print(f"Wrote {config.output_dir / 'data_quality_report.md'}")
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .pipeline import run_embed

    config = _config(args)
    if args.no_cache:
        config.encoder.use_cache = False
    embeddings = run_embed(config, encoder=_encoder(config, args))
    print(f"Saved {embeddings.vectors.shape[0]} x {embeddings.vectors.shape[1]} baseline embeddings "
          f"to {config.output_dir / 'baseline'}")
    return 0


def cmd_project(args: argparse.Namespace) -> int:
    from .pipeline import run_project

    target = run_project(_config(args), args.checkpoint, allow_untrained=args.allow_untrained)
    print(f"Wrote projected embeddings to {target}")
    if args.allow_untrained:
        print("WARNING: output may come from an UNTRAINED projection; see its manifest.json.")
    return 0


def _index(args: argparse.Namespace):
    from .search import RecipeSearchIndex

    config = _config(args)
    return RecipeSearchIndex.from_outputs(config, encoder=_encoder(config, args),
                                          checkpoint=getattr(args, "checkpoint", None))


def cmd_search(args: argparse.Namespace) -> int:
    from .search import SearchFilters

    index = _index(args)
    filters = SearchFilters(cone=args.cone, atmospheres=args.atmosphere, surfaces=args.surface,
                            transparencies=args.transparency, include_unknown=args.include_unknown,
                            cone_match=args.cone_match)
    results = index.search(args.query, top_k=args.top_k, filters=filters)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        return 0
    print(f"[{index.space} space] {index.count_matching(filters)} candidates after filters")
    for r in results:
        flags = f"  flags: {', '.join(r.review_flags)}" if r.review_flags else ""
        print(f"{r.rank:>3}. {r.score:.3f}  #{r.recipe_id}  {r.name or '(no name)'}{flags}\n       {r.description}")
    if not results:
        print("No recipes match the filters.")
    return 0


def cmd_lookup(args: argparse.Namespace) -> int:
    from .search import NameIndex
    from .pipeline import load_preprocessed

    recipes, _ = load_preprocessed(_config(args))
    matches = NameIndex(recipes).lookup(args.name)
    if args.json:
        print(json.dumps([dataclasses.asdict(m) for m in matches], indent=2, ensure_ascii=False))
        return 0
    for m in matches:
        print(f"#{m.recipe_id}  [{m.match_type} on {m.matched_field}] {m.matched_text!r}  ->  "
              f"{m.description or '(no usable semantic tags)'}")
    if not matches:
        print("No recipe name or alias matches.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    index = _index(args)
    text = run_demo(index, top_k=args.top_k)
    out = Path(args.out) if args.out else _config(args).output_dir / "demo_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m text_branch", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config (default: configs/text_branch.yaml)")
    common.add_argument("--output-dir", help="override paths.output_dir")
    enc = argparse.ArgumentParser(add_help=False)
    enc.add_argument("--device", help="cpu, cuda, cuda:0, mps (default: auto)")
    enc.add_argument("--batch-size", type=int)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", parents=[common], help="clean CSV -> recipes_clean.jsonl + quality report")
    p.add_argument("--source", help="override paths.source_csv")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("embed", parents=[common, enc], help="pretrained text embeddings -> baseline/")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("project", parents=[common], help="apply a projection checkpoint -> projected/")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--allow-untrained", action="store_true",
                   help="dev only: export an untrained checkpoint to projected_UNTRAINED_DEV_ONLY/")
    p.set_defaults(func=cmd_project)

    p = sub.add_parser("search", parents=[common, enc], help="free-text search with explicit filters")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cone", help='cone or range, e.g. "6", "06", "5 - 7"')
    p.add_argument("--cone-match", choices=["within", "overlap"], default="within")
    p.add_argument("--atmosphere", action="append", help="repeatable; matches if ANY is listed")
    p.add_argument("--surface", action="append", help="repeatable")
    p.add_argument("--transparency", action="append", help="repeatable")
    p.add_argument("--include-unknown", action="store_true", help="keep recipes with unknown filtered fields")
    p.add_argument("--checkpoint", help="TRAINED projection checkpoint for shared-space search")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("lookup", parents=[common], help="exact/normalised name or alias lookup")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser("demo", parents=[common, enc], help="run the retrieval demonstration")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out", help="markdown output (default: <output_dir>/demo_results.md)")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    from .projection import UntrainedProjectionError

    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, UntrainedProjectionError) as exc:  # incl. incompatible checkpoints
        print(f"error: {exc}", file=sys.stderr)
        return 2
