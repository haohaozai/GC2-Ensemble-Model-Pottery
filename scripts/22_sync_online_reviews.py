"""Bring the decisions made on the online review page back to this computer.

1. Decisions (table `decisions`, public read, no login needed): for each object the newer of the online decision
   and the local one (review/object_review.csv, reviewed_at) is kept. Changed objects are then saved exactly like the
   local review app's Save button: review CSV + full_auto_tags.csv updated, objects marked "not_needed" MOVED to
   does_not_needed/, restored ones moved back, every move logged in review/move_log.csv.
2. Tile (attribute) reviews (table `tile_reviews`, made by web/supabase_tile_reviews.sql): the review_tile category
   and the six-attribute review of an object. The newer of the online and the local record (review/tile_review.json)
   is kept; online records are checked like a local save (glazeprep/review_tile.py) and written with their online
   time and editor; every import is added to review/tile_review_history.jsonl (before / after).

usage: python scripts/22_sync_online_reviews.py [--dry-run]
       (Supabase URL / anon key are read from web/publish_settings.json, written by scripts/21_publish_web.py)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from glazeprep.config import CG2_DATA as BASE  # noqa: E402  (env CG2_DATA / CG2_ROOT or .env; default: next to the project)


def fetch_table(url, key, table):
    """every row of a public table (1000 per request); None when the table does not exist (yet)"""
    rows, offset = [], 0
    while True:
        req = urllib.request.Request(f"{url.rstrip('/')}/rest/v1/{table}?select=*&order=object_id&limit=1000&offset={offset}",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                chunk = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        rows += chunk
        if len(chunk) < 1000:
            return rows
        offset += 1000


def fetch_decisions(url, key):
    return fetch_table(url, key, "decisions") or []


def ts(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.astimezone()   # local review times are local time without zone


def tile_changes(online, tiles, objects):
    """online tile reviews to import: (oid, record, updated_at, updated_by); records kept locally; problems"""
    from glazeprep.review_tile import TileReviewStore
    todo, kept_local, same, problems = [], 0, 0, []
    for t in online:
        oid, rec = t.get("object_id"), t.get("review") or {}
        if oid not in objects:
            problems.append(f"{oid}: unknown object")
            continue
        try:
            new = TileReviewStore.clean(rec)
        except ValueError as exc:
            problems.append(f"{oid}: {exc}")
            continue
        loc = tiles.get(oid)
        if loc:
            try:
                old = TileReviewStore.clean(loc)
            except ValueError:
                old = None
            if old == new and (loc.get("reviewed_at") or "")[:19] == str(rec.get("reviewed_at") or loc.get("reviewed_at") or "")[:19]:
                same += 1
                continue
            t_loc = max([x for x in (ts(loc.get("reviewed_at")), ts(loc.get("category_at")), ts(loc.get("saved_at"))) if x],
                        default=None)
            t_on = ts(t.get("updated_at"))
            if t_loc and t_on and t_loc > t_on:
                kept_local += 1
                continue
        todo.append((oid, rec, t.get("updated_at") or "", t.get("updated_by") or ""))
    return todo, kept_local, same, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=BASE / "full_segmentation_claude_process")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg_p = CODE / "web" / "publish_settings.json"
    cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    if not cfg.get("supabase_url") or not cfg.get("supabase_anon_key"):
        print("No Supabase settings in web/publish_settings.json (publish with Supabase first).")
        return 2
    online = fetch_decisions(cfg["supabase_url"], cfg["supabase_anon_key"])
    print(f"[sync] {len(online)} online decisions", flush=True)

    spec = importlib.util.spec_from_file_location("review_app", CODE / "scripts" / "18_review_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    bonus = BASE / "bonus_image_claude_process"          # bonus_image set: its decisions go to its own review CSV
    extra = [(app.BONUS_SET, bonus.resolve(), (BASE / "others_image").resolve())] if (bonus / "full_auto_tags.csv").exists() else []
    store = app.Store(args.output.resolve(), (BASE / "7k_matched_images" / "7k_matched_images").resolve(), extra)
    changes, unknown, kept_local = {}, [], 0
    for d in online:
        o = store.objects.get(d["object_id"])
        if o is None:
            unknown.append(d["object_id"])
            continue
        if (o["review_status"], o["review_note"]) == (d.get("status") or "", d.get("note") or ""):
            continue
        t_on, t_loc = ts(d.get("updated_at")), ts(o.get("reviewed_at"))
        if t_loc and t_on and t_loc > t_on:
            kept_local += 1
            continue
        changes[d["object_id"]] = {"status": d.get("status") or "", "note": d.get("note") or ""}
    print(f"[sync] to apply: {len(changes)} | local newer, kept: {kept_local} | unknown object ids: {len(unknown)} {unknown[:10]}", flush=True)
    for oid, ch in list(changes.items())[:20]:
        print(f"    {oid}: -> {ch['status'] or 'unreviewed'} {('(' + ch['note'] + ')') if ch['note'] else ''}")

    # ---------------- tile (attribute) reviews
    sys.path.insert(0, str(CODE))
    from glazeprep.review_tile import TileReviewStore
    tiles = TileReviewStore(store.out / "review")
    t_online = fetch_table(cfg["supabase_url"], cfg["supabase_anon_key"], "tile_reviews")
    t_todo = []
    if t_online is None:
        print("[sync] tile reviews: not set up online yet (run web/supabase_tile_reviews.sql in Supabase)", flush=True)
    else:
        t_todo, t_kept, t_same, t_bad = tile_changes(t_online, tiles, store.objects)
        print(f"[sync] tile reviews: {len(t_online)} online | to import: {len(t_todo)} | identical: {t_same} | "
              f"local newer, kept: {t_kept} | refused: {len(t_bad)}", flush=True)
        for oid, rec, when, who in t_todo[:20]:
            n_attr = sum(1 for a in (rec.get("attributes") or {}).values() if a.get("status") not in (None, "", "unknown"))
            print(f"    {oid}: category {rec.get('category') or '-'} · {n_attr} attribute(s) reviewed · by {who or '?'} {when[:16]}")
        for b in t_bad[:10]:
            print(f"    refused {b}")

    if args.dry_run or not (changes or t_todo):
        return 0
    errors = 0
    if changes:
        res = store.save(changes)
        errors += len(res["errors"])
        print(f"[sync] saved: moved to does_not_needed {res['moved_to_does_not_needed']}, restored {res['restored']}, "
              f"errors {len(res['errors'])}", flush=True)
    for oid, rec, when, who in t_todo:
        tiles.import_record(oid, rec, saved_at=when, saved_by=who, source="online")
    if t_todo:
        print(f"[sync] tile reviews imported: {len(t_todo)} -> {tiles.path} (history: {tiles.hist.name})", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
