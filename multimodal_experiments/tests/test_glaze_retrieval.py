"""Tests of card G (colour match model): parity with the original notebook, cone / surface / HEX handling, the two result
lists, the result fields, missing or broken model files and the three HTTP endpoints (real request handler of app/server.py).
run (from multimodal_experiments):  ..\\.venv\\Scripts\\python.exe -m unittest tests.test_glaze_retrieval -v
Needs the installed model (INSTALL_GLAZE_MODEL.bat). Tests of names / original cones are skipped without the 7k CSVs.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP), str(EXP / "app")]

import numpy as np  # noqa: E402

from mm import glaze_reference as GR  # noqa: E402
from mm import glaze_retrieval as G  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "glaze_notebook_golden.json").read_text(encoding="utf-8"))
MODEL_DIR = G.resolve(G.read_config()["model_dir"])
_R: dict = {}


def setUpModule():
    if not all((MODEL_DIR / f).is_file() for f in G.FILES.values()):
        raise unittest.SkipTest(f"glaze model not installed in {MODEL_DIR} (run INSTALL_GLAZE_MODEL.bat)")


def retriever(mode: str) -> G.GlazeRetriever:
    """one retriever per cone scoring mode, sharing the 7k reference data."""
    if mode not in _R:
        ref = _R.setdefault("ref", GR.Reference())
        _R[mode] = G.GlazeRetriever(cfg={**G.read_config(), "cone_scoring": mode}, reference=ref)
    return _R[mode]


def has_7k() -> bool:
    return GR.find_7k("7k_matched_all_data*.csv") is not None


def is_low_fire(cone) -> bool:
    r = G.cone_range_orton(cone) if cone else None
    return bool(r) and r[0] < 0


# ---------------------------------------------------------------- the notebook's results
class ParityWithNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = retriever("notebook")
        if cls.r.version != GOLDEN["model_version"]:
            raise unittest.SkipTest(f"installed model {cls.r.version} is not the one of the golden results "
                                    f"({GOLDEN['model_version']}): regenerate tests/fixtures/glaze_notebook_golden.json")

    def test_rank_and_diversify_match_notebook(self):
        for g in GOLDEN["queries"]:
            q = g["query"]
            with self.subTest(q["color"]):
                ranked = self.r.rank(q["color"], q["surface"], q["cone"], q["query_hex"])
                best = ranked.head(10)
                self.assertEqual(list(best.recipe_id), g["best"]["recipe_id"])
                np.testing.assert_allclose(best.final_score, g["best"]["final_score"], atol=1e-5)
                np.testing.assert_allclose(best.latent_similarity, g["best"]["latent_similarity"], atol=1e-5)
                np.testing.assert_allclose(best.delta_e_2000, g["best"]["delta_e_2000"], atol=1e-6)
                explore = G.diversify(ranked, 10, 2)
                self.assertEqual(list(explore.recipe_id), g["explore"]["recipe_id"])
                np.testing.assert_allclose(explore.final_score, g["explore"]["final_score"], atol=1e-5)

    def test_api_search_matches_notebook(self):
        for g in GOLDEN["queries"]:
            q = g["query"]
            with self.subTest(q["color"]):
                res = self.r.search({"hex": q["query_hex"], "color_name": q["color"], "cone": q["cone"], "surface": q["surface"]})
                self.assertEqual(res["model_text"], f"color: {q['color']}, Cone {q['cone']}, {q['surface']}, unknown, unknown")
                self.assertEqual([x["recipe_id"] for x in res["weighted_results"]], g["best"]["recipe_id"])
                self.assertEqual([x["recipe_id"] for x in res["diverse_results"]], g["explore"]["recipe_id"])
                np.testing.assert_allclose([x["similarity_score"] for x in res["weighted_results"]], g["best"]["final_score"], atol=1e-5)

    def test_notebook_parser_is_kept_for_this_mode(self):
        self.assertEqual(G.cone_overlap("04", "4"), 1.0)          # the notebook's parser: the leading zero is lost
        self.assertEqual(G.cone_overlap("5 1/2", "5.5 - 6"), 1.0)
        self.assertEqual(G.cone_overlap("6", "8 - 10"), 0.0)


# ---------------------------------------------------------------- cones
class Cones(unittest.TestCase):
    def test_orton_parser(self):
        cases = {"04": (-4, -4), "4": (4, 4), "05 1/2 - 03": (-5.5, -3), "04 - 3": (-4, 3), "5 1/2 - 6": (5.5, 6),
                 "5.5 - 6": (5.5, 6), "05.5": (-5.5, -5.5), "010": (-10, -10), "10": (10, 10), "6 - 4": (4, 6),
                 "022 - 010": (-22, -10), "nan": None, "": None, "unknown": None}
        for s, want in cases.items():
            with self.subTest(s):
                self.assertEqual(G.cone_range_orton(s), want)

    def test_overlap_keeps_04_and_4_apart(self):
        o = lambda a, b: G.cone_overlap(a, b, G.cone_range_orton)  # noqa: E731
        self.assertEqual(o("04", "4"), 0.0)
        self.assertEqual(o("4", "04"), 0.0)
        self.assertEqual(o("6", "06"), 0.0)
        self.assertEqual(o("04", "04 - 3"), 1.0)
        self.assertEqual(o("1", "04 - 3"), 1.0)          # 'Gooey White Cone 1' (recipe 6172): range 04 - 3 contains cone 1
        self.assertEqual(o("6", "5 - 7"), 1.0)
        self.assertEqual(o("5 - 6", "05 1/2 - 6"), 1.0)
        self.assertTrue(np.isnan(o("unknown", "6")))

    def test_normalize_cone(self):
        cases = {"04": "04", "6": "6", "5.5": "5 1/2", "5½": "5 1/2", "6 - 5": "5 - 6", "05 1/2 - 03": "05 1/2 - 03",
                 "03 - 05 1/2": "05 1/2 - 03", "6 - 6": "6", "04-6": "04 - 6", "": None, None: None, "any": None, "Unknown": None}
        for s, want in cases.items():
            with self.subTest(s):
                self.assertEqual(GR.normalize_cone(s), want)
        for bad in ("99", "abc", "1 - 2 - 3", "0", "15"):
            with self.subTest(bad), self.assertRaises(ValueError):
                GR.normalize_cone(bad)

    @unittest.skipUnless(has_7k(), "7k_matched_all_data.csv not found")
    def test_recipe_2952_is_cone_04(self):
        orton, nb = retriever("orton"), retriever("notebook")
        row = lambda df: df[df.recipe_id == 2952].iloc[0]  # noqa: E731
        r4 = row(orton.rank("warm gray", "unknown", "4", "#A89F91"))
        self.assertEqual(r4.cone_text, "04")
        self.assertEqual(r4.cone_match, 0.0)
        self.assertEqual(row(orton.rank("warm gray", "unknown", "04", "#A89F91")).cone_match, 1.0)
        self.assertEqual(row(nb.rank("warm gray", "unknown", "4", "#A89F91")).cone_match, 1.0)   # the defect, notebook mode only

    @unittest.skipUnless(has_7k(), "7k_matched_all_data.csv not found")
    def test_cone_06_search_finds_low_fire_glazes(self):
        p = {"hex": "#B5523B", "color_name": "terracotta red", "cone": "06", "surface": "Glossy"}
        low = lambda res: sum(is_low_fire(retriever("orton").ref.cones.get(x["recipe_id"])) for x in res["weighted_results"])  # noqa: E731
        n_orton, n_nb = low(retriever("orton").search(p)), low(retriever("notebook").search(p))
        self.assertGreaterEqual(n_orton, 5)
        self.assertGreater(n_orton, n_nb)


# ---------------------------------------------------------------- inputs
class Inputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = retriever("orton")

    def test_surface_words(self):
        ch = self.r.surfaces
        self.assertEqual(len(ch), 9)
        cases = {"Glossy": "Glossy", "glossy": "Glossy", "gloss": "Glossy", "GLOSSY": "Glossy", "semi gloss": "Semi-glossy",
                 "Semi-glossy": "Semi-glossy", "semi-matte": "Semi-matte", "satin matte": "Satin-matte", "matt": "Matte",
                 "Smooth Matte": "Smooth Matte", "": None, "any": None, None: None}
        for s, want in cases.items():
            with self.subTest(s):
                self.assertEqual(GR.normalize_surface(s, ch), want)
        self.assertNotEqual(GR.normalize_surface("Semi-glossy", ch), GR.normalize_surface("Glossy", ch))
        with self.assertRaises(ValueError):
            GR.normalize_surface("shiny wet", ch)

    def test_hex(self):
        self.assertEqual(self.r.query({"hex": "3054a5"})["hex"], "#3054A5")
        self.assertEqual(self.r.query({"hex": "#35A"})["hex"], "#3355AA")
        for bad in ("#GGGGGG", "", "12345", None, "#3054A", "blue"):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.r.query({"hex": bad})

    def test_colour_words(self):
        names = [n["name"] for n in self.r.names]
        self.assertGreaterEqual(len(names), 80)
        self.assertNotIn("clear", names)                          # no HEX in the catalog: cannot be a colour query
        for n in self.r.names[:20]:
            with self.subTest(n["name"]):
                self.assertEqual(self.r.color_name(n["hex"])["name"], n["name"])
        q = self.r.query({"hex": "#3054A5", "color_name": "  Blue "})
        self.assertEqual((q["color_name"], q["color_name_source"]), ("Blue", "you"))
        auto = self.r.query({"hex": "#3054A5"})
        self.assertIn(auto["color_name"], names)
        self.assertTrue(auto["color_name_source"].startswith("nearest"))

    def test_any_cone_and_surface(self):
        res = self.r.search({"hex": "#3054A5"})
        self.assertTrue(res["model_text"].endswith("Cone unknown, unknown, unknown, unknown"))
        for x in res["weighted_results"]:
            self.assertEqual((x["contributions"]["surface"], x["contributions"]["cone"]), (0.0, 0.0))

    def test_weights_and_limits(self):
        for bad in ({"latent": 0.5, "color": 0.5, "surface": 0.5, "cone": 0}, {"latent": -0.1, "color": 0.9, "surface": 0.1, "cone": 0.1},
                    {"latent": 1.0}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.r.search({"hex": "#3054A5", "weights": bad})
        res = self.r.search({"hex": "#3054A5", "weights": {"latent": 0, "color": 1, "surface": 0, "cone": 0}})
        de = [x["score_components"]["delta_e_2000"] for x in res["weighted_results"]]
        self.assertEqual(de, sorted(de))                          # colour only: ranked by CIEDE2000
        for bad in ({"top_k": 0}, {"top_k": 51}, {"max_per_cluster": -1}, {"surface": "shiny wet"}, {"cone": "99"}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.r.search({"hex": "#3054A5", **bad})


# ---------------------------------------------------------------- results
class Results(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = retriever("orton")
        cls.res = cls.r.search({"hex": "#3054A5", "color_name": "Blue", "cone": "6", "surface": "Glossy"})

    def test_json_without_nan(self):
        json.dumps(self.res, allow_nan=False)

    def test_two_lists_from_one_ranking(self):
        best, explore = self.res["weighted_results"], self.res["diverse_results"]
        self.assertEqual((len(best), len(explore)), (10, 10))
        for lst in (best, explore):
            s = [x["similarity_score"] for x in lst]
            self.assertEqual(s, sorted(s, reverse=True))
        self.assertEqual(best[0]["recipe_id"], explore[0]["recipe_id"])
        per = {}
        for x in explore:
            per[x["cluster_id"]] = per.get(x["cluster_id"], 0) + 1
        self.assertLessEqual(max(per.values()), self.res["max_per_cluster"])
        self.assertGreater(len(per), len({x["cluster_id"] for x in best}) - 1)

    def test_fields(self):
        for x in self.res["weighted_results"] + self.res["diverse_results"]:
            with self.subTest(x["recipe_id"]):
                self.assertEqual(x["recipe_url"], f"https://glazy.org/recipes/{x['recipe_id']}")
                self.assertTrue(x["image_url"].startswith("https://cdn.glazy.org/"))
                self.assertEqual(x["thumb_url"], x["image_url"].rsplit("/", 1)[0] + "/m_" + x["image_url"].rsplit("/l_", 1)[1])
                self.assertRegex(x["hex"], r"^#[0-9A-F]{6}$")
                self.assertAlmostEqual(sum(x["contributions"].values()), x["similarity_score"], places=5)
                self.assertIsInstance(x["cluster_id"], int)
                if has_7k():
                    self.assertTrue(x["recipe_name"])
                    self.assertTrue(all({"name", "percent", "addition"} <= set(i) for i in x["ingredients"]))

    def test_stable(self):
        again = self.r.search({"hex": "#3054A5", "color_name": "Blue", "cone": "6", "surface": "Glossy"})
        for k in ("weighted_results", "diverse_results"):
            self.assertEqual([(x["recipe_id"], x["similarity_score"]) for x in again[k]],
                             [(x["recipe_id"], x["similarity_score"]) for x in self.res[k]])

    def test_thumb_url(self):
        u = "https://cdn.glazy.org/public/uploads/recipes/18/l_18.5b34210f27f69.jpg"
        self.assertEqual(G.thumb_url(u), "https://cdn.glazy.org/public/uploads/recipes/18/m_18.5b34210f27f69.jpg")
        self.assertIsNone(G.thumb_url(None))
        self.assertIsNone(G.thumb_url("https://cdn.glazy.org/public/uploads/recipes/18/18.jpg"))


# ---------------------------------------------------------------- model files
class Artifacts(unittest.TestCase):
    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(FileNotFoundError, "INSTALL_GLAZE_MODEL"):
            G.load_artifacts(Path(d))

    def test_vectors_in_another_order(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("catalog", "heads"):
                shutil.copy(MODEL_DIR / G.FILES[f], Path(d) / G.FILES[f])
            with np.load(MODEL_DIR / G.FILES["vectors"]) as v:
                np.savez(Path(d) / G.FILES["vectors"], recipe_ids=v["recipe_ids"][::-1], vectors=v["vectors"][::-1])
            with self.assertRaisesRegex(ValueError, "same order"):
                G.load_artifacts(Path(d))

    def test_no_eligible_recipes(self):
        with self.assertRaisesRegex(ValueError, "No eligible"):
            retriever("orton").rank("Blue", "Glossy", "6", "#3054A5", allowed_splits=["no such split"])


# ---------------------------------------------------------------- HTTP endpoints (app/server.py request handler)
class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.srv, cls.tmp = server, tempfile.TemporaryDirectory()
        cls.saved = (server.QDIR, server.GLAZE_LOG, dict(server.S))
        server.QDIR, server.GLAZE_LOG = Path(cls.tmp.name), Path(cls.tmp.name) / "glaze_search_log.jsonl"
        server.S["glaze"] = retriever("orton")
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.srv.QDIR, cls.srv.GLAZE_LOG = cls.saved[:2]
        cls.srv.S.clear()
        cls.srv.S.update(cls.saved[2])
        cls.tmp.cleanup()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_options(self):
        code, o = self.call("/api/glaze/options")
        self.assertEqual(code, 200)
        cones = [c["value"] for c in o["cones"]]
        self.assertIn("04", cones)
        self.assertIn("4", cones)
        self.assertLess(cones.index("04"), cones.index("4"))      # Orton order: low fire first
        self.assertEqual(len(o["surfaces"]), 9)
        self.assertEqual(o["cone_scoring"], "orton")
        self.assertAlmostEqual(sum(o["weights"].values()), 1.0)

    def test_color_name(self):
        code, r = self.call("/api/glaze/color_name?hex=3054A5")
        self.assertEqual((code, r["hex"]), (200, "#3054A5"))
        self.assertTrue(r["name"])
        self.assertEqual(self.call("/api/glaze/color_name?hex=zz")[0], 400)

    def test_search(self):
        code, r = self.call("/api/glaze/search", {"hex": "#3054A5", "color_name": "Blue", "cone": "6", "surface": "Glossy", "top_k": 10})
        self.assertEqual(code, 200)
        self.assertEqual((len(r["weighted_results"]), len(r["diverse_results"])), (10, 10))
        for k in ("query", "weighted_results", "diverse_results", "model_version"):
            self.assertIn(k, r)
        log = self.srv.GLAZE_LOG.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(json.loads(log[-1])["best"], [x["recipe_id"] for x in r["weighted_results"]])

    def test_bad_requests(self):
        for body, word in (({"hex": "#GGGGGG"}, "hex"), ({"hex": "#3054A5", "surface": "shiny wet"}, "surface"),
                           ({"hex": "#3054A5", "cone": "99"}, "cone")):
            with self.subTest(body):
                code, r = self.call("/api/glaze/search", body)
                self.assertEqual(code, 400)
                self.assertIn(word, r["error"])

    def test_model_missing(self):
        g = self.srv.S.pop("glaze")
        self.srv.S["glaze_error"] = "FileNotFoundError: test"
        try:
            for path, body in (("/api/glaze/options", None), ("/api/glaze/search", {"hex": "#3054A5"})):
                code, r = self.call(path, body)
                self.assertEqual(code, 503)
                self.assertIn("INSTALL_GLAZE_MODEL", r["error"])
        finally:
            self.srv.S["glaze"] = g

    def test_other_routes_unchanged(self):
        req = urllib.request.urlopen(self.base + "/lib/index.json", timeout=10)
        self.assertEqual(req.status, 200)
        self.assertEqual(self.call("/api/nothing", {})[0], 404)
        page = urllib.request.urlopen(self.base + "/", timeout=10).read().decode("utf-8")
        self.assertIn('id="gcard"', page)
        self.assertTrue(re.search(r"<title>Glaze Search</title>", page))


if __name__ == "__main__":
    unittest.main()
