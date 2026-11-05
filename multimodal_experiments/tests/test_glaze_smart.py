"""Tests of card G's smart search: the free-text reader (mm/glaze_text.py), the joint model search (mm/glaze_joint.py)
and the endpoints /api/glaze/understand and /api/glaze/smart_search (real request handler of app/server.py).
run (from multimodal_experiments):  ..\\.venv\\Scripts\\python.exe -m unittest tests.test_glaze_smart -v
The joint-model tests need the trained model (INSTALL_GLAZE_MODEL.bat); the reader tests need nothing.
"""
from __future__ import annotations

import json
import os
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

from mm import glaze_joint as J  # noqa: E402
from mm import glaze_text as GT  # noqa: E402
from mm.glaze_retrieval import cone_range_orton  # noqa: E402

MODEL = J.CODE / J.read_config()["model_dir"]
HAVE_MODEL = all((MODEL / f).is_file() for f in ("heads.pt", "gallery.npz", "meta.json"))
_JS: dict = {}


def joint() -> J.JointSearch:
    if "js" not in _JS:
        _JS["js"] = J.JointSearch()
    return _JS["js"]


def val(x):
    return x["value"] if x else None


class Reader(unittest.TestCase):
    CASES = {
        "shinyy cobolt blu glaze ^6 ox, kinda runny": {"cone": "6", "surface": "Glossy", "atm": ["Oxidation"], "fx": ["flowing"], "colour": "cobalt blue"},
        "looking for a satin matte light celadon cone 10 reduction pls": {"cone": "10", "surface": "Satin-matte", "atm": ["Reduction"], "colour": "light celadon"},
        "low fire red earthenware, opaque, cone 04 with crackle": {"cone": "04", "transparency": "Opaque", "fx": ["crackle"], "colour": "red"},
        "glossy black tenmoku c10 reducton with 8% red iron oxide": {"cone": "10", "surface": "Glossy", "atm": ["Reduction"], "colour": "black"},
        "matt white liner for my mugs mid fire": {"cone": "5 - 6", "surface": "Matte", "colour": "white"},
        "semi-gloss teal c 5 1/2 soda fired": {"cone": "5 1/2", "surface": "Semi-glossy", "atm": ["Salt & Soda"], "colour": "teal"},
        "Light Bluish Gray SEMI MATT cone 6-7 oxidation": {"cone": "6 - 7", "surface": "Semi-matte", "atm": ["Oxidation"], "colour": "light blue gray"},
        "transparant clear base cone 05-03 electric kiln": {"cone": "05 - 03", "transparency": "Transparent", "atm": ["Oxidation"], "colour": None},
    }

    def test_sentences(self):
        for text, want in self.CASES.items():
            u = GT.extract(text)
            with self.subTest(text):
                self.assertEqual(val(u["cone"]), want["cone"])
                self.assertEqual(val(u["surface"]), want.get("surface"))
                self.assertEqual(val(u["transparency"]), want.get("transparency"))
                self.assertEqual([a["value"] for a in u["atmospheres"]], want.get("atm", []))
                self.assertEqual([e["value"] for e in u["effects"]], want.get("fx", []))
                self.assertEqual(u["colour"]["words"] if u["colour"] else None, want["colour"])
                if u["colour"]:
                    self.assertRegex(u["colour"]["hex"], r"^#[0-9A-F]{6}$")

    def test_values(self):
        u = GT.extract("glossy black tenmoku c10 reducton with 8% red iron oxide")
        self.assertEqual([(c["name"], c["oxide"], c["percent"]) for c in u["colourants"]], [("red iron oxide", "Fe2O3", 8.0)])
        self.assertIn(["reducton", "reduction"], u["corrections"])
        u = GT.extract("SiO2 64.2 Al2O3 13.1 CaO 9.0 K2O 3.2 Fe2O3 2.0 cone 6 glosy")
        self.assertEqual(u["oxides"], {"SiO2": 64.2, "Al2O3": 13.1, "CaO": 9.0, "K2O": 3.2, "Fe2O3": 2.0})
        self.assertEqual((val(u["cone"]), val(u["surface"])), ("6", "Glossy"))
        self.assertIsNone(GT.extract("cone 6 cobalt blue glossy")["colourants"] or None)     # "6 cobalt" is not 6 % cobalt
        self.assertEqual(val(GT.extract("cone 04")["cone"]), "04")
        self.assertNotEqual(val(GT.extract("cone 04")["cone"]), val(GT.extract("cone 4")["cone"]))
        self.assertIsNone(GT.extract("")["colour"])

    def test_noisy_sentences(self):
        facts = {"colour": "cobalt blue", "surface": "glossy", "transparency": "", "cone_lo": "6", "cone_hi": "", "atmospheres": ["oxidation"], "subtype": ""}
        s1, s2 = GT.noisy_variants(facts, 40, 3), GT.noisy_variants(facts, 40, 3)
        self.assertEqual(s1, s2)                                   # same seed, same sentences
        cones = [val(GT.extract(s)["cone"]) for s in s1]
        surf = [val(GT.extract(s)["surface"]) for s in s1]
        read_c, read_s = [c for c in cones if c], [s for s in surf if s]
        self.assertTrue(all(c == "6" or c == "5 - 6" for c in read_c), read_c)
        self.assertTrue(all(s == "Glossy" for s in read_s), read_s)
        self.assertGreaterEqual(len(read_c) / len(s1), 0.6)


@unittest.skipUnless(HAVE_MODEL, f"smart search model not trained ({MODEL})")
class JointModel(unittest.TestCase):
    def test_meta(self):
        m = joint().meta
        t = m["eval"]["test"]
        for task in ("photo -> recipe", "formula -> recipe", "noisy text -> recipe", "description -> recipe"):
            self.assertGreater(t[task]["R@10"], 5 * t[task]["random_R@10"], task)     # far above chance on held-out recipes
        self.assertGreater(t["noisy text -> recipe (trained, all parts)"]["R@10"], t["noisy text -> tag text (pretrained MiniLM only, no training)"]["R@10"])

    def test_text_search(self):
        r = joint().search({"text": "shinyy cobolt blu glaze ^6 ox"})
        self.assertEqual((len(r["weighted_results"]), len(r["diverse_results"])), (10, 10))
        self.assertAlmostEqual(sum(r["parts"].values()), 1.0, places=4)
        self.assertEqual(set(r["parts"]), {"text", "colour", "cone", "surface", "atmosphere"})
        json.dumps(r, allow_nan=False)
        for x in r["weighted_results"]:
            self.assertAlmostEqual(sum(x["contributions"].values()), x["similarity_score"], places=3)
            self.assertTrue(x["recipe_url"].startswith("https://glazy.org/recipes/"))
        fam = [x["cluster_id"] for x in r["diverse_results"]]
        self.assertLessEqual(max(fam.count(f) for f in fam), r["max_per_cluster"])
        s = [x["similarity_score"] for x in r["weighted_results"]]
        self.assertEqual(s, sorted(s, reverse=True))

    def test_parts_alone(self):
        js = joint()
        r = js.search({"hex": "#3054A5"})
        self.assertEqual(set(r["parts"]), {"colour"})
        self.assertLess(r["weighted_results"][0]["score_components"]["delta_e_2000"], 5)
        ox = {"SiO2": 64.2, "Al2O3": 13.1, "CaO": 9.0, "K2O": 3.2, "Fe2O3": 2.0}
        self.assertEqual(set(js.search({"oxides": ox})["parts"]), {"formula"})
        r = js.search({"cone": "04"})
        self.assertTrue(all(cone_range_orton(x["cone"])[0] < 0 for x in r["weighted_results"]), [x["cone"] for x in r["weighted_results"]])
        with self.assertRaises(ValueError):
            js.search({})
        with self.assertRaises(ValueError):
            js.search({"cone": "99"})

    def test_formula_needs_a_flux(self):
        js = joint()
        with self.assertRaisesRegex(ValueError, "flux"):                 # SiO2 + Al2O3 only: no flux, cannot be used
            js.search({"oxides": {"SiO2": 64.2, "Al2O3": 13.1}})
        r = js.search({"text": "glossy white cone 6", "oxides": {"SiO2": 64.2, "Al2O3": 13.1}})
        self.assertNotIn("formula", r["parts"])
        self.assertTrue(any("formula ignored" in w and "no flux" in w for w in r["warnings"]), r["warnings"])
        r = js.search({"text": "glossy white cone 6", "oxides": {"SiO2": 64.2, "Al2O3": 13.1, "Fe2O3": 2.0}})
        self.assertNotIn("formula", r["parts"])                         # 3 oxides but still no flux
        r = js.search({"oxides": {"SiO2": 64.2, "Al2O3": 13.1, "CaO": 9.0}})
        self.assertEqual(set(r["parts"]), {"formula"})
        self.assertEqual(r["warnings"], [])

    def test_photo_part(self):
        js = joint()
        q = np.random.default_rng(0).normal(size=js.heads["photo"].net[0].in_features).astype(np.float32)
        r = js.search({}, photo_vec=q / np.linalg.norm(q))
        self.assertEqual(set(r["parts"]), {"photo"})

    def test_understand_suggestions(self):
        u = joint().understand("a celadon for my gas kiln", lambda h: "sage green")
        self.assertIn("inferred", u)
        self.assertTrue({"surface", "transparency", "atmosphere"} <= set(u["inferred"]))


@unittest.skipUnless(HAVE_MODEL, "smart search model not trained")
class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.srv, cls.tmp = server, tempfile.TemporaryDirectory()
        cls.saved = (server.QDIR, server.GLAZE_LOG, dict(server.S))
        server.QDIR, server.GLAZE_LOG = Path(cls.tmp.name), Path(cls.tmp.name) / "glaze_search_log.jsonl"
        server.S["joint"] = joint()
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

    def call(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_understand(self):
        code, u = self.call("/api/glaze/understand", {"text": "shinyy cobolt blu ^6 ox"})
        self.assertEqual(code, 200)
        self.assertEqual((u["cone"]["value"], u["surface"]["value"]), ("6", "Glossy"))

    def test_smart_search(self):
        code, r = self.call("/api/glaze/smart_search", {"text": "matt white liner mid fire", "top_k": 10})
        self.assertEqual(code, 200)
        self.assertEqual(len(r["weighted_results"]), 10)
        self.assertEqual(json.loads(self.srv.GLAZE_LOG.read_text(encoding="utf-8").splitlines()[-1])["mode"], "smart search")
        self.assertEqual(self.call("/api/glaze/smart_search", {})[0], 400)
        self.assertEqual(self.call("/api/glaze/smart_search", {"use_photo": True, "qid": "nope", "object": 1})[0], 400)

    def test_not_installed(self):
        js = self.srv.S.pop("joint")
        try:
            code, r = self.call("/api/glaze/smart_search", {"text": "blue"})
            self.assertEqual(code, 503)
            self.assertIn("INSTALL_GLAZE_MODEL", r["error"])
        finally:
            self.srv.S["joint"] = js


if __name__ == "__main__":
    unittest.main()
