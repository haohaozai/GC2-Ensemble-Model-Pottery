"""Grounding DINO evidence for the six object attributes (attributes_v2).

For every attribute the object cut-out is shown to Grounding DINO base with one text prompt that lists several short
phrases per class, written from Ceramic_Real_Photo_Comparison_Guide_V4 (visible cues, not class names only).
Per phrase we keep the raw grounding evidence instead of a thresholded detection:

  score     max over the 900 queries of the max sigmoid logit over the phrase's tokens (the HF post-processing score)
  cover     area of the best box / area of the object's bounding box
  n         number of boxes with score >= 0.25 (several drips / spots / crystals -> many boxes)
  box       best box (x0, y0, x1, y1) in object-crop pixels - shown in the review app as "where DINO found it"

A class score is later derived from its phrases (max / mean / best phrase, chosen on validation) - see
mm/dino_calib.py. Nothing here decides a label.
"""
from __future__ import annotations

import numpy as np

# attribute -> class -> phrases (lower case, no '.' inside a phrase)
PROMPTS: dict[str, dict[str, list[str]]] = {
    "form": {
        "bowl": ["bowl", "round open bowl"],
        "plate": ["plate", "flat dish"],
        "cup": ["cup", "mug with handle"],
        "vase": ["vase", "bottle"],
        "jar": ["jar", "lidded pot"],
        "test tile": ["test tile", "flat rectangular tile"],
        "sculpture": ["sculpture", "ceramic statue"],
        "figurine": ["figurine", "small animal figure"],
        "abstract ceramic artwork": ["abstract art piece"],
        "irregular ceramic form": ["irregular hand-built form"],
        "other vessel": ["teapot", "pitcher"],
    },
    "surface_finish": {
        "glossy": ["glossy glaze", "sharp bright reflection"],
        "semi-glossy": ["semi-glossy glaze", "soft highlight"],
        "satin": ["satin glaze", "broad gentle sheen"],
        "satin-matte": ["satin matte glaze", "faint sheen"],
        "semi-matte": ["semi-matte glaze", "little shine"],
        "matte": ["matte glaze", "dull surface"],
        "smooth matte": ["smooth matte glaze", "even low-gloss surface"],
        "stony matte": ["stony matte glaze", "mineral stone-like surface"],
        "dry matte": ["dry matte glaze", "chalky surface"],
    },
    "transparency": {
        "transparent": ["transparent glaze", "clear glass-like glaze"],
        "translucent": ["translucent glaze", "milky glaze"],
        "semi-opaque": ["semi-opaque glaze", "partly hidden clay"],
        "opaque": ["opaque glaze", "solid colour glaze"],
    },
    "texture": {
        "smooth": ["smooth surface", "even surface"],
        "rough": ["rough surface", "uneven surface"],
        "granular": ["granular surface", "sandy grains"],
        "dimpled": ["dimpled surface", "tiny depressions"],
    },
    "glaze_effect": {
        "speckled": ["speckles", "small dark dots"],
        "oil spot": ["oil spots", "round metallic spots"],
        "crystalline": ["crystals", "starburst crystal clusters"],
        "mottled": ["mottled pattern", "patchy colour variation"],
        "flowing": ["glaze drips", "running streaks"],
        "breaking at edges": ["colour break on edges", "thin glaze on ridges"],
        "metallic / iridescent": ["metallic sheen", "iridescent surface"],
        "visible crack network (crackle)": ["crackle lines", "crack network"],
    },
    "defect": {
        "pinholes": ["pinholes", "tiny holes"],
        "blistering": ["blisters", "burst bubbles"],
        "crawling": ["bare clay patches", "crawled glaze"],
        "crazing": ["fine cracks", "crazing"],
        "excessive glaze running": ["thick glaze pool", "glaze run at base"],
    },
}

# prompt variants tried per attribute (round 1 = "mixed" above; the others are scored on the same objects and the best
# variant + aggregation per attribute is chosen on the Claude-labelled VALIDATION objects - mm/dino_calib.py)
VARIANTS: dict[str, dict[str, dict[str, list[str]]]] = {
    "form": {
        "names": {c: [c if c not in ("abstract ceramic artwork", "irregular ceramic form", "other vessel") else
                      {"abstract ceramic artwork": "abstract sculpture", "irregular ceramic form": "irregular pottery",
                       "other vessel": "teapot"}[c]] for c in PROMPTS["form"]},
        "cues": {"bowl": ["round bowl with open rim", "bowl interior"], "plate": ["flat round plate", "shallow dish with rim"],
                 "cup": ["cup with handle", "drinking cup"], "vase": ["vase with narrow neck", "bottle shape"],
                 "jar": ["lidded jar", "round storage jar"], "test tile": ["small flat ceramic tile", "rectangular glaze sample"],
                 "sculpture": ["ceramic sculpture", "carved statue"], "figurine": ["animal figurine", "owl figure with eyes"],
                 "abstract ceramic artwork": ["abstract organic form"], "irregular ceramic form": ["rough irregular lump"],
                 "other vessel": ["teapot with spout", "pitcher with spout"]},
    },
    "surface_finish": {
        "names": {c: [c + " glaze"] for c in PROMPTS["surface_finish"]},
        "cues": {"glossy": ["mirror-like reflection", "wet shiny glass surface"], "semi-glossy": ["soft shiny highlight", "shiny but blurred reflection"],
                 "satin": ["silky sheen", "soft satin shine"], "satin-matte": ["faint soft sheen", "low sheen surface"],
                 "semi-matte": ["mostly dull surface", "slight shine"], "matte": ["dull non-reflective surface", "flat matte finish"],
                 "smooth matte": ["smooth eggshell surface", "even velvety surface"], "stony matte": ["stone-like rough surface", "mineral texture"],
                 "dry matte": ["chalky powdery surface", "dry unglazed-looking surface"]},
    },
    "transparency": {
        "names": {c: [c + " glaze"] for c in PROMPTS["transparency"]},
        "cues": {"transparent": ["see-through clear glass", "clay visible under clear glaze"],
                 "translucent": ["glaze pooling darker in recesses", "partly see-through glaze"],
                 "semi-opaque": ["clay faintly showing through glaze", "thin glaze on edges"],
                 "opaque": ["solid coloured coating", "glaze hiding the clay"]},
    },
    "texture": {
        "names": {c: [c + " texture"] for c in PROMPTS["texture"]},
        "cues": {"smooth": ["smooth even surface", "uniform glaze surface"], "rough": ["rough bumpy surface", "coarse uneven glaze"],
                 "granular": ["sandy granular surface", "grainy speckled texture"], "dimpled": ["small craters", "pitted dimpled surface"]},
    },
    "glaze_effect": {
        "names": {c: [{"visible crack network (crackle)": "crackle", "metallic / iridescent": "metallic glaze",
                       "breaking at edges": "glaze breaking on edges"}.get(c, c)] for c in PROMPTS["glaze_effect"]},
        "cues": {"speckled": ["dark speckles", "iron spots"], "oil spot": ["silvery oil spots"],
                 "crystalline": ["flower-shaped crystals", "crystal clusters"], "mottled": ["mottled patches", "variegated colour"],
                 "flowing": ["glaze drips running down", "streaky flow lines"], "breaking at edges": ["lighter colour on ridges", "glaze thin on rim edge"],
                 "metallic / iridescent": ["shiny metal-like surface", "rainbow iridescent sheen"],
                 "visible crack network (crackle)": ["web of fine cracks", "crazed crackle lines"]},
    },
    "defect": {
        "names": {c: [c] for c in PROMPTS["defect"]},
        "cues": {"pinholes": ["small pin holes", "tiny pits in glaze"], "blistering": ["glaze bubbles", "blistered craters"],
                 "crawling": ["glaze islands with bare clay", "glaze pulled away"], "crazing": ["fine crack lines"],
                 "excessive glaze running": ["glaze pooled at the bottom", "thick drip at base"]},
    },
}
VARIANTS = {a: {"mixed": PROMPTS[a], **v} for a, v in VARIANTS.items()}
# object parts used for colour regions (not an attribute): areas that should not drive colour matching
PARTS = {"label": ["handwritten text", "printed label"], "hole": ["hanging hole"], "bare clay": ["bare unglazed clay", "clay foot"],
         "stand": ["stand", "finger"]}
ATTRS = list(PROMPTS)
SEP_ID = 1012          # BERT token id of "."
N_BOX_THR = 0.25
BG = 200               # neutral light-grey background for the cut-out


def bank(attr: str, variant: str = "mixed") -> dict[str, list[str]]:
    """phrases of a prompt set; '<set>_tiles' = the same phrases scored on zoomed tiles (round 2)."""
    if attr == "parts":
        return PARTS
    return VARIANTS[attr][variant[:-6] if variant.endswith("_tiles") else variant]


def phrase_list(attr: str, variant: str = "mixed") -> list[tuple[str, str]]:
    return [(c, p) for c, ps in bank(attr, variant).items() for p in ps]


def prompt_text(attr: str, variant: str = "mixed") -> str:
    return " . ".join(p for _, p in phrase_list(attr, variant)) + " ."


ALL_PROMPTS = [(a, v) for a in ATTRS for v in VARIANTS[a]] + [("parts", "parts")]
# round 2 (after round 1 missed the 98 % target): decoder embeddings of the best boxes for every attribute, and zoomed
# 2 x 2 tiles for the small, local attributes (speckles, crystals, crackle, pinholes, texture)
R2_EMBED = [(a, "mixed") for a in ATTRS]
R2_TILES = [(a, "mixed") for a in ("glaze_effect", "defect", "texture")]
TILE = 0.6                  # tile side as a share of the object side (2 x 2 grid, overlapping)
EMB_TOPQ = 5


def cutout(rgba: np.ndarray, pad=0.06) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """object cut-out on a neutral background, cropped to the mask's bounding box (+ padding)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgba[..., :3].astype(np.float32) * a + BG * (1 - a)).astype(np.uint8)
    ys, xs = np.nonzero(rgba[..., 3] >= 125)
    if not len(ys):
        return rgb, (0, 0, rgb.shape[1], rgb.shape[0])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, x0 = max(0, y0 - py), max(0, x0 - px)
    y1, x1 = min(rgb.shape[0], y1 + py), min(rgb.shape[1], x1 + px)
    return rgb[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


class GpuGuard:
    """keeps a long GPU job gentle on a laptop: hard VRAM cap for this process, a short duty-cycle pause after every
    forward pass, and a temperature check (waits until the GPU cools down). After the 2026-10-18 crashes (bugcheck 0x20001
    while an 8-prompt batch filled 7.9 / 8 GB VRAM) every standalone DINO job runs through this guard.
    2026-10-19 (the full run lost ~4.5 h): Windows throttled the below-normal-priority process in the background (GPU idle,
    7 s / object) and the laptop went to sleep for 3.5 h. Now: normal priority, opted out of Windows power throttling
    (EcoQoS), the system is kept awake while the job runs (released when it ends), pause 10 % instead of 35 %."""

    def __init__(self, vram_cap_gb=3.0, duty_pause=0.05, max_temp=82, cool_temp=72, check_every=20, cpu_threads=4):
        import torch
        self.torch, self.duty_pause, self.max_temp, self.cool_temp, self.check_every = torch, duty_pause, max_temp, cool_temp, check_every
        self.n = 0
        self.log = []
        if torch.cuda.is_available() and vram_cap_gb:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(min(0.9, vram_cap_gb / total))
        self.windows_settings = self._windows()
        # the CPU only prepares inputs; PyTorch's default 16 spinning threads took ~10 cores and starved the main thread
        torch.set_num_threads(cpu_threads)
        self.windows_settings["cpu_threads"] = cpu_threads

    @staticmethod
    def _windows() -> dict:
        """normal priority, no EcoQoS background throttling, no idle sleep while this process runs (Windows only)."""
        out = {}
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE            # 64-bit pseudo handle (-1); int truncation broke it
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
            k32.SetThreadExecutionState.restype = wintypes.DWORD
            proc = k32.GetCurrentProcess()
            out["priority_normal"] = bool(k32.SetPriorityClass(proc, 0x20))            # NORMAL_PRIORITY_CLASS

            class PPTS(ctypes.Structure):                                                # PROCESS_POWER_THROTTLING_STATE
                _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
            st = PPTS(1, 0x1, 0x0)                    # control EXECUTION_SPEED, state 0 = never throttle this process
            out["no_power_throttling"] = bool(k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st)))
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep until the process ends (the screen may still turn off)
            out["keep_awake"] = bool(k32.SetThreadExecutionState(0x80000000 | 0x00000001))
        except Exception as exc:  # noqa: BLE001
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    @staticmethod
    def gpu_state() -> dict:
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
            t, m, u, w = (x.strip() for x in out.split(","))
            return {"temp_C": float(t), "vram_used_MB": float(m), "util_pct": float(u), "power_W": float(w)}
        except Exception:  # noqa: BLE001
            return {}

    def after_pass(self, seconds: float):
        import time
        if self.duty_pause:
            time.sleep(seconds * self.duty_pause)

    def after_object(self):
        import time
        self.n += 1
        if self.n % self.check_every:
            return
        st = self.gpu_state()
        st["peak_alloc_GB"] = round(self.torch.cuda.max_memory_allocated() / 1e9, 2) if self.torch.cuda.is_available() else 0
        self.log.append(st)
        if st.get("temp_C", 0) >= self.max_temp:
            while self.gpu_state().get("temp_C", 0) > self.cool_temp:
                time.sleep(15)


def _reuse_backbone(inner):
    """wraps GroundingDinoModel.backbone: the image backbone does not depend on the text, so for several prompts on the
    SAME prepared image it is computed once. `key` is set explicitly by DinoAttr for every prepared image (no guessing
    from tensor addresses); the lists are copied on every call because the model appends to the position list."""
    import torch

    class ReuseBackbone(torch.nn.Module):
        def __init__(self, inner_):
            super().__init__()
            self.inner = inner_
            self.position_embedding = inner_.position_embedding
            self.conv_encoder = inner_.conv_encoder
            self.key, self._k, self._v = None, None, None
            self.hits = self.misses = 0

        def forward(self, pixel_values, pixel_mask):
            if self.key is None or self.key != self._k:
                self._v = self.inner(pixel_values, pixel_mask)
                self._k = self.key
                self.misses += 1
            else:
                self.hits += 1
            out, pos = self._v
            return list(out), list(pos)

    return ReuseBackbone(inner)


class DinoAttr:
    def __init__(self, device=None, short_side=512, model=None, proc=None, guard: GpuGuard | None = None, batch: int = 1,
                 batch_full: bool = True):
        """model / proc: an already loaded Grounding DINO (the search server shares the segmenter's model - no extra VRAM)."""
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        sys_path_glazeprep()
        from glazeprep import config as C
        self.torch = torch
        self.guard = guard
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.short_side = short_side
        self.proc = proc or AutoProcessor.from_pretrained(C.GROUNDING_DINO_ID)
        self.model = model or AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_DINO_ID).to(self.device).eval()
        if model is None and self.device == "cuda":
            self.model = self.model.half()
        self.dtype = next(self.model.parameters()).dtype
        self.spans = {}
        self._txt = {}
        self._img_id = 0
        # backbone reuse only on a model this object owns (the search server's shared segmenter model is left untouched)
        self.reuse = None
        if model is None:
            self.reuse = _reuse_backbone(self.model.model.backbone)
            self.model.model.backbone = self.reuse
        self.batch = max(1, int(batch))     # prompts / tiles per forward pass (same image size)
        self.batch_full = batch_full        # False: batch only the tiles (same prompt text -> no text padding)

    def _spans(self, key):
        """token spans of every phrase of a prompt key (attr, variant)"""
        if key not in self.spans:
            ids = self.proc.tokenizer(prompt_text(*key))["input_ids"]
            spans, cur = [], []
            for k, t in enumerate(ids[1:-1], 1):
                if t == SEP_ID:
                    if cur:
                        spans.append(cur)
                    cur = []
                else:
                    cur.append(k)
            if cur:
                spans.append(cur)
            assert len(spans) == len(phrase_list(*key)), (key, len(spans), len(phrase_list(*key)))
            self.spans[key] = spans
        return self.spans[key]

    def _resize(self, pil):
        """own resize: short side = short_side (long side <= 1.667x), up- or down-scaling. Tiny crops must be enlarged:
        Grounding DINO needs more image tokens than its 900 queries. The processor's own size setting is left unchanged
        because the search server shares the processor with the segmenter."""
        from PIL import Image
        W, H = pil.size
        k = min(self.short_side / min(W, H), 1.667 * self.short_side / max(W, H))
        return pil.resize((max(32, round(W * k)), max(32, round(H * k))), Image.BICUBIC)

    def prepare(self, pil) -> dict:
        """image inputs for every prompt on this image: prepared once (was: once per prompt) + a fresh reuse key."""
        x = self.proc.image_processor(images=pil, do_resize=False, return_tensors="pt")
        self._img_id += 1
        return {"pixel_values": x["pixel_values"].to(self.device, self.dtype), "pixel_mask": x["pixel_mask"].to(self.device),
                "_id": self._img_id}

    def text_inputs(self, key) -> dict:
        """tokenised prompt, prepared once per prompt for the whole run."""
        if key not in self._txt:
            t = self.proc.tokenizer(prompt_text(*key), return_tensors="pt")
            self._txt[key] = {k: v.to(self.device) for k, v in t.items()}
        return self._txt[key]

    def _pass(self, pil, key, want_emb=False):
        """one forward pass of one prompt -> per phrase max score [P], best box (relative cx cy w h) [P, 4],
        box count >= N_BOX_THR [P] (+ mean decoder embedding of the EMB_TOPQ best queries [256])."""
        import time
        torch = self.torch
        t0 = time.time()
        if isinstance(pil, dict):                     # prepared image (+ cached text): no repeated CPU preprocessing
            inp = {"pixel_values": pil["pixel_values"], "pixel_mask": pil["pixel_mask"], **self.text_inputs(key)}
            if self.reuse is not None:
                self.reuse.key = pil["_id"]
        else:
            inp = self.proc(images=pil, text=prompt_text(*key), return_tensors="pt", do_resize=False).to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
            if self.reuse is not None:
                self.reuse.key = None
        with torch.no_grad():
            o = self.model(**inp)
        lg = o.logits[0].float().sigmoid()
        S = torch.stack([lg[:, sp].max(-1).values for sp in self._spans(key)], 1)       # [Q, P]
        best = S.argmax(0)
        out = {"score": S.max(0).values.cpu().numpy().astype(np.float32), "rel": o.pred_boxes[0].float()[best].cpu().numpy(),
               "n": (S >= N_BOX_THR).sum(0).cpu().numpy().astype(np.int16)}
        if want_emb:
            top = S.max(1).values.argsort(descending=True)[:EMB_TOPQ]
            out["emb"] = o.last_hidden_state[0].float()[top].mean(0).cpu().numpy().astype(np.float32)
        del o, inp, lg, S
        if self.guard:
            self.guard.after_pass(time.time() - t0)
        return out

    def _pass_many(self, items, batch=None):
        """items: [(pil, key, want_emb)] -> outputs like _pass, computed `self.batch` at a time. Every pil in one call has
        the same size (one object or its equal-sized tiles), so images are never padded; texts are right-padded and masked,
        so each prompt's token positions (spans) are unchanged."""
        import time
        torch = self.torch
        batch = batch or self.batch
        if batch == 1:
            return [self._pass(pil, key, want_emb=e) for pil, key, e in items]
        out = []
        for c0 in range(0, len(items), batch):
            chunk = items[c0:c0 + batch]
            t0 = time.time()
            inp = self.proc(images=[c[0] for c in chunk], text=[prompt_text(*c[1]) for c in chunk], return_tensors="pt",
                            padding=True, do_resize=False).to(self.device)
            inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
            with torch.no_grad():
                o = self.model(**inp)
            lg = o.logits.float().sigmoid()
            for b, (_, key, want_emb) in enumerate(chunk):
                S = torch.stack([lg[b][:, sp].max(-1).values for sp in self._spans(key)], 1)
                best = S.argmax(0)
                r = {"score": S.max(0).values.cpu().numpy().astype(np.float32), "rel": o.pred_boxes[b].float()[best].cpu().numpy(),
                     "n": (S >= N_BOX_THR).sum(0).cpu().numpy().astype(np.int16)}
                if want_emb:
                    top = S.max(1).values.argsort(descending=True)[:EMB_TOPQ]
                    r["emb"] = o.last_hidden_state[b].float()[top].mean(0).cpu().numpy().astype(np.float32)
                out.append(r)
            del o, inp, lg
            if self.guard:
                self.guard.after_pass(time.time() - t0)
        return out

    def score_spec(self, images: list[np.ndarray], spec: dict) -> list[dict]:
        """everything a calibrated model set needs, in one go: spec = {"keys": [(a, v)] full-image prompts,
        "emb": [(a, v)] prompts whose decoder embedding is kept (computed on the same pass when also in keys),
        "tiles": [(a, v)] prompts scored on the 2 x 2 zoomed tiles}. Same output format as score() / score_r2()."""
        from PIL import Image
        keys = [tuple(k) for k in spec.get("keys", [])]
        emb = {tuple(k) for k in spec.get("emb", [])}
        tiles = [tuple(k) for k in spec.get("tiles", [])]
        out = []
        for im in images:
            pil0 = Image.fromarray(im)
            W, H = pil0.size
            pil = self._resize(pil0)
            res = {}
            full = keys + [k for k in emb if k not in keys]
            src = self.prepare(pil) if self.batch == 1 else pil
            for key, r in zip(full, self._pass_many([(src, k, k in emb) for k in full], None if self.batch_full else 1)):
                cx, cy, bw, bh = r["rel"].T
                res[key] = {"score": r["score"], "n": r["n"], "cover": np.clip(bw * bh, 0, 1).astype(np.float32),
                            "box": np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1).round(1).astype(np.float32)}
                if "emb" in r:
                    res[key]["emb"] = r["emb"]
            if tiles:
                res.update(self._tiles(pil0, tiles))
            out.append(res)
            if self.guard:
                self.guard.after_object()
        return out

    def _tiles(self, pil0, tile_keys):
        W, H = pil0.size
        res = {}
        tw, th = int(round(W * TILE)), int(round(H * TILE))
        origins = [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]
        crops = [self._resize(pil0.crop((x0, y0, x0 + tw, y0 + th))) for x0, y0 in origins]   # 4 equal-size tiles
        if self.batch == 1:                            # tile-major: every tile prepared once, its backbone reused
            by_key = {k: [] for k in tile_keys}
            for c in crops:
                pc = self.prepare(c)
                for key in tile_keys:
                    by_key[key].append(self._pass(pc, key))
        else:
            by_key = {k: self._pass_many([(c, k, False) for c in crops]) for k in tile_keys}
        for key in tile_keys:
            per = [(r, x0, y0) for r, (x0, y0) in zip(by_key[key], origins)]
            S = np.stack([r["score"] for r, _, _ in per])
            bi = S.argmax(0)
            box, cover = [], []
            for j, t in enumerate(bi):
                r, x0, y0 = per[t]
                cx, cy, bw, bh = r["rel"][j]
                box.append([x0 + (cx - bw / 2) * tw, y0 + (cy - bh / 2) * th, x0 + (cx + bw / 2) * tw, y0 + (cy + bh / 2) * th])
                cover.append(bw * bh * TILE * TILE)
            res[(key[0], key[1] + "_tiles")] = {"score": S.max(0), "tmean": S.mean(0), "n": np.sum([r["n"] for r, _, _ in per], 0).astype(np.int16),
                                                "cover": np.array(cover, np.float32), "box": np.array(box, np.float32).round(1)}
        return res

    def score_r2(self, images: list[np.ndarray], embed_keys=R2_EMBED, tile_keys=R2_TILES) -> list[dict]:
        """round 2 evidence: {(a, v): {"emb": [256]}} for embed_keys and {(a, v + "_tiles"): {"score": max over the 4 tiles,
        "tmean": mean over tiles, "n": summed box count, "cover", "box": best tile box in crop pixels}} for tile_keys."""
        from PIL import Image
        out = []
        for im in images:
            pil0 = Image.fromarray(im)
            W, H = pil0.size
            pil = self._resize(pil0)
            res = {}
            for key in embed_keys:
                r = self._pass(pil, key, want_emb=True)
                res[key] = {"emb": r["emb"]}
            tw, th = int(round(W * TILE)), int(round(H * TILE))
            origins = [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]
            for key in tile_keys:
                per = []
                for x0, y0 in origins:
                    r = self._pass(self._resize(pil0.crop((x0, y0, x0 + tw, y0 + th))), key)
                    per.append((r, x0, y0))
                S = np.stack([r["score"] for r, _, _ in per])                                   # [4, P]
                bi = S.argmax(0)
                box, cover = [], []
                for j, t in enumerate(bi):
                    r, x0, y0 = per[t]
                    cx, cy, bw, bh = r["rel"][j]
                    box.append([x0 + (cx - bw / 2) * tw, y0 + (cy - bh / 2) * th, x0 + (cx + bw / 2) * tw, y0 + (cy + bh / 2) * th])
                    cover.append(bw * bh * TILE * TILE)
                res[(key[0], key[1] + "_tiles")] = {"score": S.max(0), "tmean": S.mean(0), "n": np.sum([r["n"] for r, _, _ in per], 0).astype(np.int16),
                                                    "cover": np.array(cover, np.float32), "box": np.array(box, np.float32).round(1)}
            out.append(res)
            if self.guard:
                self.guard.after_object()
        return out

    def score(self, images: list[np.ndarray], keys=None) -> list[dict]:
        """images: RGB crops; keys: [(attr, variant)] (default: the first prompt set of every attribute + parts)
        -> [{(attr, variant): {"score": [P], "cover": [P], "n": [P], "box": [P, 4]}}]   (boxes in the crop's pixels)
        One prompt per forward pass (batch 1): ~1/8 of the VRAM of the earlier 8-prompt batches."""
        import time
        from PIL import Image
        torch = self.torch
        keys = keys or [(a, "mixed") for a in ATTRS] + [("parts", "parts")]
        out = []
        for im in images:
            pil0 = Image.fromarray(im)
            W, H = pil0.size
            pil = self._resize(pil0)
            res = {}
            for key in keys:
                t0 = time.time()
                inp = self.proc(images=pil, text=prompt_text(*key), return_tensors="pt", do_resize=False).to(self.device)
                inp["pixel_values"] = inp["pixel_values"].to(self.dtype)
                with torch.no_grad():
                    o = self.model(**inp)
                lg = o.logits[0].float().sigmoid()                  # [Q, 256]
                bx = o.pred_boxes[0].float()                        # [Q, 4] cx cy w h, relative to the image
                S = torch.stack([lg[:, sp].max(-1).values for sp in self._spans(key)], 1)   # [Q, P]
                best = S.argmax(0)
                bb = bx[best]
                xyxy = torch.stack([(bb[:, 0] - bb[:, 2] / 2) * W, (bb[:, 1] - bb[:, 3] / 2) * H,
                                    (bb[:, 0] + bb[:, 2] / 2) * W, (bb[:, 1] + bb[:, 3] / 2) * H], 1)
                res[key] = {"score": S.max(0).values.cpu().numpy().astype(np.float32),
                            "cover": (bb[:, 2] * bb[:, 3]).clamp(0, 1).cpu().numpy().astype(np.float32),
                            "n": (S >= N_BOX_THR).sum(0).cpu().numpy().astype(np.int16),
                            "box": xyxy.cpu().numpy().round(1).astype(np.float32)}
                del o, inp, lg, bx, S
                if self.guard:
                    self.guard.after_pass(time.time() - t0)
            out.append(res)
            if self.guard:
                self.guard.after_object()
        return out


def sys_path_glazeprep():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
