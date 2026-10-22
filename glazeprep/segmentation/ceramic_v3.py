"""Upgraded ceramic segmentation (v3), built on the same Grounding DINO + SAM 2.1 checkpoints.

Why a new module (see docs/CLAUDE_PIPELINE.md, "Code audit"):
  * legacy `GroundedSAM` drops every near-full-frame box/mask when smaller boxes exist, so
    frame-filling test tiles keep only a hole, drip or unglazed strip;
  * it uses SAM's single-mask output (often the rim ring of a plate);
  * NMS at IoU 0.6 deletes the inner (glazed) box of a plate;
  * Grounding DINO "finger" boxes on curled ridges are subtracted as hands (24 of the 52
    reviewer-labelled MISS_INNER_PART images carry `hand_removed`);
  * per-box masks are unioned, seam-closed and later re-split by connected components, so
    adjacent tiles merge and instance identity is lost.

Per image (every decision is appended to `result.decisions`):
  1. Grounding DINO proposals with a low floor threshold (recall first, filtered later) and a
     CLIP whole-image scene score (close-up / single object / multi-object / non-ceramic).
  2. SAM 2.1 encodes the image ONCE; each proposal box plus a full-frame box is decoded with
     multimask_output=True -> 3 candidates per prompt with SAM's predicted IoU.
  3. Hypothesis A "dominant object": a near-full-frame candidate is accepted only if it is not
     background-like, contains the strong detections, the photo is not a multi-object scene, and it
     does not merely group several similar separate objects.
  4. Hypothesis B "separate objects": per-detection candidates -> background rejection -> group-box
     removal (with recovery of undetected members) -> ceramic hole filling -> nesting / duplicate
     merge -> overlap resolution by SAM logits -> split of disconnected instances.
  5. Hand boxes are verified with CLIP before any pixel is subtracted; instances that CLIP rates as
     clearly non-ceramic (paper sheet, tape, wire ...) are removed and reported.
  6. Final masks are rebuilt at full resolution from SAM logits, guided by the work-resolution
     decision masks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi

from .. import config as C  # noqa: F401  (redirects HF / torch caches to D: before model loading)
from ..ceramic_clip import CeramicClip, region_crop
from .grounded_sam import EXCLUDE_PROMPT, OBJECT_PROMPT as LEGACY_OBJECT_PROMPT

FULL = -1  # detection index used for the synthetic full-frame prompt


@dataclass
class V3Params:
    object_prompt: str = LEGACY_OBJECT_PROMPT
    hand_prompt: str = EXCLUDE_PROMPT
    # ablation switches (scripts/12_evaluate_upgrade.py); the production pipeline uses all True
    enable_dominant: bool = True        # hypothesis A (frame-filling object) + its fallbacks
    enable_alternatives: bool = True    # SAM multimask alternatives for rings, ceramic hole filling
    enable_clip_checks: bool = True     # CLIP scene regime, hand verification, non-ceramic removal
    enable_refinement: bool = True      # merged-mass point split, object recovery after backdrop removal
    # proposals
    dino_floor: float = 0.15          # box/text threshold passed to Grounding DINO (recall first)
    rel_score: float = 0.30           # keep proposals >= rel_score * best score
    # instance seeds: only these detections may create a separate object. Multi-object scenes
    # (CLIP multi >= scene_multi_veto) use the low floor to recover every tile; other scenes keep the
    # legacy thresholds (box 0.24, relative 0.35), so weak boxes on background clutter only support
    # (merge into) objects found by stronger boxes.
    single_seed_abs: float = 0.24
    single_seed_rel: float = 0.35
    min_box_frac: float = 0.0015      # ignore boxes smaller than 0.15 % of the image
    dedup_box_iou: float = 0.92       # only near-identical boxes are de-duplicated before SAM
    max_boxes: int = 40
    work_side: int = 1024             # analysis resolution (longest side)
    # hypothesis A: dominant / frame-filling object
    near_full_box_frac: float = 0.80
    near_full_rel_score: float = 0.50
    dominant_min_area: float = 0.45
    dominant_min_iou: float = 0.80
    sub_rel_score: float = 0.45       # sub-detections considered as possible separate objects
    strong_rel_score: float = 0.70    # sub-detections a dominant candidate must not exclude
    excluded_area_max: float = 0.15   # excluded strong detections (area) relative to the candidate
    scene_multi_veto: float = 0.50
    group_cover_min: float = 0.85
    group_similar_ratio: float = 0.35
    zone_box_score_ratio: float = 0.90
    # inside a frame-filling candidate, >= 3 near-identical parts (min/max area >= 0.70) are separate
    # objects (a row of tiles); 2 parts or dissimilar parts are glaze zones of one object (flagged).
    dominant_group_min_kids: int = 4
    dominant_group_similar: float = 0.70
    surface_border_no_clip: float = 0.60  # multi-object scenes: a border-hugging mask holding objects is the backdrop
    iou_tie: float = 0.06             # dominant candidates within this IoU of the best are "equally confident"
    # low-SAM-confidence close-up fallback (always flagged for review)
    lowconf_closeup_scene: float = 0.40     # low-SAM-IoU full frame: CLIP close-up evidence needed
    lowconf_box_frac: float = 0.85
    lowconf_min_area: float = 0.60
    lowconf_min_border: float = 0.0
    lowconf_min_iou: float = 0.08
    lowconf_sure_closeup: float = 0.98      # ... above this, SAM's IoU estimate is ignored (706313)
    inner_closeup_scene: float = 0.90
    inner_dominant_min_iou: float = 0.85
    # a region holding the strongest detection only as an enclosed hole is a backdrop unless it is
    # detected about as strongly (relief dabs on a slab, 684208, vs a tile on a backdrop, 696223)
    backdrop_score_ratio: float = 0.75
    # whole image not ceramic (sheet of paper, 560308)
    nonceramic_scene_drop: float = 0.90
    # support surfaces in single-object scenes (wheel bat, table, cloth): a secondary instance that
    # spans the frame, touches >= 2 borders, is larger than the primary object and is detected much
    # more weakly is removed and flagged
    support_span: float = 0.90
    support_score_ratio: float = 0.60
    # merged-mass refinement (multi-object scenes only): an instance much larger than the typical
    # object whose crop CLIP also sees as several objects is re-segmented with a grid of SAM point
    # prompts; accepted only if compact, object-sized pieces explain most of it
    merge_oversize_factor: float = 3.0
    merge_min_area: float = 0.03
    merge_crop_multi: float = 0.50
    merge_grid_step: float = 0.5
    merge_max_points: int = 128
    merge_piece_min: float = 0.30
    merge_piece_max: float = 2.50
    merge_piece_iou: float = 0.80
    merge_piece_solidity: float = 0.80
    merge_min_cover: float = 0.50
    # object recovery after a large backdrop region is removed in a multi-object scene
    complement_min_removed: float = 0.10
    complement_min_p_ceramic: float = 0.50
    complement_default_ref: float = 0.01
    # hypothesis B: separate objects
    min_cand_iou: float = 0.55
    min_instance_frac: float = 0.0008
    min_instance_px: int = 300
    bg_border_cov: float = 0.50
    bg_center_cov: float = 0.35
    scene_cand_area: float = 0.85
    absorb_inside: float = 0.80
    dup_iou: float = 0.70
    child_max_rel_area: float = 0.70
    pinhole_frac: float = 0.01
    hole_bg_like_max: float = 0.40
    hole_cover_min: float = 0.60
    hole_bg_delta_e: float = 12.0
    speck_frac: float = 0.02
    split_min_frac: float = 0.12
    max_instances: int = 100
    alt_min_iou: float = 0.35         # SAM alternative used as "filled" version of a ring candidate
    alt_min_solidity: float = 0.80
    surface_min_enclosed: int = 2     # a border-touching mask enclosing >= 2 objects is a table/board
    bg_cluster_support: float = 0.10
    # hands
    hand_threshold: float = 0.35
    hand_min_clip: float = 0.50
    hand_object_iou: float = 0.30
    max_hand_removal: float = 0.50
    # non-ceramic removal (secondary instances only; the primary object is never deleted, only flagged)
    single_drop_p_ceramic: float = 0.35
    multi_drop_p_ceramic: float = 0.20
    oversize_factor: float = 3.0      # multi-object scenes: a secondary region this many times larger than
    oversize_min_area: float = 0.15   # the typical object (and >= 15 % of the image) that CLIP does not
    oversize_max_p_ceramic: float = 0.50  # rate as clearly ceramic is the table / board
    surface_max_p_ceramic: float = 0.35
    backdrop_colour_max_p_ceramic: float = 0.70   # multi-object scenes: secondary region along an image
    backdrop_colour_min_bg: float = 0.80          # edge whose pixels match the background colour model
    flag_p_ceramic: float = 0.35
    scene_nonceramic_flag: float = 0.50


@dataclass
class V3Instance:
    mask: np.ndarray                     # full resolution bool
    source: str                          # dominant | detection | split | group_member
    det_ids: list
    dino_score: float
    sam_iou: float
    p_ceramic: float | None = None
    flags: list = field(default_factory=list)
    primary: bool = False                # instance holding the strongest detection / the dominant object

    def box(self):
        ys, xs = np.where(self.mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@dataclass
class V3Result:
    instances: list
    union: np.ndarray
    mode: str
    detections: list
    decisions: list
    flags: list
    timings: dict
    scene: dict
    context: dict                        # work-resolution arrays for diagnostics (not serialised)

    def meta(self) -> dict:
        return {
            "segmentation_method": "v3_dino_sam2_multimask_hypothesis",
            "mode": self.mode,
            "n_instances": len(self.instances),
            "area_frac": round(float(self.union.mean()), 5),
            "scene_scores": self.scene,
            "instances": [{"index": k + 1, "box_xyxy": inst.box(), "source": inst.source,
                           "det_ids": inst.det_ids, "dino_score": round(inst.dino_score, 3),
                           "sam_iou": round(inst.sam_iou, 3),
                           "p_ceramic": None if inst.p_ceramic is None else round(inst.p_ceramic, 3),
                           "primary": inst.primary,
                           "area_frac": round(float(inst.mask.mean()), 5), "flags": inst.flags}
                          for k, inst in enumerate(self.instances)],
            "detections": self.detections,
            "decisions": self.decisions,
            "flags": self.flags,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }


# ----------------------------------------------------------------------------- small helpers
def _box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / (_box_area(a) + _box_area(b) - inter + 1e-6)


def _kernel(r):
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dilate(m, r):
    return cv2.dilate(m.astype(np.uint8), _kernel(r)).astype(bool)


def _erode(m, r):
    return cv2.erode(m.astype(np.uint8), _kernel(r)).astype(bool)


def _close(m, r):
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(r)).astype(bool) | m


def _hull(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    if cnts:
        cv2.fillPoly(out, [cv2.convexHull(np.vstack(cnts))], 1)
    return out.astype(bool)


def _resize_bool(m, wh):
    return cv2.resize(m.astype(np.uint8), wh, interpolation=cv2.INTER_NEAREST).astype(bool)


def _solidity(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c)) / hull if hull > 0 else 0.0


class _Cand:
    """One SAM candidate mask (work resolution) and its statistics."""
    __slots__ = ("pid", "det", "k", "iou", "obj", "m", "area", "border", "center", "score", "boxfrac")

    def __init__(self, pid, det, k, iou, obj, m, ring, center, score, boxfrac):
        self.pid, self.det, self.k, self.iou, self.obj, self.m = pid, det, k, iou, obj, m
        self.area = float(m.mean())
        self.border = float((m & ring).sum()) / max(1, int(ring.sum()))
        self.center = float(m[center].mean()) if m[center].size else 0.0
        self.score, self.boxfrac = score, boxfrac


class CeramicSegmenterV3:
    def __init__(self, params: V3Params | None = None, device: str | None = None,
                 clip: CeramicClip | None = None):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        self.p = params or V3Params()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dino_proc = AutoProcessor.from_pretrained(C.GROUNDING_DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_DINO_ID).to(self.device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(C.SAM2_ID)
        self.sam = Sam2Model.from_pretrained(C.SAM2_ID).to(self.device).eval()
        self.clip = clip or CeramicClip(device=self.device)

    # ------------------------------------------------------------------ models
    @torch.no_grad()
    def detect(self, img: Image.Image, prompt: str, thr: float):
        inputs = self.dino_proc(images=img, text=prompt, return_tensors="pt").to(self.device)
        out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=thr, text_threshold=thr, target_sizes=[img.size[::-1]])[0]
        boxes = res["boxes"].float().cpu().numpy().tolist()
        scores = res["scores"].float().cpu().numpy().tolist()
        raw_phrases = res["text_labels"] if "text_labels" in res else [""] * len(boxes)
        phrases = [str(x) for x in raw_phrases]
        W, H = img.size
        boxes = [[max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])] for b in boxes]
        return boxes, scores, phrases

    @torch.no_grad()
    def sam_embed(self, img: Image.Image):
        inputs = self.sam_proc(images=img, return_tensors="pt").to(self.device)
        return self.sam.get_image_embeddings(inputs["pixel_values"])

    @torch.no_grad()
    def sam_decode(self, emb, boxes, W, H):
        """boxes in original pixels -> (logits n x 3 x 256 x 256 [cpu], iou n x 3, obj n)."""
        if not boxes:
            return torch.zeros((0, 3, 256, 256)), np.zeros((0, 3)), np.zeros((0,))
        b = torch.tensor(boxes, dtype=torch.float32)
        b[:, [0, 2]] *= 1024.0 / W
        b[:, [1, 3]] *= 1024.0 / H
        logits, ious, objs = [], [], []
        for s in range(0, len(b), 16):
            out = self.sam(image_embeddings=emb, input_boxes=b[s:s + 16][None].to(self.device),
                           multimask_output=True)
            logits.append(out.pred_masks[0].float().cpu())
            ious.append(out.iou_scores[0].float().cpu().numpy())
            o = out.object_score_logits[0].float().cpu().numpy()
            objs.append(o.reshape(o.shape[0], -1)[:, 0])
        return torch.cat(logits, 0), np.concatenate(ious, 0), np.concatenate(objs, 0)

    @torch.no_grad()
    def sam_decode_points(self, emb, points, W, H):
        """single positive point prompts (original pixels) -> (logits n x 3 x 256 x 256 [cpu], iou n x 3)."""
        if not points:
            return torch.zeros((0, 3, 256, 256)), np.zeros((0, 3))
        pt = torch.tensor(points, dtype=torch.float32)
        pt[:, 0] *= 1024.0 / W
        pt[:, 1] *= 1024.0 / H
        logits, ious = [], []
        for s in range(0, len(pt), 32):
            chunk = pt[s:s + 32]
            out = self.sam(image_embeddings=emb, input_points=chunk[None, :, None, :].to(self.device),
                           input_labels=torch.ones((1, len(chunk), 1), dtype=torch.long, device=self.device),
                           multimask_output=True)
            logits.append(out.pred_masks[0].float().cpu())
            ious.append(out.iou_scores[0].float().cpu().numpy())
        return torch.cat(logits, 0), np.concatenate(ious, 0)

    # ------------------------------------------------------------------ main entry
    def __call__(self, rgb: np.ndarray) -> V3Result:
        p = self.p
        t = {}
        t0 = time.time()
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        s = min(1.0, p.work_side / max(H, W))
        wh, ww = max(8, round(H * s)), max(8, round(W * s))
        work = cv2.resize(rgb, (ww, wh), interpolation=cv2.INTER_AREA) if s < 1 else rgb.copy()
        lab = cv2.cvtColor(work.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        rw = max(2, round(0.015 * min(wh, ww)))
        ring = np.zeros((wh, ww), bool)
        ring[:rw], ring[-rw:], ring[:, :rw], ring[:, -rw:] = True, True, True, True
        center = (slice(int(0.3 * wh), max(int(0.3 * wh) + 1, int(0.7 * wh))),
                  slice(int(0.3 * ww), max(int(0.3 * ww) + 1, int(0.7 * ww))))
        decisions, flags = [], []

        # 1. proposals + scene -------------------------------------------------------------
        boxes, scores, phrases = self.detect(img, p.object_prompt, p.dino_floor)
        scene = (self.clip.scene_scores(Image.fromarray(work)) if p.enable_clip_checks
                 else {"closeup": 0.0, "single": 1.0, "multi": 0.0, "nonceramic": 0.0})
        t["dino_scene"] = time.time() - t0
        img_area = float(W * H)
        idx = [i for i in range(len(boxes)) if _box_area(boxes[i]) >= p.min_box_frac * img_area]
        best = max([scores[i] for i in idx], default=0.0)
        idx = [i for i in idx if scores[i] >= p.rel_score * best]
        idx.sort(key=lambda i: -scores[i])
        kept = []
        for i in idx:
            if all(_box_iou(boxes[i], boxes[j]) < p.dedup_box_iou for j in kept):
                kept.append(i)
        kept = kept[:p.max_boxes]
        dets = [{"box": [round(v, 1) for v in boxes[i]], "score": round(float(scores[i]), 4),
                 "phrase": phrases[i], "box_frac": round(_box_area(boxes[i]) / img_area, 4)} for i in kept]
        decisions.append(f"proposals raw={len(boxes)} kept={len(dets)} best={best:.3f}; scene={scene}")
        if scene.get("nonceramic", 0) >= p.scene_nonceramic_flag:
            flags.append("image_may_not_show_ceramic")
        multi_regime = scene.get("multi", 0) >= p.scene_multi_veto
        seed_abs = p.dino_floor if multi_regime else p.single_seed_abs
        seed_rel = p.rel_score if multi_regime else p.single_seed_rel
        for d in dets:
            d["seed"] = bool(d["score"] >= seed_abs and d["score"] >= seed_rel * best)
        if dets and not any(d["seed"] for d in dets):
            for d in dets:
                d["seed"] = bool(d["score"] >= 0.8 * best)
            flags.append("low_detection_confidence")
            decisions.append(f"no detection reaches the seed threshold {seed_abs:.2f}; strongest boxes used")
        decisions.append(f"regime={'multi-object' if multi_regime else 'single/close-up'} "
                         f"seeds={sum(d['seed'] for d in dets)}/{len(dets)} (abs>={seed_abs:.2f}, rel>={seed_rel:.2f})")

        # 2. SAM candidates ---------------------------------------------------------------
        t1 = time.time()
        emb = self.sam_embed(img)
        prompts = [d["box"] for d in dets] + [[0.0, 0.0, float(W), float(H)]]
        logits, ious, objs = self.sam_decode(emb, prompts, W, H)
        with torch.no_grad():
            masks = np.concatenate([
                (F.interpolate(logits[s0:s0 + 8].to(self.device), (wh, ww), mode="bilinear",
                               align_corners=False) > 0).cpu().numpy()
                for s0 in range(0, len(prompts), 8)], 0)                     # n x 3 x wh x ww
        t["sam"] = time.time() - t1
        t2 = time.time()
        n = len(dets)
        cands = []          # cands[pid][k]
        for pid in range(n + 1):
            det = pid if pid < n else FULL
            sc = dets[pid]["score"] if pid < n else 0.0
            bf = dets[pid]["box_frac"] if pid < n else 1.0
            cands.append([_Cand(pid, det, k, float(ious[pid, k]), float(objs[pid]), masks[pid, k],
                                ring, center, sc, bf) for k in range(3)])

        def top_iou(pid):
            row = cands[pid]
            return max(row, key=lambda c: (round(c.iou, 2), c.area))

        def bg_like(c):
            return c.border >= p.bg_border_cov and c.center < p.bg_center_cov

        sub, strong = [], []
        for pid in range(n):
            c = top_iou(pid)
            if (dets[pid]["box_frac"] < p.near_full_box_frac and dets[pid]["score"] >= p.sub_rel_score * best
                    and c.iou >= 0.70 and c.area >= 0.001 and not bg_like(c)):
                sub.append(c)
                if dets[pid]["score"] >= p.strong_rel_score * best and c.area >= 0.002:
                    strong.append(c)

        # the strongest detection, when it is a compact object (used by the backdrop rules)
        best_det_cand = None
        if n and dets[0]["box_frac"] < p.near_full_box_frac:
            c0 = top_iou(0)
            if c0.iou >= 0.70 and c0.area >= 0.002 and not bg_like(c0):
                best_det_cand = c0
        near_full_score = max([d["score"] for d in dets if d["box_frac"] >= p.near_full_box_frac], default=0.0)

        def encloses_best_as_hole(c):
            """c leaves the strongest detection out of its pixels but surrounds it (a hole)."""
            c0 = best_det_cand
            if c0 is None or c is c0 or c.pid == c0.pid or (c0.m & c.m).sum() >= 0.5 * c0.m.sum():
                return False
            filled = ndi.binary_fill_holes(_close(c.m, max(2, int(0.01 * max(wh, ww)))))
            return (c0.m & filled).sum() >= 0.7 * c0.m.sum()

        def support_score(c):
            return near_full_score if c.det == FULL else c.score

        def backdrop_like(c):
            # In a confident close-up a region around the strongest detection is another glaze zone
            # of the same surface (bowl interior with a centre pool, 682220), not a backdrop.
            return (scene.get("closeup", 0) < p.inner_closeup_scene and encloses_best_as_hole(c)
                    and support_score(c) < p.backdrop_score_ratio * best)

        # background colour model: border pixels not covered by any plausible object candidate
        objcover = np.zeros((wh, ww), bool)
        for pid in range(n):
            c = top_iou(pid)
            if c.iou >= p.min_cand_iou and not bg_like(c) and c.area < p.scene_cand_area:
                objcover |= c.m
        bg_centers = self._bg_model(lab, ring & ~objcover, ring)

        # supporting-surface test: a border-touching mask that encloses >= 2 detected objects as
        # holes is the table / board they lie on (not applied to close-ups of a single piece)
        closeup_scene = scene.get("closeup", 0) >= 0.5 and scene.get("multi", 0) < 0.3

        multi_scene = scene.get("multi", 0) >= p.scene_multi_veto

        surface_cache = {}

        def surface_like(c):
            """Table / board test: touches the border, holds >= 2 detected objects as holes, and CLIP
            does not see ceramic in it (a mass of touching tiles also encloses gaps)."""
            key = (c.pid, c.k)
            if key in surface_cache:
                return surface_cache[key]
            res = False
            # objects lying on the border lower the table's border coverage, hence the lower bar
            min_border = 0.35 if multi_scene else p.bg_border_cov
            if c.border >= min_border and not closeup_scene and sub:
                enclosed_area = ndi.binary_fill_holes(_close(c.m, max(2, int(0.01 * max(wh, ww)))))
                k = 0
                for o in sub:
                    if o is c:
                        continue
                    osz = max(1, int(o.m.sum()))
                    if (o.m & c.m).sum() < 0.3 * osz and (o.m & enclosed_area).sum() >= 0.7 * osz:
                        k += 1
                if k >= p.surface_min_enclosed and multi_scene and c.border >= p.surface_border_no_clip:
                    res = True
                    decisions.append(f"det{c.det}/k{c.k} hugs the border ({c.border:.2f}) and encloses {k} objects "
                                     f"in a multi-object scene -> supporting surface (background)")
                elif k >= p.surface_min_enclosed:
                    crop = region_crop(work, c.m)
                    pc = self.clip.ceramic_scores([crop])[0]["p_ceramic"] if crop is not None else 0.0
                    res = pc < p.surface_max_p_ceramic
                    decisions.append(f"det{c.det}/k{c.k} encloses {k} objects; CLIP p_ceramic={pc:.2f} -> "
                                     f"{'supporting surface (background)' if res else 'kept as possible ceramic'}")
            surface_cache[key] = res
            return res

        backdrop_logged = set()

        def background(c):
            if bg_like(c) or surface_like(c):
                return True
            if c.border >= 0.35 and backdrop_like(c):
                if (c.pid, c.k) not in backdrop_logged:
                    backdrop_logged.add((c.pid, c.k))
                    decisions.append(f"det{c.det}/k{c.k} surrounds the strongest detection as a hole and is "
                                     f"detected much more weakly ({support_score(c):.2f} vs {best:.2f}) "
                                     f"-> backdrop (background)")
                return True
            return False

        primary_cache = {}

        def primary(pid):
            """Top-IoU candidate, replaced by SAM's 'filled' alternative for ring / C-shaped masks."""
            if pid not in primary_cache:
                primary_cache[pid] = (self._prefer_filled_alt(cands, pid, top_iou(pid), background, top_iou,
                                                              n, bg_centers, lab, allow_cover=not multi_regime)
                                      if p.enable_alternatives else top_iou(pid))
            return primary_cache[pid]

        ctx = {"work_hw": (wh, ww), "scale": s, "lab": lab, "ring": ring, "center": center,
               "bg_centers": bg_centers, "dets": dets, "cands": cands, "sub": sub, "strong": strong,
               "best": best, "scene": scene, "work_rgb": work, "multi_regime": multi_regime,
               "best_det_cand": best_det_cand, "backdrop_like": backdrop_like}

        # 3. hypothesis A: dominant / frame-filling object --------------------------------
        dominant, dom_reason = (self._dominant(cands, dets, sub, strong, best, scene, background, ctx)
                                if p.enable_dominant else (None, ["hypothesis A disabled (ablation)"]))
        ctx["dominant"] = dominant
        decisions.extend(dom_reason)

        # 4. instances --------------------------------------------------------------------
        if dominant is not None:
            mode = "DOMINANT_OBJECT"
            if ctx.get("dominant_low_confidence"):
                flags.append("dominant_low_sam_confidence")
            others = self._hole_evidence(cands, dominant, background, primary, n)
            m = self._fill_holes(dominant.m, others, bg_centers, lab, decisions, "dominant", pad_border=True)
            insts = [{"m": m, "members": [(dominant.pid, dominant.k)], "source": "dominant",
                      "det_ids": [] if dominant.det == FULL else [dominant.det],
                      "score": dominant.score, "iou": dominant.iou, "flags": list(ctx.get("dominant_flags", [])),
                      "primary": True}]
            flags.extend(ctx.get("dominant_flags", []))
        else:
            mode = "SEPARATE_OBJECTS"
            insts = self._objects(cands, dets, n, primary, top_iou, background, bg_centers, lab, scene, decisions,
                                  multi_regime, flags, work)
            best_pid = 0 if n else None
            for d in insts:
                d["primary"] = best_pid is not None and best_pid in d["det_ids"]
            if insts and not any(d["primary"] for d in insts):
                max(insts, key=lambda d: (d["score"], int(d["m"].sum())))["primary"] = True
        for pid, c in primary_cache.items():
            if c is not top_iou(pid):
                decisions.append(f"det{pid}: used SAM alternative k{c.k} (area {c.area:.2f}, iou {c.iou:.2f}) "
                                 f"instead of ring/partial top-IoU mask")
        t["assemble"] = time.time() - t2

        # 5. hands (CLIP-verified) ----------------------------------------------------------
        t3 = time.time()
        hand_masks = []
        if insts:
            hb, hs, _ = self.detect(img, p.hand_prompt, p.hand_threshold)
            if hb:
                crops = []
                for b in hb:
                    pad = 0.1 * max(b[2] - b[0], b[3] - b[1])
                    crops.append(img.crop((max(0, int(b[0] - pad)), max(0, int(b[1] - pad)),
                                           min(W, int(b[2] + pad)), min(H, int(b[3] + pad)))))
                ph = self.clip.hand_scores(crops) if p.enable_clip_checks else [1.0] * len(hb)
                verified = [b for b, v in zip(hb, ph) if v >= p.hand_min_clip]
                rejected = len(hb) - len(verified)
                if rejected:
                    flags.append("hand_detection_rejected_by_clip")
                    decisions.append(f"{rejected} 'hand' box(es) rejected: CLIP p_hand < {p.hand_min_clip} "
                                     f"(scores {[round(v, 2) for v in ph]})")
                if verified:
                    hl, hi, _ = self.sam_decode(emb, verified, W, H)
                    with torch.no_grad():
                        hm = (F.interpolate(hl.to(self.device), (wh, ww), mode="bilinear",
                                            align_corners=False) > 0).cpu().numpy()
                    for j in range(len(verified)):
                        hand_masks.append(hm[j, int(np.argmax(hi[j]))])
            self._subtract_hands(insts, hand_masks, decisions, flags)
        ctx["hand_masks"] = hand_masks
        t["hands"] = time.time() - t3

        # 6. overlap resolution, merged-mass refinement, split, specks ----------------------
        insts = self._resolve_overlaps(insts, logits, (wh, ww))
        if multi_regime and mode == "SEPARATE_OBJECTS" and p.enable_refinement:
            t6 = time.time()
            insts = self._refine_merged(insts, emb, work, W, H, cands, dets, decisions, flags)
            t["refine_merged"] = time.time() - t6
        insts = self._split_and_clean(insts, decisions)
        insts = [d for d in insts if d["m"].sum() >= max(p.min_instance_px * s * s, p.min_instance_frac * wh * ww)]

        # 7. CLIP non-ceramic check ------------------------------------------------------
        t4 = time.time()
        crops = [region_crop(work, d["m"]) for d in insts]
        valid = [k for k, cimg in enumerate(crops) if cimg is not None]
        for k, sc in zip(valid, self.clip.ceramic_scores([crops[k] for k in valid]) if valid else []):
            insts[k]["p_ceramic"] = sc["p_ceramic"]
            insts[k]["top_non_ceramic"] = sc["top_non_ceramic"]
            if not p.enable_clip_checks:          # ablation: scores recorded, never acted on
                insts[k]["p_ceramic"] = None
        removed = []
        drop_thr = p.multi_drop_p_ceramic if multi_regime else p.single_drop_p_ceramic
        areas = [int(d["m"].sum()) for d in insts]
        prim = next((d for d in insts if d.get("primary")), None)
        for j, d in enumerate(insts):
            pc = d.get("p_ceramic")
            # The primary object (strongest detection / dominant object) is never deleted, only flagged.
            # In multi-object scenes the strongest box is often the board the tiles lie on (388290), so
            # there the protection only applies to a lone instance.
            if (d.get("primary") and not multi_regime) or len(insts) == 1:
                if pc is not None and pc < p.flag_p_ceramic:
                    d["flags"].append("possible_non_ceramic")
                continue
            if not multi_regime and prim is not None and self._support_like(d, prim, wh, ww):
                d["drop_reason"] = "support surface under / behind the primary object (wheel, table, cloth)"
                removed.append(d)
                flags.append("support_surface_removed")
                continue
            if pc is None:
                continue
            others = [a for k, a in enumerate(areas) if k != j]
            oversized = (multi_regime and others and areas[j] >= p.oversize_factor * float(np.median(others))
                         and areas[j] >= p.oversize_min_area * wh * ww and pc < p.oversize_max_p_ceramic)
            backdrop_colour = False
            if multi_regime and pc < p.backdrop_colour_max_p_ceramic and self._edge_hugging(d["m"]):
                bgf = self._bg_like_fraction(lab, d["m"], bg_centers)
                backdrop_colour = bgf is not None and bgf >= p.backdrop_colour_min_bg
            # NOTE: CLIP under-rates some white / cream glazes (scores them as paper or cloth, 704116). A
            # protection by Grounding DINO score was tested and rejected: printed recipe sheets get DINO
            # scores as high as such tiles (803226), so it would bring the paper back.
            if pc < drop_thr or oversized or backdrop_colour:
                d["drop_reason"] = ("oversized low-ceramic region (table/board)" if oversized
                                    else "edge region with the background colour (paper/table)" if backdrop_colour
                                    else f"p_ceramic {pc:.2f} < {drop_thr:.2f}")
                removed.append(d)
            elif pc < p.flag_p_ceramic:
                d["flags"].append("possible_non_ceramic")
        if removed:
            for d in removed:
                pc = d.get("p_ceramic")
                decisions.append(f"removed secondary instance ({d['drop_reason']}); CLIP p_ceramic="
                                 f"{'n/a' if pc is None else f'{pc:.2f}'}, looks like '{d.get('top_non_ceramic')}'")
            flags.append("non_ceramic_region_removed")
            gone = {id(d) for d in removed}
            insts = [d for d in insts if id(d) not in gone]
        ctx["removed_non_ceramic"] = [d["m"] for d in removed]
        # objects hidden inside a removed backdrop region (tiles cut out of a paper sheet, 388290)
        big_removed = [d for d in removed if d["m"].sum() >= p.complement_min_removed * wh * ww]
        if multi_regime and mode == "SEPARATE_OBJECTS" and big_removed and p.enable_refinement:
            rec = self._recover_complement(insts, big_removed, emb, work, W, H, decisions)
            if rec:
                insts.extend(rec)
                flags.append("objects_recovered_after_backdrop_removal")
        if mode == "DOMINANT_OBJECT" and insts and (insts[0].get("p_ceramic") or 1.0) < p.flag_p_ceramic \
                and scene.get("nonceramic", 0) >= p.scene_nonceramic_flag:
            flags.append("dominant_object_may_not_be_ceramic")
        # CLIP is near-certain the whole photo is not ceramic (sheet of paper, screenshot, bag) and no
        # region looks ceramic either: report no ceramic target instead of exporting a wrong object
        if (insts and p.enable_clip_checks and scene.get("nonceramic", 0) >= p.nonceramic_scene_drop
                and all(d.get("p_ceramic") is not None and d["p_ceramic"] < p.flag_p_ceramic for d in insts)):
            decisions.append(f"image judged not to show ceramic (CLIP non-ceramic scene "
                             f"{scene['nonceramic']:.2f}, all {len(insts)} region(s) p_ceramic < {p.flag_p_ceramic}); "
                             f"no object exported")
            flags.append("image_not_ceramic")
            removed.extend(insts)
            insts = []
        t["clip"] = time.time() - t4

        if not insts:
            mode = "NO_CERAMIC_TARGET" if (removed or dets) else "NO_DETECTION"
            flags.append(mode.lower())

        insts.sort(key=lambda d: -int(d["m"].sum()))
        if len(insts) > p.max_instances:
            flags.append("instance_limit_reached")
            decisions.append(f"instance cap {p.max_instances} reached ({len(insts)} found); smallest dropped")
            insts = insts[:p.max_instances]

        # 8. full-resolution reconstruction ---------------------------------------------
        t5 = time.time()
        final = []
        for d in insts:
            full = self._full_res(d, logits, H, W, s)
            if full.sum() < p.min_instance_px:
                continue
            final.append(V3Instance(mask=full, source=d["source"], det_ids=d["det_ids"],
                                    dino_score=float(d["score"]), sam_iou=float(d["iou"]),
                                    p_ceramic=d.get("p_ceramic"), flags=list(d["flags"]),
                                    primary=bool(d.get("primary"))))
        union = np.zeros((H, W), bool)
        for inst in final:
            union |= inst.mask
        t["full_res"] = time.time() - t5
        ctx["instances_work"] = [d["m"] for d in insts]
        if len(final) > 1:
            flags.append("multi_object")
        t["total"] = time.time() - t0
        return V3Result(instances=final, union=union, mode=mode, detections=dets, decisions=decisions,
                        flags=sorted(set(flags)), timings=t, scene=scene, context=ctx)

    # ------------------------------------------------------------------ hypothesis A
    def _dominant(self, cands, dets, sub, strong, best, scene, bg_like, ctx):
        """Hypothesis A. Candidate sources, tried in this order:
          1. confident frame-filling SAM masks of near-full DINO boxes (or of the full-frame prompt);
          2. CLIP is sure the photo is one close-up surface and the strongest box is (nearly) the whole
             frame, but SAM's IoU estimate is low (uniform / smooth glaze; 706349) -> flagged;
          3. the same close-up situation where every seed box is near-full but SAM cannot segment the
             frame at all (small images; 33711): a large, high-IoU mask of an inner box -> flagged.
        """
        p = self.p
        n = len(dets)
        reasons = []
        checked = []
        ctx["dominant_candidates_checked"] = checked
        closeup = scene.get("closeup", 0)
        no_multi = scene.get("multi", 0) < 0.10

        def try_pool(pool, label, low_conf=False):
            for c in pool:
                ok, why, extra = self._dominant_checks(c, sub, strong, scene, bg_like, ctx)
                tag = f"{'FULL' if c.det == FULL else 'det' + str(c.det)}/k{c.k} area={c.area:.2f} iou={c.iou:.2f}"
                checked.append((c, f"{label}:{why}"))
                if ok:
                    reasons.append(f"dominant ACCEPTED [{label}] {tag} ({why})")
                    ctx["dominant_flags"] = extra
                    ctx["dominant_low_confidence"] = low_conf
                    return c
                reasons.append(f"dominant rejected [{label}] {tag}: {why}")
            return None

        # 1. confident near-full candidates
        near = [pid for pid in range(n) if dets[pid]["box_frac"] >= p.near_full_box_frac
                and dets[pid]["score"] >= max(p.dino_floor, p.near_full_rel_score * best)]
        allc = [c for pid in near for c in cands[pid]] + list(cands[n])
        pool = [c for c in allc if c.iou >= p.dominant_min_iou and c.area >= p.dominant_min_area]
        # the full-frame prompt without DINO near-full support needs stronger SAM evidence
        pool = [c for c in pool if c.det != FULL or near or (c.iou >= 0.90 and c.area >= 0.70)]
        # among equally confident candidates prefer the most complete one
        top = max((c.iou for c in pool), default=0.0)
        pool.sort(key=lambda c: (c.iou < top - p.iou_tie, -c.area if c.iou >= top - p.iou_tie else -c.iou))
        if not pool:
            reasons.append("no near-full-frame candidate with sufficient SAM confidence")
        c = try_pool(pool, "confident")
        if c is not None or n == 0 or not no_multi:
            return c, reasons
        seeds = [i for i in range(n) if dets[i].get("seed")]
        only_full_seeds = all(dets[i]["box_frac"] >= p.near_full_box_frac for i in seeds)
        # 2. low-confidence full-frame mask of the strongest (near-full) box; only when the detector
        #    proposes nothing smaller (every seed box is near-full) and CLIP sees a close-up
        rank0 = max(range(n), key=lambda i: dets[i]["score"])
        # confident close-up (>= 0.90): as before; weaker close-up evidence (>= 0.40, white tiles that
        # CLIP rates half "single object", 567629) only if the detector proposes nothing smaller
        lowconf_ok = dets[rank0]["box_frac"] >= p.lowconf_box_frac and (
            closeup >= p.inner_closeup_scene or (closeup >= p.lowconf_closeup_scene and only_full_seeds))

        def lowconf(min_iou, label):
            low = [c for c in list(cands[rank0]) + list(cands[n]) if c.area >= p.lowconf_min_area
                   and c.border >= p.lowconf_min_border and c.iou >= min_iou]
            reasons.append(f"{label} fallback (CLIP close-up {closeup:.2f}, min SAM IoU {min_iou:.2f}): "
                           f"{len(low)} candidate(s)")
            return try_pool(sorted(low, key=lambda c: -c.area), label, low_conf=True)

        if lowconf_ok:
            c = lowconf(p.lowconf_min_iou, "low-confidence")
            if c is not None:
                return c, reasons
        # 3. inner-box candidates, only when every seed box is near-full AND none of them yields a mask
        #    the separate-objects path could use (706351 is handled better by that path)

        def usable(i):
            c = max(cands[i], key=lambda c: (round(c.iou, 2), c.area))
            return c.iou >= p.min_cand_iou and not bg_like(c) and c.area < p.scene_cand_area

        if closeup >= p.inner_closeup_scene and only_full_seeds and not any(usable(i) for i in seeds):
            inner = [c for pid in range(n) if pid not in near for c in cands[pid]
                     if c.iou >= p.inner_dominant_min_iou and c.area >= p.dominant_min_area]
            reasons.append(f"inner-box close-up fallback: {len(inner)} candidate(s)")
            c = try_pool(sorted(inner, key=lambda c: (-round(c.iou, 2), -c.area)), "inner-box", low_conf=True)
            if c is not None:
                return c, reasons
        # 4. CLIP is near-certain the photo is one close-up surface but SAM's IoU estimate collapsed
        #    (uniform glaze filling the frame, 706313): the frame-filling mask is used, flagged
        if lowconf_ok and closeup >= p.lowconf_sure_closeup:
            c = lowconf(0.0, "sure-close-up")
            if c is not None:
                return c, reasons
        return None, reasons

    def _dominant_checks(self, c, sub, strong, scene, bg_like, ctx):
        """-> (accept, reason, review_flags) for one frame-filling candidate."""
        p = self.p
        if bg_like(c):
            return False, f"background-like (border={c.border:.2f}, centre={c.center:.2f})", []
        csz = max(1, int(c.m.sum()))
        excluded = sum(int(sc.m.sum()) for sc in strong if (sc.m & c.m).sum() < 0.7 * sc.m.sum())
        if excluded >= p.excluded_area_max * csz:
            return False, f"excludes strong detections ({excluded / csz:.2f} of its area)", []
        if ctx.get("multi_regime") and len(sub) >= 2:
            return False, f"multi-object scene (CLIP multi={scene['multi']:.2f}) with {len(sub)} separate detections", []
        # The strongest detection, if it is a compact object, must not be cut out of the candidate unless
        # the candidate encloses it (dabs on a slab). A candidate that leaves it open to the frame edge is
        # the surroundings of an object held or placed in front of it (233147: tile in hand).
        # CLIP region scores are NOT used here: cream / white tiles score as paper or tape (2343).
        c0 = ctx.get("best_det_cand")
        if c0 is not None and c0 is not c and (c0.m & c.m).sum() < 0.5 * c0.m.sum():
            filled = ndi.binary_fill_holes(_close(c.m, max(2, int(0.01 * max(c.m.shape)))))
            if (c0.m & filled).sum() < 0.7 * c0.m.sum():
                return False, "excludes the strongest detection, which it does not enclose", []
            if ctx["backdrop_like"](c):
                return False, ("surrounds the strongest detection as a hole but is detected much more "
                               "weakly -> backdrop around an object"), []
        kids = [sc for sc in sub if (sc.m & c.m).sum() >= 0.8 * sc.m.sum() and sc.area <= p.child_max_rel_area * c.area]
        kids = [a for a in kids if not any(b is not a and (a.m & b.m).sum() >= 0.8 * a.m.sum() and b.area > a.area for b in kids)]
        if len(kids) >= 2 and scene.get("multi", 0) >= p.scene_multi_veto:
            return False, f"multi-object scene (CLIP multi={scene['multi']:.2f}, {len(kids)} sub-objects)", []
        kid_union = np.zeros_like(c.m)
        for sc in kids:
            kid_union |= sc.m
        covered = float((kid_union & c.m).sum()) / csz
        extra = []
        if len(kids) >= 2 and covered >= p.group_cover_min:
            areas = sorted(sc.area for sc in kids)
            ratio = areas[0] / areas[-1]
            if len(kids) >= p.dominant_group_min_kids and ratio >= p.dominant_group_similar:
                return False, f"{len(kids)} near-identical separate objects cover {covered:.2f} (size ratio {ratio:.2f})", []
            # two parts (or dissimilar parts) tiling the frame: usually glaze zones of one tile,
            # occasionally two touching tiles -> keep as one object, ask for review
            extra.append("possible_touching_objects_merged")
        return True, f"sub-objects inside={len(kids)}, covered={covered:.2f}", extra

    # ------------------------------------------------------------------ hypothesis B
    def _objects(self, cands, dets, n, primary, top_iou, bg_like, bg_centers, lab, scene, decisions,
                 multi_regime, flags, work):
        p = self.p
        wh, ww = lab.shape[:2]
        min_area = max(p.min_instance_frac, p.min_instance_px / float(wh * ww))
        pool = []
        for pid in range(n):
            c = primary(pid)
            # prompt quality is judged on SAM's best output, even when a filled alternative is used
            if top_iou(pid).iou < p.min_cand_iou or c.obj < 0 or c.area < min_area:
                continue
            if bg_like(c):
                decisions.append(f"det{pid} candidate rejected: background-like (border={c.border:.2f})")
                continue
            if dets[pid]["box_frac"] >= p.near_full_box_frac and c.area >= p.scene_cand_area:
                decisions.append(f"det{pid} candidate rejected: whole-scene mask area={c.area:.2f}")
                continue
            pool.append(c)
        # group / container boxes
        pool.sort(key=lambda c: -c.area)
        drop, recovered = set(), []
        multi = scene.get("multi", 0)
        rfill = max(2, int(0.01 * max(wh, ww)))
        rtouch = rfill                       # parts separated by a glaze lip still count as touching
        for X in pool:
            # In multi-object scenes the children may be holes of X (tiles cut out of a board mask,
            # 32817), so containment is tested against X's filled outline.
            Xm = ndi.binary_fill_holes(_close(X.m, rfill)) if multi_regime else X.m
            # a seed box is only replaced by children that can themselves become objects (seeds)
            kids = [Y for Y in pool if Y is not X and id(Y) not in drop and (Y.m & Xm).sum() >= 0.8 * Y.m.sum()
                    and Y.area <= p.child_max_rel_area * X.area and (dets[Y.det]["seed"] or not dets[X.det]["seed"])]
            kids = [a for a in kids if not any(b is not a and (a.m & b.m).sum() >= 0.8 * a.m.sum() and b.area > a.area for b in kids)]
            if len(kids) < 2:
                continue
            ku = np.zeros_like(X.m)
            for Y in kids:
                ku |= Y.m
            cover = float((ku & Xm).sum()) / max(1, int(Xm.sum()))
            areas = sorted(Y.area for Y in kids)
            similar = areas[0] / areas[-1] >= p.group_similar_ratio
            if multi_regime:
                # multi-object scenes: boards / tables hold touching tiles of any size, so a box whose
                # mask is explained by its children is a group box (zone rules below do not apply)
                is_group = ((cover >= p.group_cover_min and similar) or cover >= 0.35
                            or (len(kids) >= 3 and similar and cover >= 0.50))
                if is_group:
                    self._drop_group(X, kids, ku, cover, multi, drop, recovered, decisions, Xm)
                continue
            touching = any(self._touching(a.m, b.m, rtouch) for i, a in enumerate(kids) for b in kids[i + 1:])
            if similar:
                # similar-sized children: separate objects (rows of tiles, sets of cups)
                is_group = (cover >= p.group_cover_min or (multi >= p.scene_multi_veto and cover >= 0.35)
                            or (len(kids) >= 3 and cover >= 0.50))
                if is_group and len(kids) == 2 and touching and not multi_regime \
                        and X.score >= p.zone_box_score_ratio * max(Y.score for Y in kids):
                    # Two touching, both-ceramic parts in a box that DINO scores about as high as its best
                    # part: glaze zones of one object. A tile in front of a paper stack (386537) fails
                    # the ceramic test and stays split.
                    crops = [region_crop(work, Y.m) for Y in kids]
                    pcs = [s["p_ceramic"] for s in self.clip.ceramic_scores([c for c in crops if c is not None])]
                    if len(pcs) == 2 and min(pcs) >= 0.5:
                        decisions.append(f"det{X.det}: 2 touching ceramic parts, box scored like its parts "
                                         f"({X.score:.2f} vs {max(Y.score for Y in kids):.2f}) -> one object")
                        is_group = False
            else:
                # dissimilar children (glazed top, bare foot, base strip of one tile, 667422) are zones
                # of one object unless they are spatially separate
                is_group = (not touching) and cover >= 0.50
                if not is_group:
                    decisions.append(f"det{X.det}: {len(kids)} touching parts of different size -> zones of one "
                                     f"object (not split)")
            if is_group:
                self._drop_group(X, kids, ku, cover, multi, drop, recovered, decisions, Xm)
        pool = [c for c in pool if id(c) not in drop]
        # NOTE: rejecting large near-full-frame masks here (they were already rejected as the dominant
        # object) was tested and reverted: it fixed 689166 (sculpture + backdrop) but removed the main
        # object in 7972 (mug), 529979 (bowl) and 432716 (tile).
        if recovered:
            decisions.append(f"recovered {len(recovered)} undetected member(s) inside group boxes")
        # ceramic hole filling per candidate. In multi-object scenes a hole occupied by another
        # detected object is that object (a tile on a board), never part of this one.
        items = []
        for c in pool:
            others = self._hole_evidence(cands, c, bg_like, primary, n)
            m = self._fill_holes(c.m, others, bg_centers, lab, decisions, f"det{c.det}",
                                 objects_are_separate=multi_regime)
            items.append({"m": m, "members": [(c.pid, c.k)], "source": "detection", "det_ids": [c.det],
                          "score": c.score, "iou": c.iou, "flags": []})
        for r, X in recovered:
            if dets[X.det]["seed"]:
                items.append({"m": r, "members": [(X.pid, X.k)], "source": "group_member", "det_ids": [X.det],
                              "score": X.score, "iou": X.iou, "flags": ["recovered_from_group_box"]})
        # nesting / duplicates / hole-filling parts. Seed detections create objects (largest first);
        # weak (non-seed) detections may only merge into an existing object.
        for d in items:
            d["seed"] = any(dets[i]["seed"] for i in d["det_ids"])
        if items and not any(d["seed"] for d in items):
            # every seed box was rejected (background / whole scene): fall back to the strongest
            # remaining boxes (within 85 % of the best remaining score)
            top = max(d["score"] for d in items)
            for d in items:
                d["seed"] = d["score"] >= 0.85 * top
            flags.append("weak_detection_fallback")
            decisions.append("no seed detection produced a valid mask; strongest weaker detections used (review)")
        items.sort(key=lambda d: (not d["seed"], -int(d["m"].sum())))
        insts = []
        weak_dropped = []
        rclose = max(2, int(0.02 * max(wh, ww)))
        for d in items:
            a = int(d["m"].sum())
            placed = False
            for I in insts:
                inter = int((d["m"] & I["m"]).sum())
                union = a + int(I["m"].sum()) - inter
                nested = inter >= p.absorb_inside * a or inter / max(1, union) >= p.dup_iou
                why = "nested/duplicate"
                # in multi-object scenes an enclosing mask is more likely a board than a plate rim, so
                # hole merging there needs a ring with a single hole (plate + centre pool, 2602)
                if not nested and inter < 0.5 * a and multi >= p.scene_multi_veto:
                    if "_nholes" not in I:
                        I["_nholes"] = self._n_holes(I["m"], rclose)
                    if "_nholes" not in d:
                        d["_nholes"] = self._n_holes(d["m"], rclose)
                    hi, hd = I["_nholes"], d["_nholes"]
                    ring_pair = (hi == 1 and hd == 0) or (hd == 1 and hi == 0)
                else:
                    ring_pair = True
                if not nested and inter < 0.5 * a and ring_pair:
                    if "_filled" not in I:
                        I["_filled"] = ndi.binary_fill_holes(_close(I["m"], rclose))
                    if int((d["m"] & I["_filled"]).sum()) >= 0.85 * a:
                        nested, why = True, "fills an enclosed hole of"
                    else:
                        # the reverse: the existing object fills a hole of this ring (bowl interior + rim)
                        d_filled = ndi.binary_fill_holes(_close(d["m"], rclose))
                        isz = max(1, int(I["m"].sum()))
                        if int((I["m"] & d_filled).sum()) >= 0.85 * isz and d["seed"]:
                            nested, why = True, "is the rim enclosing"
                if nested:
                    decisions.append(f"det{d['det_ids'][0]} (area {a / float(wh * ww):.3f}) merged: {why} "
                                     f"det{I['det_ids'][0]}")
                    # A duplicate or weak detection only supports the object: its pixels are not added,
                    # because a sloppier mask of the same object usually carries table / shadow pixels
                    # (549668). Pixels are added for seed masks nested inside the object (handles, rims)
                    # and for hole / rim completion.
                    grow = why != "nested/duplicate" or (d["seed"] and inter >= p.absorb_inside * a)
                    if grow:
                        I["m"] = I["m"] | d["m"]
                        I.pop("_nholes", None)
                    I.pop("_filled", None)
                    I["members"] += d["members"]
                    I["det_ids"] += d["det_ids"]
                    I["score"] = max(I["score"], d["score"])
                    placed = True
                    break
            if not placed:
                if d["seed"]:
                    insts.append(d)
                else:
                    weak_dropped.append(d["det_ids"][0])
        if weak_dropped:
            decisions.append(f"weak detections not used as separate objects: "
                             f"{['det%d (s=%.2f)' % (i, dets[i]['score']) for i in weak_dropped]}")
        for I in insts:
            I.pop("_filled", None)
            I.pop("_nholes", None)
            I.pop("seed", None)
        return insts

    # ------------------------------------------------------------------ refinements
    def _prefer_filled_alt(self, cands, pid, c, background, top_iou, n, bg_centers, lab, allow_cover=True):
        """SAM's highest-IoU output for a plate / bowl is often only the rim ring (or a C-shape when
        the rim leaves the frame). Use a larger alternative output of the same prompt when it
        contains the ring, the pixels it adds lie inside the ring's convex hull, it is compact, and
        the added pixels are not background (colour model) or are covered by another detection."""
        p = self.p
        if c.area <= 0 or background(c):
            return c
        csz = int(c.m.sum())
        hull = None
        for alt in sorted(cands[pid], key=lambda a: -a.area):
            if (alt is c or alt.area < 1.3 * c.area or alt.iou < p.alt_min_iou or alt.area >= p.scene_cand_area
                    or background(alt) or (alt.m & c.m).sum() < 0.9 * csz):
                continue
            added = alt.m & ~c.m
            asz = max(1, int(added.sum()))
            if hull is None:
                hull = _hull(c.m)
            if (added & hull).sum() < 0.8 * asz or _solidity(alt.m) < p.alt_min_solidity:
                continue
            bgf = self._bg_like_fraction(lab, added, bg_centers)
            cover = 0.0
            for q in range(n) if allow_cover else ():
                if q == pid:
                    continue
                o = top_iou(q)
                if o.iou >= p.min_cand_iou and not background(o) and o.area < p.scene_cand_area:
                    cover = max(cover, float((added & o.m).sum()) / asz)
            if ((bgf is not None and bgf <= p.hole_bg_like_max)
                    or (cover >= p.hole_cover_min and (bgf is None or bgf <= 0.75))
                    or (bgf is None and cover >= 0.30)):
                return alt
        return c

    def _refine_merged(self, insts, emb, work, W, H, cands, dets, decisions, flags):
        """Split instances that are clusters of touching objects (MULTI_OBJECT_MERGED)."""
        p = self.p
        wh, ww = work.shape[:2]
        areas = [int(d["m"].sum()) for d in insts]
        out = []
        for j, d in enumerate(insts):
            a = areas[j]
            others = [x for k, x in enumerate(areas) if k != j and x > 0]
            if len(others) >= 2:
                ref = float(np.median(others))
            else:   # typical object size from the smaller seed detections
                small = [int(max(cands[i], key=lambda c: c.iou).m.sum()) for i in range(len(dets))
                         if dets[i].get("seed") and 0 < max(cands[i], key=lambda c: c.iou).m.sum() < 0.25 * a]
                ref = float(np.median(small)) if len(small) >= 2 else None
            if ref is None or a < p.merge_oversize_factor * ref or a < p.merge_min_area * wh * ww:
                out.append(d)
                continue
            crop = region_crop(work, d["m"])
            sc = self.clip.scene_scores(crop) if crop is not None else {"multi": 0.0}
            if sc["multi"] < p.merge_crop_multi:
                decisions.append(f"large instance det{d['det_ids'][0]} ({a / (wh * ww):.3f}) not split: "
                                 f"its crop does not look like several objects (CLIP multi={sc['multi']:.2f})")
                out.append(d)
                continue
            res = self._point_split(d["m"], ref, emb, W, H, wh, ww)
            if res is None:
                decisions.append(f"large instance det{d['det_ids'][0]} ({a / (wh * ww):.3f}, crop multi="
                                 f"{sc['multi']:.2f}) kept: point prompts gave no consistent object pieces")
                d["flags"].append("possible_merged_objects")
                flags.append("possible_merged_objects")
                out.append(d)
                continue
            pieces, cover = res
            decisions.append(f"merged mass det{d['det_ids'][0]} ({a / (wh * ww):.3f}) split into {len(pieces)} "
                             f"objects by SAM point prompts (cover {cover:.2f}, crop multi={sc['multi']:.2f})")
            flags.append("merged_mass_split")
            union = np.zeros_like(d["m"])
            for k, (m, lg, iou) in enumerate(pieces):
                union |= m
                out.append({"m": m, "members": [], "extra_logits": [lg], "source": "point_split",
                            "det_ids": list(d["det_ids"]), "score": d["score"], "iou": float(iou),
                            "flags": ["split_from_merged_mass"], "primary": bool(d.get("primary")) and k == 0})
            residual = d["m"] & ~_dilate(union, 2)
            lab_r, nr = ndi.label(residual)
            kept_res, kept_px = 0, 0
            for k in range(1, nr + 1):
                r = lab_r == k
                if p.merge_piece_min * ref <= r.sum() <= p.merge_piece_max * ref and _solidity(r) >= p.merge_piece_solidity:
                    out.append({"m": r, "members": list(d["members"]), "source": "point_split_residual",
                                "det_ids": list(d["det_ids"]), "score": d["score"], "iou": d["iou"],
                                "flags": ["split_from_merged_mass"], "primary": False})
                    kept_res += 1
                    kept_px += int(r.sum())
            dropped = float(int(residual.sum()) - kept_px) / max(1, a)
            decisions.append(f"  residual of merged mass: {kept_res} object-sized part(s) kept, "
                             f"{dropped:.1%} of its pixels dropped as gaps/background")
        return out

    @staticmethod
    def _drop_group(X, kids, ku, cover, multi, drop, recovered, decisions, Xm=None):
        """X is a group / container box: drop it, keep its children, recover undetected members."""
        drop.add(id(X))
        decisions.append(f"det{X.det} treated as group box ({len(kids)} children cover {cover:.2f}, "
                         f"scene multi={multi:.2f}); children kept")
        # Recover undetected members: compact residual blobs of member size. Not done when X is a
        # backdrop holding its children as holes (its residual is the board, not objects).
        if Xm is not None and Xm.sum() > 1.15 * X.m.sum():
            return
        residual = X.m & ~_dilate(ku, 3)
        lab_r, nr = ndi.label(residual)
        if nr:
            med = float(np.median([int(Y.m.sum()) for Y in kids]))
            mx = max(int(Y.m.sum()) for Y in kids)
            for k in range(1, nr + 1):
                r = lab_r == k
                a = int(r.sum())
                if 0.5 * med <= a <= 2.0 * mx and _solidity(r) >= 0.75:
                    recovered.append((r, X))

    def _recover_complement(self, insts, removed, emb, work, W, H, decisions):
        """Propose objects in the area that is neither a removed backdrop nor an accepted object."""
        p = self.p
        wh, ww = work.shape[:2]
        backdrop = np.zeros((wh, ww), bool)
        for d in removed:
            backdrop |= d["m"]
        taken = np.zeros((wh, ww), bool)
        for d in insts:
            taken |= d["m"]
        hull = _hull(backdrop)
        region = hull & ~_dilate(backdrop, 2) & ~_dilate(taken, 2)
        if region.sum() < 0.02 * wh * ww:
            return []
        sizes = [int(d["m"].sum()) for d in insts]
        ref = float(np.median(sizes)) if sizes else p.complement_default_ref * wh * ww
        res = self._point_split(region, ref, emb, W, H, wh, ww, min_pieces=1)
        if res is None:
            decisions.append(f"backdrop removed; no object-sized pieces found in the remaining "
                             f"{region.mean():.1%} of the image")
            return []
        pieces, cover = res
        crops = [region_crop(work, m) for m, _, _ in pieces]
        ok = [k for k, c in enumerate(crops) if c is not None]
        pcs = dict(zip(ok, self.clip.ceramic_scores([crops[k] for k in ok]))) if ok else {}
        out = []
        for k, (m, lg, iou) in enumerate(pieces):
            pc = pcs.get(k, {}).get("p_ceramic", 0.0)
            if pc < p.complement_min_p_ceramic:
                continue
            out.append({"m": m, "members": [], "extra_logits": [lg], "source": "backdrop_complement",
                        "det_ids": [], "score": 0.0, "iou": float(iou), "p_ceramic": pc,
                        "flags": ["recovered_after_backdrop_removal"], "primary": False})
        decisions.append(f"backdrop removed: {len(out)} object(s) recovered by SAM point prompts in the "
                         f"remaining area ({len(pieces)} candidates, {len(pieces) - len(out)} rejected by CLIP)")
        return out

    def _point_split(self, m, ref, emb, W, H, wh, ww, min_pieces=2):
        p = self.p
        step = max(6, int(np.sqrt(ref) * p.merge_grid_step))
        inner = _erode(m, max(1, step // 4))
        ys, xs = np.mgrid[step // 2:wh:step, step // 2:ww:step]
        pts = [(int(x), int(y)) for y, x in zip(ys.ravel(), xs.ravel()) if inner[y, x]]
        if len(pts) < 2:
            return None
        if len(pts) > p.merge_max_points:
            sel = np.linspace(0, len(pts) - 1, p.merge_max_points).round().astype(int)
            pts = [pts[i] for i in sel]
        sx, sy = W / float(ww), H / float(wh)
        logits, ious = self.sam_decode_points(emb, [[x * sx, y * sy] for x, y in pts], W, H)
        low_area = (logits > 0).float().mean(dim=(2, 3)).numpy()
        ref_frac = ref / float(wh * ww)
        chosen = []
        for i in range(len(pts)):
            ok = [k for k in range(3) if p.merge_piece_min * ref_frac <= low_area[i, k] <= p.merge_piece_max * ref_frac]
            if ok:
                k = max(ok, key=lambda k: ious[i, k])
                if ious[i, k] >= p.merge_piece_iou:
                    chosen.append((i, k, float(ious[i, k])))
        if len(chosen) < min_pieces:
            return None
        chosen.sort(key=lambda c: -c[2])
        kept = []
        taken = np.zeros_like(m)
        for s0 in range(0, len(chosen), 16):
            part = chosen[s0:s0 + 16]
            sel = torch.stack([logits[i, k] for i, k, _ in part])[:, None]
            with torch.no_grad():
                up = (F.interpolate(sel.to(self.device), (wh, ww), mode="bilinear",
                                    align_corners=False)[:, 0] > 0).cpu().numpy()
            for (i, k, iou), mk in zip(part, up):
                sz = int(mk.sum())
                if sz == 0 or (mk & m).sum() < 0.85 * sz or _solidity(mk) < p.merge_piece_solidity:
                    continue
                if (mk & taken).sum() > 0.3 * sz:
                    continue
                piece = mk & m & ~taken
                if piece.sum() < p.merge_piece_min * ref:
                    continue
                kept.append((piece, logits[i, k], iou))
                taken |= piece
        if len(kept) < min_pieces:
            return None
        cover = float(taken.sum()) / max(1, int(m.sum()))
        if min_pieces >= 2 and cover < p.merge_min_cover:
            return None
        return kept, cover

    @staticmethod
    def _touching(a, b, r=2, min_frac=0.05):
        """True if b lies within r px of a along more than min_frac of the smaller perimeter."""
        ring_a = _dilate(a, r) & ~a
        contact = int((ring_a & b).sum()) / max(1, r)
        per = min(int((_dilate(a, 1) & ~a).sum()), int((_dilate(b, 1) & ~b).sum()))
        return per > 0 and contact >= min_frac * per

    @staticmethod
    def _edge_hugging(m, min_side_cover=0.30):
        """True if the mask covers >= min_side_cover of the length of at least one image side."""
        return max(m[0].mean(), m[-1].mean(), m[:, 0].mean(), m[:, -1].mean()) >= min_side_cover

    @staticmethod
    def _n_holes(m, r, min_frac=0.01):
        filled = ndi.binary_fill_holes(_close(m, r))
        holes = filled & ~_close(m, r)
        if not holes.any():
            return 0
        lab_h, nh = ndi.label(holes)
        sizes = ndi.sum(holes, lab_h, range(1, nh + 1))
        return int((np.asarray(sizes) >= min_frac * filled.sum()).sum())

    def _support_like(self, d, prim, wh, ww):
        p = self.p
        m = d["m"]
        ys, xs = np.where(m)
        if len(ys) == 0:
            return False
        band = max(2, round(0.015 * min(wh, ww)))
        touched = sum([m[:band].any(), m[-band:].any(), m[:, :band].any(), m[:, -band:].any()])
        span = max((xs.max() - xs.min() + 1) / ww, (ys.max() - ys.min() + 1) / wh)
        return (touched >= 2 and span >= p.support_span and m.sum() >= prim["m"].sum()
                and d["score"] <= p.support_score_ratio * prim["score"])

    def _hole_evidence(self, cands, c, bg_like, primary, n):
        """Masks that may cover a hole of `c`: other detections' primaries + c's own alternatives."""
        p = self.p
        ev = []
        for pid in range(n):
            if pid == c.pid:
                continue
            o = primary(pid)
            if o.iou >= p.min_cand_iou and not bg_like(o) and o.area < p.scene_cand_area:
                ev.append(o.m)
        filled_area = float(ndi.binary_fill_holes(c.m).mean())
        for alt in cands[c.pid]:
            if alt.k != c.k and alt.iou >= 0.40 and not bg_like(alt) and alt.area <= 1.3 * filled_area:
                ev.append(alt.m)
        return ev

    def _bg_model(self, lab, sel, ring):
        if sel.sum() < max(50, 0.15 * ring.sum()):
            return None
        px = lab[sel].reshape(-1, 3).astype(np.float32)
        if len(px) > 6000:
            px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
        k = min(4, len(px))
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                        2, cv2.KMEANS_PP_CENTERS)
        # ignore minor clusters (object shadows, rim slivers) so they do not mark glaze as background
        counts = np.bincount(labels.flatten(), minlength=k)
        keep = counts >= self.p.bg_cluster_support * len(px)
        return centers[keep] if keep.any() else centers

    def _bg_like_fraction(self, lab, region, centers):
        if centers is None or not region.any():
            return None
        px = lab[region].reshape(-1, 3)
        if len(px) > 8000:
            px = px[np.random.default_rng(1).choice(len(px), 8000, replace=False)]
        d = np.sqrt(((px[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(1)
        return float((d < self.p.hole_bg_delta_e).mean())

    def _fill_holes(self, m, others, bg_centers, lab, decisions, tag, pad_border=False,
                    objects_are_separate=False):
        """Fill enclosed holes that are ceramic (glazed interiors); keep genuine see-through openings.

        Small gaps in a ring (a plate rim interrupted by a highlight) are bridged by a closing whose
        radius is 2 % of the object size before holes are searched. With pad_border (frame-filling
        objects), image sides that the mask covers for >= 60 % are treated as closed, so speckle gaps
        that open onto the frame edge are evaluated like holes instead of being left as bays.
        """
        p = self.p
        if not m.any():
            return m
        ys, xs = np.where(m)
        size = max(int(ys.max() - ys.min()), int(xs.max() - xs.min())) + 1
        r = max(2, int(0.02 * size))
        if pad_border:
            P = r + 2
            mp = np.pad(m, P, constant_values=False)
            sides = {"top": m[0].mean(), "bottom": m[-1].mean(), "left": m[:, 0].mean(), "right": m[:, -1].mean()}
            if sides["top"] >= 0.6:
                mp[:P] = True
            if sides["bottom"] >= 0.6:
                mp[-P:] = True
            if sides["left"] >= 0.6:
                mp[:, :P] = True
            if sides["right"] >= 0.6:
                mp[:, -P:] = True
            closed = (cv2.morphologyEx(mp.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(r)).astype(bool) | mp)
            filled = ndi.binary_fill_holes(closed)[P:-P, P:-P]
            closed = closed[P:-P, P:-P]
        else:
            closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(r)).astype(bool) | m
            filled = ndi.binary_fill_holes(closed)
        holes = filled & ~closed
        if not holes.any():
            return m
        lab_h, nh = ndi.label(holes)
        sizes = ndi.sum(holes, lab_h, range(1, nh + 1))
        total = float(filled.sum())
        out = m.copy()
        bridges = closed & ~m
        for k, sz in enumerate(sizes, 1):
            h = lab_h == k
            if sz < p.pinhole_frac * total:
                out |= h
                continue
            cover = max((float((h & o).sum()) / sz for o in others), default=0.0)
            bgf = self._bg_like_fraction(lab, h, bg_centers)
            if objects_are_separate and cover >= p.hole_cover_min:
                decisions.append(f"{tag}: hole {sz / total:.1%} holds another detected object; kept separate")
                continue
            fill = ((bgf is not None and bgf <= p.hole_bg_like_max)
                    or (cover >= p.hole_cover_min and (bgf is None or bgf <= 0.75))
                    or (bgf is None and sz < 0.25 * total))
            if fill:
                out |= h | (bridges & _dilate(h, r + 1))
                decisions.append(f"{tag}: filled ceramic hole {sz / total:.1%} (covered={cover:.2f}, bg_like={bgf})")
            else:
                decisions.append(f"{tag}: kept opening {sz / total:.1%} as background (covered={cover:.2f}, bg_like={bgf})")
        if pad_border:
            # narrow speckle gaps closed by the morphology itself (not enclosed holes)
            gaps = closed & ~out
            lab_g, ng = ndi.label(gaps)
            if ng:
                gs = ndi.sum(gaps, lab_g, range(1, ng + 1))
                small = [k for k, g in enumerate(gs, 1) if g < p.pinhole_frac * total]
                if small:
                    out |= np.isin(lab_g, small)
        return out

    def _subtract_hands(self, insts, hand_masks, decisions, flags):
        p = self.p
        if not hand_masks:
            return
        real = []
        for h in hand_masks:
            iou = max((float((h & d["m"]).sum()) / max(1, int((h | d["m"]).sum())) for d in insts), default=0.0)
            if iou >= p.hand_object_iou:
                flags.append("hand_is_object_ignored")
            else:
                real.append(h)
        if not real:
            return
        hand = np.any(real, axis=0)
        for d in insts:
            a = int(d["m"].sum())
            ov = int((d["m"] & hand).sum()) / max(1, a)
            if 0 < ov <= p.max_hand_removal:
                d["m"] = d["m"] & ~hand
                if ov > 0.001:
                    d["flags"].append("hand_removed")
                    flags.append("hand_removed")
                    decisions.append(f"verified hand pixels removed from instance ({ov:.1%})")
            elif ov > p.max_hand_removal:
                d["flags"].append("hand_overlap_ignored")
                flags.append("hand_overlap_ignored")

    def _resolve_overlaps(self, insts, logits, work_hw):
        """Pixels claimed by several instances go to the instance with the highest SAM logit."""
        if len(insts) < 2:
            return insts
        total = np.zeros(work_hw, np.int32)
        for d in insts:
            total += d["m"]
        if (total < 2).all():
            return insts
        stack = []
        for d in insts:
            lg = self._instance_logits(d, logits)
            with torch.no_grad():
                lg = F.interpolate(lg[None, None].to(self.device), work_hw, mode="bilinear",
                                   align_corners=False)[0, 0].cpu().numpy()
            # filled holes / recovered members have no positive logit: give them a small positive value
            lg = np.where(d["m"], np.maximum(lg, 0.01), -np.inf)
            stack.append(lg)
        stack = np.stack(stack, 0)
        owner = stack.argmax(0)
        anym = np.isfinite(stack).any(0)
        for j, d in enumerate(insts):
            d["m"] = d["m"] & (owner == j) & anym
        return insts

    def _split_and_clean(self, insts, decisions):
        p = self.p
        out = []
        for d in insts:
            m = d["m"]
            lab_c, nc = ndi.label(m)
            if nc == 0:
                continue
            sizes = ndi.sum(m, lab_c, range(1, nc + 1))
            tot = float(sizes.sum())
            keep = [k for k, sz in enumerate(sizes, 1) if sz >= p.speck_frac * tot]
            if len(keep) < nc:
                d["flags"].append("specks_removed")
            big = [k for k in keep if sizes[k - 1] >= p.split_min_frac * tot]
            if len(big) >= 2 and d["source"] != "dominant":
                decisions.append(f"instance split into {len(big)} disconnected parts")
                rest = [k for k in keep if k not in big]
                for j, k in enumerate(big):
                    part = lab_c == k
                    if j == 0 and rest:
                        part |= np.isin(lab_c, rest)
                    out.append({**d, "m": part, "source": "split",
                                "flags": d["flags"] + ["split_from_disconnected_mask"]})
            else:
                if len(big) >= 2:
                    d["flags"].append("dominant_object_disconnected_parts")
                d["m"] = np.isin(lab_c, keep)
                out.append(d)
        return out

    @staticmethod
    def _instance_logits(d, logits):
        """Max over the 256x256 SAM logits that produced an instance (box members and point pieces)."""
        lg = None
        for pid, k in d["members"]:
            v = logits[pid, k]
            lg = v if lg is None else torch.maximum(lg, v)
        for v in d.get("extra_logits", []):
            lg = v if lg is None else torch.maximum(lg, v)
        if lg is None:
            lg = torch.full((256, 256), -1.0)
        return lg

    def _full_res(self, d, logits, H, W, s):
        lg = self._instance_logits(d, logits)
        with torch.no_grad():
            up = F.interpolate(lg[None, None].to(self.device), (H, W), mode="bilinear", align_corners=False)
            cand = (up[0, 0] > 0).cpu().numpy()
        if s >= 1.0:
            guide, r = d["m"], 1
        else:
            guide = _resize_bool(d["m"], (W, H))
            r = int(np.ceil(1.0 / s)) + 1
        return (cand & _dilate(guide, r)) | _erode(guide, r)
