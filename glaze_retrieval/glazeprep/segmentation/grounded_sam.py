"""Stage 1 - zero-shot object segmentation: Grounding DINO (text -> boxes) + SAM 2.1 (boxes -> masks).

Target definition (from the project brief):
    "all visually relevant ceramic/glaze samples belonging to the main subject"
so we do NOT simply keep the largest object. Selection rules are listed in
`_select_boxes` and every decision is written to the per-image JSON so it can
be audited in notebooks/01_segmentation_review.ipynb.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          Sam2Model, Sam2Processor)

from .. import config as C

# Grounding DINO expects lowercase phrases separated by " . "
OBJECT_PROMPT = (
    "ceramic object. pottery piece. ceramic test tile. "
    "ceramic test sample. ceramic fragment. pottery shard. "
    "broken ceramic piece. ceramic plate. ceramic bowl. "
    "glazed pottery."
)
EXCLUDE_PROMPT = "human hand. finger."


@dataclass
class SegParams:
    box_threshold: float = 0.24        # Grounding DINO box confidence
    text_threshold: float = 0.25       # Grounding DINO phrase confidence
    rel_score: float = 0.35            # keep boxes scoring >= rel_score * best score (drops weak background clutter)
    nms_iou: float = 0.60              # class-agnostic NMS between overlapping phrases
    min_box_frac: float = 0.005        # ignore boxes smaller than 0.5 % of the image
    container_frac: float = 0.80       # a box covering >80 % of the image that contains other boxes is "the table/scene"
    max_objects: int = 20
    exclude_threshold: float = 0.35    # Grounding DINO threshold for hands / fingers
    max_hand_removal: float = 0.50     # if removing hands deletes >50 % of the object, trust the object instead
    hand_object_iou: float = 0.30      # a "hand" mask with IoU > 0.3 to an object mask is that object (false hand)
    closeup_score: float = 0.45        # tiny mask + weak score -> try the frame-filling close-up fallback
    min_component_frac: float = 0.01   # drop specks < 1 % of the mask
    max_hole_frac: float = 0.01   
    preserve_parent_boxes: bool = True
    protect_ambiguous_hand_masks: bool = True
    # Background contamination protection
    max_candidate_mask_frac: float = 0.92
    max_scene_box_frac: float = 0.95
    enable_scene_filter: bool = True     # fill pin-holes < 1 % of the mask (keeps mug-handle openings)


@dataclass
class SegResult:
    mask: np.ndarray                                   # HxW bool
    boxes: list = field(default_factory=list)          # kept object boxes [x0,y0,x1,y1]
    scores: list = field(default_factory=list)         # Grounding DINO scores of kept boxes
    phrases: list = field(default_factory=list)
    sam_iou: list = field(default_factory=list)        # SAM's own predicted IoU per kept box
    n_raw_boxes: int = 0
    hand_boxes: int = 0
    hand_removed_frac: float = 0.0
    area_frac: float = 0.0
    flags: list = field(default_factory=list)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("mask")
        return d


def _box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _iou(a, b):
    i = _inter(a, b)
    return i / (_box_area(a) + _box_area(b) - i + 1e-6)


def _mask_iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


class GroundedSAM:
    def __init__(self, params: SegParams | None = None, device: str = C.DEVICE):
        self.p = params or SegParams()
        self.device = device
        self.dino_proc = AutoProcessor.from_pretrained(C.GROUNDING_DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(C.GROUNDING_DINO_ID).to(device).eval()
        self.sam_proc = Sam2Processor.from_pretrained(C.SAM2_ID)
        self.sam = Sam2Model.from_pretrained(C.SAM2_ID).to(device).eval()

    # ------------------------------------------------------------------ detection
    @torch.no_grad()
    def detect(self, img: Image.Image, prompt: str, box_thr: float):
        inputs = self.dino_proc(images=img, text=prompt, return_tensors="pt").to(self.device)
        out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=box_thr, text_threshold=self.p.text_threshold,
            target_sizes=[img.size[::-1]])[0]
        boxes = res["boxes"].float().cpu().numpy().tolist()
        scores = res["scores"].float().cpu().numpy().tolist()
        phrases = list(res.get("text_labels", res.get("labels", [""] * len(boxes))))
        return boxes, scores, phrases

    def _select_boxes(self, boxes, scores, phrases, W, H):
        img_area = float(W * H)
        idx = [i for i in range(len(boxes)) if _box_area(boxes[i]) >= self.p.min_box_frac * img_area]
        if not idx:
            return []
        # 1) relative score: keep everything close to the best detection (multi-object images keep all bowls)
        best = max(scores[i] for i in idx)
        idx = [i for i in idx if scores[i] >= self.p.rel_score * best]
        # 2) class-agnostic NMS (the same pot is often matched by "bowl" and "pottery")
        idx.sort(key=lambda i: -scores[i])
        kept = []
        for i in idx:
            if all(_iou(boxes[i], boxes[j]) < self.p.nms_iou for j in kept):
                kept.append(i)
        # 3) container suppression: a box that wraps >=2 kept objects is a "group" box, and a
        #    near-full-frame box wrapping any object is "the scene" - the children describe it better
        # Reject near-full-image detection boxes when smaller
# ceramic-object candidates are also available.
        if self.p.enable_scene_filter and len(kept) > 1:
            smaller = [
                i for i in kept
                if _box_area(boxes[i]) / img_area
                < self.p.max_scene_box_frac
            ]

            if smaller:
                kept = smaller
        if len(kept) > 1 and not self.p.preserve_parent_boxes:
            def contains(a, b):
                return _inter(boxes[a], boxes[b]) >= 0.9 * _box_area(boxes[b])

            def is_container(a):
                n_children = sum(contains(a, b) for b in kept if b != a)
                return n_children >= 2 or (n_children >= 1 and _box_area(boxes[a]) > self.p.container_frac * img_area)
            kept = [a for a in kept if not is_container(a)]
        return kept[: self.p.max_objects]

    # ------------------------------------------------------------------ SAM
    @torch.no_grad()
    def sam_masks(self, img: Image.Image, boxes):
        inputs = self.sam_proc(images=img, input_boxes=[boxes], return_tensors="pt").to(self.device)
        out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
        masks = masks[:, 0].numpy().astype(bool)                         # (n, H, W)
        iou = out.iou_scores[0, :, 0].float().cpu().numpy().tolist()
        return masks, iou

    # ------------------------------------------------------------------ clean-up
    def _clean(self, m: np.ndarray) -> np.ndarray:
        if not m.any():
            return m
        # bridge thin seams between adjacent object masks (e.g. two halves of one test tile)
        m = ndi.binary_closing(m, structure=np.ones((5, 5)), iterations=2, border_value=0) | m
        area = m.sum()
        # fill small holes only (a mug-handle opening is a large hole and must stay background)
        holes = ndi.binary_fill_holes(m) & ~m
        lab, n = ndi.label(holes)
        if n:
            sizes = ndi.sum(holes, lab, range(1, n + 1))
            for k, s in enumerate(sizes, 1):
                if s < self.p.max_hole_frac * area:
                    m[lab == k] = True
        # remove specks
        lab, n = ndi.label(m)
        if n > 1:
            sizes = ndi.sum(m, lab, range(1, n + 1))
            keep = {k for k, s in enumerate(sizes, 1) if s >= self.p.min_component_frac * area}
            m = np.isin(lab, list(keep))
        return ndi.binary_opening(m, structure=np.ones((3, 3)), iterations=1)

    # ------------------------------------------------------------------ main entry
    def __call__(self, rgb: np.ndarray) -> SegResult:
        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb)
        boxes, scores, phrases = self.detect(img, OBJECT_PROMPT, self.p.box_threshold)
        n_raw = len(boxes)
        flags = []
        if not boxes:   # fallback: lower threshold once before giving up
            boxes, scores, phrases = self.detect(img, OBJECT_PROMPT, self.p.box_threshold * 0.6)
            if boxes:
                flags.append("low_threshold_fallback")

        # NOTE: a "paper label. tag." exclusion prompt was tried and rejected - Grounding DINO
        # scores pale glaze tiles as "paper label" (0.43 vs 0.45 as ceramic), so it deleted tiles.
        kept = self._select_boxes(boxes, scores, phrases, W, H)
        eb, _, _ = self.detect(img, EXCLUDE_PROMPT, self.p.exclude_threshold)

        kb, ks, kp, iou = [boxes[i] for i in kept], [scores[i] for i in kept], [phrases[i] for i in kept], []
        obj, removed = np.zeros((H, W), bool), 0.0
        if kept:
            masks, iou = self.sam_masks(img, kb)

            # Calculate image coverage of every individual SAM mask.
            mask_areas = masks.reshape(len(masks), -1).mean(axis=1)

            # Find candidate masks that do not cover nearly
            # the entire photograph.
            valid_indices = [
                i for i, area in enumerate(mask_areas)
                if area <= self.p.max_candidate_mask_frac
            ]

            if self.p.enable_scene_filter and valid_indices:
                if len(valid_indices) < len(masks):
                    flags.append("scene_mask_filtered")

                # Retain only the plausible object masks.
                masks = masks[valid_indices]
                kb = [kb[i] for i in valid_indices]
                ks = [ks[i] for i in valid_indices]
                kp = [kp[i] for i in valid_indices]
                iou = [iou[i] for i in valid_indices]

            elif self.p.enable_scene_filter and len(masks) > 0:
                # Every mask is unusually large.
                # Do not assume that the image is background:
                # genuine ceramic close-ups can fill the frame.
                flags.append("full_frame_review_required")

            # Merge the remaining masks.
            obj = masks.any(axis=0)
            # hands holding the piece are subtracted pixel-wise
            if eb:
                hm, _ = self.sam_masks(img, eb)
                # Grounding DINO sometimes calls a pale tile a "hand": an exclusion mask that
                # largely coincides with an object mask IS that object, so it is ignored.
                real = [h for h in hm if max(_mask_iou(h, o) for o in masks) < self.p.hand_object_iou]
                if len(real) < len(hm):
                    flags.append("hand_is_object_ignored")
                if real:
                    hand = np.any(real, axis=0)
                    overlap = (obj & hand).sum() / max(obj.sum(), 1)
                    if 0 < overlap <= self.p.max_hand_removal:
                        obj = obj & ~hand
                        removed = float(overlap)
                        if removed > 0.001:
                            flags.append("hand_removed")
                    elif overlap > self.p.max_hand_removal:
                        flags.append("hand_overlap_ignored")
            obj = self._clean(obj)

        # close-up fallback: a frame-filling glaze surface often gets only a tiny box
        # (e.g. the hanging hole of a tile). Let SAM segment the whole frame instead.
        if obj.mean() < 0.02 and (not ks or np.mean(ks) < self.p.closeup_score):
            fm, fi = self.sam_masks(img, [[0.0, 0.0, float(W), float(H)]])
            full = self._clean(fm[0])
            if full.mean() > 0.5:
                obj, kb, ks, kp, iou = full, [[0.0, 0.0, float(W), float(H)]], [0.0], ["full_frame"], fi
                flags.append("closeup_fallback")
        if not obj.any():
            return SegResult(mask=obj, n_raw_boxes=n_raw, flags=flags + ["no_detection"])

        area = float(obj.mean())
        if len(kb) > 1:
            flags.append("multi_object")
        if np.mean(ks) < 0.40 and "closeup_fallback" not in flags:
            flags.append("low_confidence")
        if area < 0.02:
            flags.append("mask_too_small")
        if area > 0.90:
            flags.append("mask_too_large")
        if min(iou) < 0.80:
            flags.append("sam_uncertain")

        return SegResult(mask=obj, boxes=[[round(v, 1) for v in b] for b in kb],
                         scores=[round(s, 3) for s in ks], phrases=kp,
                         sam_iou=[round(v, 3) for v in iou], n_raw_boxes=n_raw, hand_boxes=len(eb),
                         hand_removed_frac=round(removed, 3), area_frac=round(area, 4), flags=flags)
