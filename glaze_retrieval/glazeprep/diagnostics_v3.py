"""Segmentation diagnostics for the upgraded pipeline.

Two different things, kept apart on purpose:

1. risk_assessment(): heuristic error-risk flags for the NEW (v3) segmentation, using only evidence
   available at run time (model confidences, decisions taken, mask geometry). The six categories are
   MISS_INNER_PART, MISS_OUTER_PART, NOT_RELATED, SMALL_CHANGE, MULTI_OBJECT_MERGED, BOUNDING_BOX_ALL.
   The quality score is a triage heuristic (0-100), NOT a calibrated probability of being correct.

2. legacy_change(): how the LEGACY mask (full_segmentation) differs from the new one, expressed in the
   same six categories, *assuming the new mask is closer to the truth*. Used for before/after reports
   and for comparison with the human review labels; it is not a measurement of accuracy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CATEGORIES = ["MISS_INNER_PART", "MISS_OUTER_PART", "NOT_RELATED", "SMALL_CHANGE",
              "MULTI_OBJECT_MERGED", "BOUNDING_BOX_ALL"]

PENALTY = {
    "no_target": 100, "image_not_ceramic": 100, "image_may_not_show_ceramic": 30,
    "primary_possible_non_ceramic": 30, "secondary_possible_non_ceramic": 10,
    "dominant_low_sam_confidence": 20, "weak_detection_fallback": 20, "low_detection_confidence": 10,
    "possible_merged_objects": 15, "possible_touching_objects_merged": 15, "instance_limit_reached": 10,
    "boundary_uncertain": 10, "boundary_very_uncertain": 20, "large_internal_holes": 10,
    "whole_frame_without_closeup_evidence": 25, "mask_much_smaller_than_detection": 15,
    "hand_overlap_ignored": 15, "complex_scene_refined": 5,
}


def _hole_fraction(m: np.ndarray) -> float:
    filled = ndi.binary_fill_holes(m)
    fa = float(filled.sum())
    return float((filled & ~m).sum()) / fa if fa else 0.0


def _small(m, side=512):
    """Downsample a bool mask for cheap geometry (nearest)."""
    h, w = m.shape
    s = max(1, int(np.ceil(max(h, w) / side)))
    return m[::s, ::s]


def risk_assessment(meta: dict, instance_masks: list) -> dict:
    """meta: V3Result.meta(); instance_masks: full-res bool masks in meta['instances'] order."""
    flags = set(meta.get("flags", []))
    scene = meta.get("scene_scores", {}) or {}
    insts = meta.get("instances", [])
    evidence = {c: [] for c in CATEGORIES}
    penalties = {}

    def pen(key, why_cat=None, why=None):
        penalties[key] = PENALTY[key]
        if why_cat:
            evidence[why_cat].append(why or key)

    if not insts:
        key = "image_not_ceramic" if "image_not_ceramic" in flags else "no_target"
        penalties[key] = PENALTY[key]
        return {"segmentation_error_flags": [], "evidence": evidence, "quality_score": 0,
                "quality_components": penalties, "segmentation_review_required": True,
                "review_reasons": ["no ceramic object segmented" if key == "no_target"
                                   else "image judged not to show a ceramic object"]}

    if "image_may_not_show_ceramic" in flags:
        pen("image_may_not_show_ceramic", "NOT_RELATED", "CLIP rates the whole photo as probably not ceramic")
    for k, inst in enumerate(insts):
        pc = inst.get("p_ceramic")
        if pc is not None and pc < 0.35:
            if inst.get("primary"):
                pen("primary_possible_non_ceramic", "NOT_RELATED",
                    f"main object {k + 1} looks non-ceramic to CLIP (p={pc:.2f})")
            else:
                pen("secondary_possible_non_ceramic", "NOT_RELATED",
                    f"object {k + 1} looks non-ceramic to CLIP (p={pc:.2f})")
    if "hand_overlap_ignored" in flags:
        pen("hand_overlap_ignored", "NOT_RELATED", "a verified hand overlaps the object but was not subtracted")
    if "dominant_low_sam_confidence" in flags:
        pen("dominant_low_sam_confidence", "SMALL_CHANGE",
            "frame-filling object accepted with low SAM confidence (boundary uncertain)")
    if "weak_detection_fallback" in flags:
        pen("weak_detection_fallback", "MISS_OUTER_PART", "object found only by weak detections")
    if "low_detection_confidence" in flags:
        pen("low_detection_confidence", None)
    for key in ("possible_merged_objects", "possible_touching_objects_merged"):
        if key in flags:
            pen(key, "MULTI_OBJECT_MERGED", key.replace("_", " "))
    if "instance_limit_reached" in flags:
        pen("instance_limit_reached", "MISS_OUTER_PART", "instance cap reached; smallest objects dropped")
    if flags & {"merged_mass_split", "objects_recovered_after_backdrop_removal"}:
        pen("complex_scene_refined", "MULTI_OBJECT_MERGED", "objects separated by point-prompt refinement")
    ious = [i.get("sam_iou", 1.0) for i in insts if i.get("source") not in ("point_split_residual",)]
    if ious:
        lo = min(ious)
        if lo < 0.70:
            pen("boundary_very_uncertain", "SMALL_CHANGE", f"lowest SAM IoU estimate {lo:.2f}")
        elif lo < 0.85:
            pen("boundary_uncertain", "SMALL_CHANGE", f"lowest SAM IoU estimate {lo:.2f}")
    holes = []
    for k, m in enumerate(instance_masks):
        hf = _hole_fraction(_small(m))
        if hf >= 0.05:
            holes.append((k + 1, hf))
    if holes:
        pen("large_internal_holes", "MISS_INNER_PART",
            "; ".join(f"object {k} has {hf:.0%} internal holes (real opening or missed glaze?)" for k, hf in holes))
    area = float(meta.get("area_frac", 0.0))
    if area >= 0.95 and scene.get("closeup", 0) < 0.5:
        pen("whole_frame_without_closeup_evidence", "BOUNDING_BOX_ALL",
            f"mask covers {area:.0%} of the photo but CLIP does not see a close-up (p={scene.get('closeup', 0):.2f})")
    dets = meta.get("detections", [])
    for k, inst in enumerate(insts):
        ids = [i for i in inst.get("det_ids", []) if isinstance(i, int) and 0 <= i < len(dets)]
        if not ids or inst.get("source") not in ("detection", "split"):
            continue
        bf = max(dets[i]["box_frac"] for i in ids)
        if bf < 0.8 and inst.get("area_frac", 0) < 0.35 * bf and inst.get("primary"):
            pen("mask_much_smaller_than_detection", "MISS_OUTER_PART",
                f"main object mask fills only {inst['area_frac'] / bf:.0%} of its detection box")
    score = max(0, 100 - sum(penalties.values()))
    cats = [c for c in CATEGORIES if evidence[c]]
    reasons = [r for c in CATEGORIES for r in evidence[c]]
    review = (score < 70 or "BOUNDING_BOX_ALL" in cats or "NOT_RELATED" in cats
              or "primary_possible_non_ceramic" in penalties)
    return {"segmentation_error_flags": cats, "evidence": evidence, "quality_score": int(score),
            "quality_components": penalties, "segmentation_review_required": bool(review),
            "review_reasons": reasons}


def legacy_change(old: np.ndarray | None, new: np.ndarray, new_instances: list, mode: str) -> dict:
    """Describe the legacy mask relative to the new one in the six categories (see module doc)."""
    if old is None:
        return {"legacy_available": False, "legacy_categories": [], "iou_legacy_vs_new": None}
    if old.shape != new.shape:
        return {"legacy_available": False, "legacy_categories": ["LEGACY_SIZE_MISMATCH"], "iou_legacy_vs_new": None}
    o, n = _small(old), _small(new)
    insts = [_small(m) for m in new_instances]
    tot = float(o.size)
    inter, union = float((o & n).sum()), float((o | n).sum())
    iou = inter / union if union else 1.0
    out = {"legacy_available": True, "iou_legacy_vs_new": round(iou, 4),
           "legacy_area_frac": round(float(o.mean()), 4), "new_area_frac": round(float(n.mean()), 4)}
    cats, notes = [], []
    if not n.any():
        out.update({"legacy_categories": ["NEW_HAS_NO_TARGET"], "legacy_notes": ["new pipeline exported no object"]})
        return out
    if iou >= 0.97:
        out.update({"legacy_categories": ["UNCHANGED"], "legacy_notes": []})
        return out
    gained, lost = n & ~o, o & ~n
    g, l_ = float(gained.sum()), float(lost.sum())
    o_filled = ndi.binary_fill_holes(o)
    if o.mean() >= 0.85 and n.mean() <= 0.70 and l_ >= 0.30 * tot:
        cats.append("BOUNDING_BOX_ALL")
        notes.append(f"legacy covered {o.mean():.0%} of the photo; new keeps {n.mean():.0%}")
    if l_ >= 0.15 * max(1.0, float(o.sum())) and "BOUNDING_BOX_ALL" not in cats:
        cats.append("NOT_RELATED")
        notes.append(f"{l_ / max(1.0, float(o.sum())):.0%} of the legacy mask is not kept by the new mask")
    if g >= 0.03 * float(n.sum()):
        inside = float((gained & o_filled).sum()) / g
        if inside >= 0.5:
            cats.append("MISS_INNER_PART")
            notes.append(f"new mask adds {g / tot:.1%} of the photo inside legacy holes")
        if g * (1 - inside) >= 0.20 * float(n.sum()):
            cats.append("MISS_OUTER_PART")
            frame = " (frame-filling object)" if mode == "DOMINANT_OBJECT" and n.mean() >= 0.75 else ""
            notes.append(f"new mask adds {g * (1 - inside) / tot:.1%} of the photo outside the legacy outline{frame}")
    lab, nl = ndi.label(o)
    if nl and len(insts) >= 2:
        for k in range(1, nl + 1):
            comp = lab == k
            cs = float(comp.sum())
            if cs < 0.01 * tot:
                continue
            members = [m for m in insts if m.sum() and (m & comp).sum() >= 0.5 * m.sum()]
            if len(members) >= 2 and sum(float((m & comp).sum()) for m in members) >= 0.5 * cs:
                cats.append("MULTI_OBJECT_MERGED")
                notes.append(f"one legacy region contains {len(members)} separate new objects")
                break
    if not cats:
        cats.append("SMALL_CHANGE" if iou >= 0.85 else "OTHER_CHANGE")
    out.update({"legacy_categories": cats, "legacy_notes": notes})
    return out
