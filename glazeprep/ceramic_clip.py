"""Shared OpenCLIP wrapper for the upgraded (v3) ceramic pipeline.

One model instance is shared by the segmentation verifier (is this region ceramic?) and the
attribute tagger. Cosine similarities are converted to *relative* scores with a softmax over a
closed prompt set. These relative scores are NOT calibrated probabilities: they only rank the
candidate descriptions that were offered to the model.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

# Region verification prompt sets (validated on the dev split, see docs/CLAUDE_PIPELINE.md).
CERAMIC_PROMPTS = [
    "a photo of a glazed ceramic surface",
    "a close-up photo of a ceramic glaze",
    "a ceramic glaze test tile",
    "a piece of pottery",
    "a glazed ceramic bowl",
    "a ceramic vase",
    "a ceramic cup",
    "a ceramic plate",
    "a pottery shard",
    "an unglazed clay test tile",
    "a ceramic sculpture",
    "several ceramic test tiles",
]
NON_CERAMIC_PROMPTS = [
    "a wooden table",
    "a wooden board",
    "a sheet of paper with printed text",
    "a handwritten note on paper",
    "a piece of cardboard",
    "a human hand",
    "fabric cloth",
    "a plastic bucket",
    "a paper bag",
    "a strip of masking tape",
    "a concrete floor",
    "grass and plants",
    "a printed label",
    "a screenshot of a computer screen",
    "a metal tool",
    "a plastic container",
    "a painting on a wall",
    "a shelf",
]

# Whole-image scene groups. Dev-split check (scratch evaluation, see docs): 70/71 reviewer
# "whole-frame" close-ups score closeup >= 0.5; 12/14 multi-object review images score multi >= 0.9.
# 'closeup' is high-recall but NOT specific (12/35 single-object photos also score >= 0.5), so it is
# only ever used together with SAM/DINO evidence.
SCENE_PROMPTS = {
    "closeup": ["a close-up photo of a glazed ceramic surface filling the whole frame",
                "a cropped close-up photo of a ceramic glaze test tile",
                "a macro photo of pottery glaze texture",
                "a close-up photo of the inside of a ceramic bowl"],
    "single": ["a photo of a ceramic bowl on a table", "a photo of a ceramic vase on a plain background",
               "a photo of a ceramic test tile on a white background", "a product photo of a ceramic cup",
               "a photo of a ceramic plate on a table", "a hand holding a ceramic test tile"],
    "multi": ["a photo of many ceramic test tiles arranged on a table",
              "a photo of several pottery pieces on a wooden table",
              "a photo of glaze test tiles on a board", "a photo of several ceramic test tiles in a row"],
    "nonceramic": ["a photo of a plastic bucket", "a photo of a bag of clay", "a photo of a printed document",
                   "a screenshot of a computer screen"],
}

# Hand verification on Grounding DINO "hand" boxes (context crop). Dev check: 19/19 false hands on
# ridged/curled ceramic scored p_hand <= 0.01; real hand boxes scored 0.78-1.00.
HAND_POSITIVE = ["a photo of a human hand", "human fingers holding an object",
                 "a person's hand with skin and fingernails"]
HAND_NEGATIVE = ["a glazed ceramic surface", "a ceramic test tile", "a piece of pottery",
                 "a ridge on a ceramic tile", "a close-up of ceramic glaze", "a wooden table",
                 "a white background"]


class CeramicClip:
    def __init__(self, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED,
                 device: str | None = None):
        import open_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name, self.pretrained = model_name, pretrained
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self._text_cache: dict[tuple, torch.Tensor] = {}
        self.cer_emb = self.text_embeddings(CERAMIC_PROMPTS)
        self.non_emb = self.text_embeddings(NON_CERAMIC_PROMPTS)

    # ------------------------------------------------------------------ encoders
    @torch.inference_mode()
    def text_embeddings(self, prompts) -> torch.Tensor:
        key = tuple(prompts)
        if key not in self._text_cache:
            tok = self.tokenizer(list(prompts)).to(self.device)
            emb = self.model.encode_text(tok).float()
            self._text_cache[key] = torch.nn.functional.normalize(emb, dim=-1)
        return self._text_cache[key]

    @torch.inference_mode()
    def image_embeddings(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            return torch.zeros((0, self.cer_emb.shape[1]), device=self.device)
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        emb = self.model.encode_image(batch).float()
        return torch.nn.functional.normalize(emb, dim=-1)

    # ------------------------------------------------------------------ region verification
    @torch.inference_mode()
    def ceramic_scores(self, images: list[Image.Image]) -> list[dict]:
        """Relative ceramic-vs-non-ceramic score per image (softmax over both prompt sets).

        Returns dicts with p_ceramic (sum of relative scores of ceramic prompts) and the best
        non-ceramic description, for audit.
        """
        emb = self.image_embeddings(images)
        if emb.shape[0] == 0:
            return []
        allt = torch.cat([self.cer_emb, self.non_emb], 0)
        logits = self.logit_scale * emb @ allt.T
        prob = logits.softmax(-1).cpu().numpy()
        n_c = self.cer_emb.shape[0]
        out = []
        for row in prob:
            j = int(np.argmax(row[n_c:]))
            out.append({"p_ceramic": round(float(row[:n_c].sum()), 4),
                        "top_non_ceramic": NON_CERAMIC_PROMPTS[j],
                        "top_non_ceramic_score": round(float(row[n_c + j]), 4)})
        return out

    @torch.inference_mode()
    def scene_scores(self, image: Image.Image) -> dict:
        """Relative scene-group scores for the whole photograph (sum to 1 over the groups)."""
        groups = list(SCENE_PROMPTS)
        texts = [t for g in groups for t in SCENE_PROMPTS[g]]
        emb = self.image_embeddings([image])
        prob = (self.logit_scale * emb @ self.text_embeddings(texts).T).softmax(-1)[0].cpu().numpy()
        out, j = {}, 0
        for g in groups:
            out[g] = round(float(prob[j:j + len(SCENE_PROMPTS[g])].sum()), 4)
            j += len(SCENE_PROMPTS[g])
        return out

    @torch.inference_mode()
    def hand_scores(self, crops: list[Image.Image]) -> list[float]:
        """Relative 'human hand' score for context crops of candidate hand boxes."""
        if not crops:
            return []
        emb = self.image_embeddings(crops)
        texts = HAND_POSITIVE + HAND_NEGATIVE
        prob = (self.logit_scale * emb @ self.text_embeddings(texts).T).softmax(-1).cpu().numpy()
        return [round(float(r[:len(HAND_POSITIVE)].sum()), 4) for r in prob]


def region_crop(rgb: np.ndarray, mask: np.ndarray, pad_frac: float = 0.04, fill=(127, 127, 127),
                min_side: int = 8) -> Image.Image | None:
    """Bounding-box crop of a region with pixels outside the region set to neutral grey."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    p = int(pad_frac * max(y1 - y0, x1 - x0))
    y0, y1, x0, x1 = max(0, y0 - p), min(h, y1 + p), max(0, x0 - p), min(w, x1 + p)
    if (y1 - y0) < min_side or (x1 - x0) < min_side:
        return None
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = fill
    return Image.fromarray(crop)
