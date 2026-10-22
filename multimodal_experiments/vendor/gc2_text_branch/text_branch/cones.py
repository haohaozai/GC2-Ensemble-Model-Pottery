"""Pyrometric cone labels, ordering and ranges.

Cone labels stay strings because "06" (about 1000 C) and "6" (about 1220 C)
are different cones. All comparisons go through an explicit ordering list,
never through numeric conversion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

_PREFIX = re.compile(r"^(?:cone|c|δ|\^)\s*")  # "cone 6", "c6", "Δ6", "^6"
_HALF = re.compile(r"^(\d+)\s*(?:1/2|\.5)$")  # "5.5", "51/2" -> "5 1/2"
_RANGE_SPLIT = re.compile(r"\s*(?:-|–|—|\bto\b)\s*")


class ConeError(ValueError):
    """A cone value that cannot be used; ``code`` becomes a quality flag."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ConeRange:
    """Inclusive cone range; a single cone has ``low == high``."""

    low: str
    high: str
    low_index: int
    high_index: int

    @property
    def is_single(self) -> bool:
        return self.low_index == self.high_index

    @property
    def label(self) -> str:
        """Glazy-style label, e.g. ``"9 - 10"`` or ``"06"``."""
        return self.low if self.is_single else f"{self.low} - {self.high}"

    @property
    def text(self) -> str:
        """Description wording, e.g. ``"9 to 10"``."""
        return self.low if self.is_single else f"{self.low} to {self.high}"

    def contains(self, other: "ConeRange") -> bool:
        return self.low_index <= other.low_index and other.high_index <= self.high_index

    def overlaps(self, other: "ConeRange") -> bool:
        return self.low_index <= other.high_index and other.low_index <= self.high_index

    def to_dict(self) -> dict[str, str]:
        return {"low": self.low, "high": self.high}


class ConeScale:
    """Parses cone labels and ranges against an explicit cone ordering."""

    def __init__(self, order: Sequence[str], aliases: Mapping[str, str] | None = None):
        if len(set(order)) != len(order):
            raise ValueError("Cone order contains duplicate labels")
        self.order = tuple(order)
        self._index = {label: i for i, label in enumerate(self.order)}
        self._aliases = {" ".join(k.split()).lower(): v for k, v in (aliases or {}).items()}

    def index(self, label: str) -> int:
        return self._index[label]

    def normalise_label(self, token: str) -> str:
        text = " ".join(token.replace("½", " 1/2").split()).lower()
        text = _PREFIX.sub("", text)
        text = self._aliases.get(text, text)
        text = _HALF.sub(r"\1 1/2", text)
        text = " ".join(text.split())
        if text not in self._index:
            raise ConeError("cone_unrecognised", f"{token.strip()!r} is not in the configured cone order")
        return text

    def parse(self, raw: str) -> ConeRange:
        """Parse ``"6"``, ``"06"``, ``"9 - 11"``, ``"cone 9 to 10"`` etc.

        Raises ``ConeError`` for malformed, unknown or reversed values rather
        than reordering or guessing.
        """
        text = raw.strip()
        parts = _RANGE_SPLIT.split(text) if text else [""]
        if len(parts) > 2 or any(not part for part in parts):
            raise ConeError("cone_malformed", f"{raw!r} is not a cone or 'low - high' cone range")
        low, high = (self.normalise_label(part) for part in (parts[0], parts[-1]))
        low_index, high_index = self._index[low], self._index[high]
        if low_index > high_index:
            raise ConeError(
                "cone_range_reversed",
                f"{raw!r}: cone {low} is hotter than cone {high} in the cone order",
            )
        return ConeRange(low, high, low_index, high_index)

    def from_dict(self, data: Mapping[str, str]) -> ConeRange:
        low, high = data["low"], data["high"]
        if low not in self._index or high not in self._index:
            raise ValueError(f"Stored cone {data} is not in the configured cone order")
        return ConeRange(low, high, self._index[low], self._index[high])
