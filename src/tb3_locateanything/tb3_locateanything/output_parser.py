"""Strict parser reused from Ling Shen's research LocateAnything evaluation.

Source commit: 694fff69d87852d592be99f39f0ca77c9b04b610.
"""

from __future__ import annotations

import math
import re
from typing import Any


BOX_TOKEN_RE = re.compile(r"<box>\s*(?P<body>.*?)\s*</box>", re.IGNORECASE | re.DOTALL)
BOX_BODY_RE = re.compile(
    r"^\s*<\s*(?P<x1>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*>\s*"
    r"<\s*(?P<y1>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*>\s*"
    r"<\s*(?P<x2>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*>\s*"
    r"<\s*(?P<y2>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*>\s*$",
    re.DOTALL,
)


def _failure(raw_text: str, code: str, *, invalid_box_count: int = 0) -> dict[str, Any]:
    return {
        "raw_text": raw_text,
        "parse_success": False,
        "parse_status": "parse_failure",
        "normalized_boxes": [],
        "pixel_boxes": [],
        "invalid_box_count": invalid_box_count,
        "duplicate_box_count": 0,
        "warnings": [],
        "errors": [code],
    }


def parse_locateanything_output(raw_text: str, image_width: int, image_height: int) -> dict[str, Any]:
    """Parse all box tokens without clipping, filtering, or deduplication.

    LocateAnything coordinates are normalized to the inclusive range 0..1000
    and are scaled independently by the source image width and height.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    tokens = list(BOX_TOKEN_RE.finditer(raw_text))
    if not tokens:
        return _failure(raw_text, "no_box_token")
    opening_tokens = len(re.findall(r"<box\b", raw_text, flags=re.IGNORECASE))
    closing_tokens = len(re.findall(r"</box\s*>", raw_text, flags=re.IGNORECASE))
    if opening_tokens != len(tokens) or closing_tokens != len(tokens):
        return _failure(
            raw_text,
            "unclosed_or_unpaired_box_token",
            invalid_box_count=max(opening_tokens, closing_tokens, 1) - len(tokens),
        )

    saw_none = False
    normalized: list[list[float]] = []
    invalid = 0
    errors: list[str] = []
    for token in tokens:
        body = token.group("body").strip()
        if body.lower() == "none":
            saw_none = True
            continue
        match = BOX_BODY_RE.fullmatch(body)
        if match is None:
            invalid += 1
            errors.append("malformed_box")
            continue
        values = [float(match.group(name)) for name in ("x1", "y1", "x2", "y2")]
        if not all(math.isfinite(value) for value in values):
            invalid += 1
            errors.append("non_finite_coordinate")
            continue
        if not all(0.0 <= value <= 1000.0 for value in values):
            invalid += 1
            errors.append("out_of_range_coordinate")
            continue
        x1, y1, x2, y2 = values
        if x1 >= x2 or y1 >= y2:
            invalid += 1
            errors.append("degenerate_or_reversed_box")
            continue
        normalized.append(values)

    if saw_none and normalized:
        return _failure(raw_text, "contradictory_none_and_boxes", invalid_box_count=invalid)
    if saw_none and (invalid or len(tokens) != 1):
        return _failure(raw_text, "malformed_none_output", invalid_box_count=invalid)
    if invalid:
        result = _failure(raw_text, sorted(set(errors))[0], invalid_box_count=invalid)
        result["errors"] = errors
        return result
    if saw_none:
        return {
            "raw_text": raw_text,
            "parse_success": True,
            "parse_status": "explicit_none",
            "normalized_boxes": [],
            "pixel_boxes": [],
            "invalid_box_count": 0,
            "duplicate_box_count": 0,
            "warnings": [],
            "errors": [],
        }
    if not normalized:
        return _failure(raw_text, "no_valid_box")

    pixel = [
        [
            box[0] * image_width / 1000.0,
            box[1] * image_height / 1000.0,
            box[2] * image_width / 1000.0,
            box[3] * image_height / 1000.0,
        ]
        for box in normalized
    ]
    seen: set[tuple[float, float, float, float]] = set()
    duplicate_count = 0
    for box in normalized:
        key = tuple(box)
        if key in seen:
            duplicate_count += 1
        seen.add(key)
    warnings = ["duplicate_boxes_preserved"] if duplicate_count else []
    return {
        "raw_text": raw_text,
        "parse_success": True,
        "parse_status": "valid_boxes",
        "normalized_boxes": normalized,
        "pixel_boxes": pixel,
        "invalid_box_count": 0,
        "duplicate_box_count": duplicate_count,
        "warnings": warnings,
        "errors": [],
    }
