from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from engine import render
from palette import CONTROL_CONTESTED

DEFAULT_TERRAIN_STRENGTH = 0.20


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def load_optional_visual_layers(
    terrain_path: str | Path | None,
    cities_path: str | Path | None,
    expected_hw: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    h, w = expected_hw
    terrain = None
    cities = None

    if terrain_path is not None and Path(terrain_path).exists():
        with Image.open(terrain_path) as im:
            if im.size != (w, h):
                raise ValueError(f"terrain layer must be {w}×{h}, got {im.width}×{im.height}")
            terrain = np.asarray(im.convert("RGB"), dtype=np.uint8)

    if cities_path is not None and Path(cities_path).exists():
        with Image.open(cities_path) as im:
            if im.size != (w, h):
                raise ValueError(f"cities layer must be {w}×{h}, got {im.width}×{im.height}")
            cities = np.asarray(im.convert("RGBA"), dtype=np.uint8)

    return terrain, cities


def _alpha_composite_rgb(base: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if overlay_rgba.shape[:2] != base.shape[:2] or overlay_rgba.ndim != 3 or overlay_rgba.shape[2] != 4:
        raise ValueError("RGBA overlay size does not match rendered map")
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = overlay_rgba[:, :, :3].astype(np.float32)
    out = base.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def render_visual(
    base: np.ndarray,
    homeland: np.ndarray,
    control: np.ndarray,
    territory: np.ndarray,
    *,
    terrain: np.ndarray | None = None,
    cities: np.ndarray | None = None,
    terrain_strength: float = DEFAULT_TERRAIN_STRENGTH,
) -> np.ndarray:
    """Render political control with optional physical relief and city overlay.

    Layer order:
      1) original base (neutral background, rivers, borders)
      2) control colors
      3) physical relief mixed into controlled territory only
      4) city points / capital overlay

    Rivers and other non-territory linework from the base survive because the
    immutable territory mask never includes those pixels. Contested frontage is
    intentionally kept flat so the ~9 px gray strip remains easy to read.
    """
    out = render(base, homeland, control, territory)

    if terrain is not None:
        if terrain.shape != out.shape:
            raise ValueError("terrain layer size does not match map")
        strength = float(np.clip(terrain_strength, 0.0, 0.65))
        shade_mask = territory & (control != CONTROL_CONTESTED)
        if shade_mask.any() and strength > 0.0:
            a = out[shade_mask].astype(np.float32)
            b = terrain[shade_mask].astype(np.float32)
            out[shade_mask] = np.clip(np.rint(a * (1.0 - strength) + b * strength), 0, 255).astype(np.uint8)

    if cities is not None:
        out = _alpha_composite_rgb(out, cities)

    return out


def resize_terrain(terrain: np.ndarray | None, size_wh: tuple[int, int]) -> np.ndarray | None:
    if terrain is None:
        return None
    w, h = size_wh
    return cv2.resize(terrain, (w, h), interpolation=cv2.INTER_AREA)


def resize_cities(cities: np.ndarray | None, size_wh: tuple[int, int]) -> np.ndarray | None:
    if cities is None:
        return None
    w, h = size_wh
    rgba = Image.fromarray(cities, mode="RGBA").resize((w, h), resample=Image.Resampling.LANCZOS)
    return np.asarray(rgba, dtype=np.uint8)
