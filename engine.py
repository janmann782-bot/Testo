from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from palette import (
    CONTESTED,
    CONTROL_CONTESTED,
    CONTROL_KEFIR,
    CONTROL_NONE,
    CONTROL_YOGURT,
    HOME_KEFIR,
    HOME_NONE,
    HOME_YOGURT,
    KEFIR_HOME,
    KEFIR_OCCUPATION,
    LEGACY_HIGHLIGHT,
    ORDER_KEFIR,
    ORDER_YOGURT,
    YOGURT_HOME,
    YOGURT_OCCUPATION,
)


def _near_color(arr: np.ndarray, color: np.ndarray, tolerance: int) -> np.ndarray:
    d = arr.astype(np.int32) - color.astype(np.int32)
    return np.sum(d * d, axis=2) <= tolerance * tolerance


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, optimize=True)


def initial_state(
    base: np.ndarray,
    tolerance: int = 8,
    *,
    reject_contested: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build immutable homeland + mutable control from a base control map.

    Native colors encode homeland and matching control. Occupation colors encode
    the opposite homeland plus the occupier, so a mid-war map with occupations
    can be imported without losing who the land actually belongs to.

    A gray/contested pixel cannot encode homeland by itself. New sessions reject
    such base maps rather than inventing history from a shade of gray.
    """
    if base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("base map must be RGB")

    k_home = _near_color(base, KEFIR_HOME, tolerance)
    y_home = _near_color(base, YOGURT_HOME, tolerance)
    k_occ = _near_color(base, KEFIR_OCCUPATION, tolerance)
    y_occ = _near_color(base, YOGURT_OCCUPATION, tolerance)
    contested = _near_color(base, CONTESTED, tolerance)
    legacy_highlight = _near_color(base, LEGACY_HIGHLIGHT, tolerance)

    if legacy_highlight.any():
        raise ValueError(
            "базовая карта содержит старый временный цвет #FFA7A4. Он не кодирует родную принадлежность надежно; "
            "сначала замени его на родной или оккупационный цвет"
        )

    if reject_contested and contested.any():
        raise ValueError(
            "базовая карта содержит серую/оспариваемую зону. По серому цвету невозможно понять, "
            "чья это родная земля. Укажи исходную принадлежность родным или оккупационным цветом"
        )

    homeland = np.full(base.shape[:2], HOME_NONE, dtype=np.uint8)
    control = np.full(base.shape[:2], CONTROL_NONE, dtype=np.uint8)

    homeland[k_home] = HOME_KEFIR
    control[k_home] = CONTROL_KEFIR

    homeland[y_home] = HOME_YOGURT
    control[y_home] = CONTROL_YOGURT

    # Kefir occupation = Yogurt homeland under Kefir control.
    homeland[k_occ] = HOME_YOGURT
    control[k_occ] = CONTROL_KEFIR

    # Yogurt occupation = Kefir homeland under Yogurt control.
    homeland[y_occ] = HOME_KEFIR
    control[y_occ] = CONTROL_YOGURT

    territory = homeland != HOME_NONE
    if int(territory.sum()) < 100:
        raise ValueError(
            "не нашел нормальной территории. Нужны цвета #4A0000 / #00134A "
            "(и при необходимости оккупационные цвета)"
        )

    return homeland, control, territory


def render(base: np.ndarray, homeland: np.ndarray, control: np.ndarray, territory: np.ndarray) -> np.ndarray:
    """Render control without ever losing native ownership."""
    if homeland.shape != control.shape or homeland.shape != territory.shape or homeland.shape != base.shape[:2]:
        raise ValueError("state arrays and base map have different sizes")

    out = base.copy()

    k_ctrl = territory & (control == CONTROL_KEFIR)
    y_ctrl = territory & (control == CONTROL_YOGURT)
    gray = territory & (control == CONTROL_CONTESTED)

    out[k_ctrl & (homeland == HOME_KEFIR)] = KEFIR_HOME
    out[k_ctrl & (homeland == HOME_YOGURT)] = KEFIR_OCCUPATION
    out[y_ctrl & (homeland == HOME_YOGURT)] = YOGURT_HOME
    out[y_ctrl & (homeland == HOME_KEFIR)] = YOGURT_OCCUPATION
    out[gray] = CONTESTED

    # Corrupt/unknown control on known homeland should fail visually safe by
    # falling back to native color rather than leaving an arbitrary old frame.
    unknown = territory & (control == CONTROL_NONE)
    out[unknown & (homeland == HOME_KEFIR)] = KEFIR_HOME
    out[unknown & (homeland == HOME_YOGURT)] = YOGURT_HOME
    return out


def _clean_color_mask(mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    clean = np.zeros_like(u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    return clean.astype(bool)


def extract_order_masks(annotated: np.ndarray) -> dict[int, np.ndarray]:
    """Extract command strokes with tolerance for brush antialiasing.

    The line can be horizontal, vertical, diagonal, curved or broken into
    several independent components. The geometry is not inferred from slope.
    """
    rgb = annotated.astype(np.int16)

    km = (
        ((rgb[:, :, 0] >= 175) & (rgb[:, :, 2] >= 175) & (rgb[:, :, 1] <= 125))
        | _near_color(annotated, ORDER_KEFIR, 80)
    )
    ym = (
        ((rgb[:, :, 1] >= 175) & (rgb[:, :, 0] <= 105) & (rgb[:, :, 2] <= 150))
        | _near_color(annotated, ORDER_YOGURT, 80)
    )

    # Ambiguous antialiased pixels should belong to neither side.
    overlap = km & ym
    km[overlap] = False
    ym[overlap] = False

    return {
        CONTROL_KEFIR: _clean_color_mask(km),
        CONTROL_YOGURT: _clean_color_mask(ym),
    }


def extract_order_mask(annotated: np.ndarray, attacker_id: int = CONTROL_KEFIR) -> np.ndarray:
    return extract_order_masks(annotated)[attacker_id]


def validate_annotation_against_reference(
    annotated: np.ndarray,
    reference: np.ndarray,
    order_masks: dict[int, np.ndarray],
    *,
    tolerance: int = 22,
    max_changed_fraction: float = 0.012,
) -> tuple[bool, str | None]:
    """Catch stale/wrong frames while ignoring the painted order strokes."""
    if annotated.shape != reference.shape:
        return False, f"размер карты изменился: нужен {reference.shape[1]}×{reference.shape[0]} px"

    order = order_masks[CONTROL_KEFIR] | order_masks[CONTROL_YOGURT]
    if order.any():
        # Ignore antialiasing around the command brush too.
        order = cv2.dilate(order.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)

    delta = np.max(np.abs(annotated.astype(np.int16) - reference.astype(np.int16)), axis=2)
    unexpected = (delta > tolerance) & ~order
    changed = int(unexpected.sum())
    total = int(unexpected.size)
    fraction = changed / max(1, total)

    # Tiny decorations/metadata damage are tolerated. A stale frame usually
    # changes tens or hundreds of thousands of pixels.
    if changed > max(200, int(total * 0.002)) and fraction > max_changed_fraction:
        return False, (
            f"картинка не похожа на последний /map: вне линий приказа изменено {fraction:.1%} пикселей. "
            "Возьми свежую карту и рисуй поверх нее; JPEG/скриншоты тоже лучше не использовать"
        )
    return True, None


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out: list[np.ndarray] = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            out.append(labels == i)
    return out


def _geodesic_progress_on_line(line: np.ndarray, attacker: np.ndarray) -> np.ndarray:
    """8-neighbor BFS distance along any-shaped command stroke, rooted at own control."""
    h, w = line.shape
    near_attacker = cv2.dilate(attacker.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    seeds = np.argwhere(line & near_attacker)
    if len(seeds) == 0:
        return np.full((h, w), -1, dtype=np.int32)

    dist = np.full((h, w), -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    for y, x in seeds:
        dist[y, x] = 0
        q.append((int(y), int(x)))

    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        y, x = q.popleft()
        nd = dist[y, x] + 1
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and line[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = nd
                q.append((yy, xx))
    return dist


def _smooth_noise(shape: tuple[int, int], seed: int, scale: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape
    gh = max(3, h // max(2, scale) + 2)
    gw = max(3, w // max(2, scale) + 2)
    small = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(1.5, scale / 5.0), sigmaY=max(1.5, scale / 5.0))
    std = float(noise.std()) or 1.0
    return np.clip(noise / (std * 2.35), -1.0, 1.0)


def _daily_progress_curve(max_progress: float, days: int, seed: int) -> np.ndarray:
    """Monotonic but uneven day-by-day operational tempo."""
    rng = np.random.default_rng(seed ^ 0xA17E11A)
    weights = rng.lognormal(mean=0.0, sigma=0.62, size=days).astype(np.float32)
    if days >= 5:
        kernel = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)
        weights = np.convolve(weights, kernel, mode="same").astype(np.float32)

    quiet = rng.random(days) < 0.12
    if quiet.any():
        weights[quiet] *= rng.uniform(0.03, 0.28, size=int(quiet.sum())).astype(np.float32)

    if days >= 12:
        for _ in range(max(1, days // 65)):
            a = int(rng.integers(0, max(1, days - 3)))
            max_len = min(9, days - a)
            if max_len >= 2:
                b = min(days, a + int(rng.integers(2, max_len + 1)))
                weights[a:b] *= float(rng.uniform(0.18, 0.58))

    total = float(weights.sum()) or 1.0
    return (np.cumsum(weights) / total * float(max_progress)).astype(np.float32)


def _keep_components_touching(mask: np.ndarray, roots: np.ndarray, dilate_px: int = 3) -> np.ndarray:
    """Keep only land corridor components connected to current friendly control."""
    if not mask.any():
        return mask
    k = max(1, int(dilate_px) * 2 + 1)
    touch = cv2.dilate(roots.astype(np.uint8), np.ones((k, k), np.uint8), iterations=1).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        comp = labels == i
        if np.any(comp & touch):
            out |= comp
    return out


def _actual_front_band(captured: np.ndarray, scope: np.ndarray, width_px: float | np.ndarray, bbox: tuple[int,int,int,int]) -> np.ndarray:
    """A spatially real 3-10 px band with local variation."""
    out = np.zeros_like(scope)
    if not captured.any() or not scope.any():
        return out

    x0, y0, x1, y1 = bbox
    cap = captured[y0:y1, x0:x1]
    sc = scope[y0:y1, x0:x1] & ~cap
    if not cap.any() or not sc.any():
        return out

    src = (~cap).astype(np.uint8)  # cap=0 => distance to captured territory
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    if isinstance(width_px, np.ndarray):
        widths = width_px[y0:y1, x0:x1].astype(np.float32)
    else:
        widths = np.full(cap.shape, float(width_px), dtype=np.float32)
    band = sc & (dist > 0) & (dist <= widths + 0.15)
    out[y0:y1, x0:x1] = band
    return out


def city_mask_from_rgba(cities_rgba: np.ndarray | None) -> np.ndarray | None:
    if cities_rgba is None:
        return None
    if cities_rgba.ndim != 3 or cities_rgba.shape[2] != 4:
        raise ValueError('cities layer must be RGBA')
    alpha = cities_rgba[:, :, 3] > 0
    if not alpha.any():
        return np.zeros(cities_rgba.shape[:2], dtype=bool)
    return cv2.dilate(alpha.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def _collapse_isolated_pockets(control: np.ndarray, homeland: np.ndarray, territory: np.ndarray, city_mask: np.ndarray | None) -> np.ndarray:
    if city_mask is None or not city_mask.any():
        return control
    out = control.copy()
    for side, opp, home in ((CONTROL_KEFIR, CONTROL_YOGURT, HOME_KEFIR), (CONTROL_YOGURT, CONTROL_KEFIR, HOME_YOGURT)):
        ctrl = territory & (out == side)
        if not ctrl.any():
            continue
        home_mask = homeland == home
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ctrl.astype(np.uint8), 8)
        for i in range(1, n):
            comp = labels == i
            if np.any(comp & home_mask):
                continue
            if np.any(comp & city_mask):
                continue
            if stats[i, cv2.CC_STAT_AREA] <= 0:
                continue
            out[comp] = opp
    return out


@dataclass(slots=True)
class AttackPlan:
    arrival: np.ndarray
    reach: np.ndarray
    gray_scope: np.ndarray
    gray_width_map: np.ndarray
    attacker_id: int
    defender_id: int
    days: int
    gray_width_px: float
    thresholds: np.ndarray
    max_progress: float
    bbox: tuple[int, int, int, int]
    warnings: list[str] = field(default_factory=list)

    def arrival_day(self) -> np.ndarray:
        out = np.full(self.reach.shape, 32767, dtype=np.int16)
        vals = self.arrival[self.reach]
        idx = np.searchsorted(self.thresholds, vals, side="left") + 1
        idx = np.clip(idx, 1, self.days).astype(np.int16)
        out[self.reach] = idx
        return out


def build_attack_plan(
    control: np.ndarray,
    territory: np.ndarray,
    order_mask: np.ndarray,
    attacker_id: int,
    days: int,
    width_px: int = 110,
    gray_width_px: float = 9.0,
    front_noise_px: float = 9.0,
    seed: int = 1,
) -> AttackPlan:
    if attacker_id not in (CONTROL_KEFIR, CONTROL_YOGURT):
        raise ValueError("attacker_id must be Kefir or Yogurt")
    if days < 1 or days > 5000:
        raise ValueError("days must be in 1..5000")
    if width_px < 4 or width_px > 1000:
        raise ValueError("width_px must be in 4..1000")
    if control.shape != territory.shape or order_mask.shape != control.shape:
        raise ValueError("state and order mask sizes differ")

    attacker = control == attacker_id
    if not attacker.any():
        raise ValueError("у атакующей стороны нет контролируемой территории")
    defender_id = CONTROL_YOGURT if attacker_id == CONTROL_KEFIR else CONTROL_KEFIR

    progress = np.full(control.shape, -1.0, dtype=np.float32)
    reach_all = np.zeros(control.shape, dtype=bool)
    hostile_all = np.zeros(control.shape, dtype=bool)
    comps = _component_masks(order_mask)
    if not comps:
        side = "Кефирстана" if attacker_id == CONTROL_KEFIR else "Йогуртстана"
        raise ValueError(f"не найдена линия приказа {side}; рисуй сплошной кистью примерно 4-8 px")

    warnings: list[str] = []
    invalid_no_root = 0
    invalid_no_enemy = 0
    h, w = control.shape

    for ci, comp in enumerate(comps):
        cy, cx = np.nonzero(comp)
        if len(cx) == 0:
            continue

        pad = int(max(32, width_px * 1.65 + gray_width_px + 24))
        x0 = max(0, int(cx.min()) - pad)
        x1 = min(w, int(cx.max()) + pad + 1)
        y0 = max(0, int(cy.min()) - pad)
        y1 = min(h, int(cy.max()) + pad + 1)

        comp_c = comp[y0:y1, x0:x1]
        attacker_c = attacker[y0:y1, x0:x1]
        terr_c = territory[y0:y1, x0:x1]
        control_c = control[y0:y1, x0:x1]

        line_prog = _geodesic_progress_on_line(comp_c, attacker_c)
        valid_line = line_prog >= 0
        if not valid_line.any():
            invalid_no_root += 1
            continue

        # If almost all of the brush touches friendly territory, the user most
        # likely painted along the front. It is valid, but worth flagging.
        near_attacker = cv2.dilate(attacker_c.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
        root_fraction = float((comp_c & near_attacker).sum()) / max(1, int(comp_c.sum()))
        front_like = root_fraction > 0.55
        if front_like:
            warnings.append("линия долго идет вдоль своей территории; коридор наступления специально заужен, чтобы не получилась жирная сосиска")

        src = (~valid_line).astype(np.uint8)
        dist_to_line, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        max_label = int(labels.max())
        if max_label <= 0:
            invalid_no_root += 1
            continue

        label_prog = np.full(max_label + 1, -1.0, dtype=np.float32)
        ys, xs = np.nonzero(valid_line)
        labs = labels[ys, xs].astype(np.int64)
        vals = line_prog[ys, xs].astype(np.float32)
        for lab, val in zip(labs, vals):
            if lab > 0 and (label_prog[lab] < 0 or val < label_prog[lab]):
                label_prog[lab] = val
        field = label_prog[np.clip(labels, 0, max_label)]

        phase1 = ((seed + ci * 9176) % 10007) / 10007.0 * 2.0 * np.pi
        phase2 = ((seed + ci * 31337) % 10009) / 10009.0 * 2.0 * np.pi
        line_max = max(1.0, float(line_prog[valid_line].max()))
        along = np.clip(field / line_max, 0.0, 1.35)
        width_base = float(width_px) * (0.70 if front_like else 0.86)
        taper = np.clip((1.16 if front_like else 1.24) - (0.48 if front_like else 0.58) * along, 0.42, 1.18)
        width_wave = 1.0 + 0.14 * np.sin(field * 0.024 + phase1) + 0.10 * np.sin(field * 0.071 + phase2)
        edge_noise = _smooth_noise(field.shape, seed ^ (0xBADC0DE + ci * 7919), max(18, int(width_px * 0.36)))
        bite_noise = _smooth_noise(field.shape, seed ^ (0xFACE777 + ci * 1237), max(11, int(width_px * 0.18)))
        edge_jitter = edge_noise * width_base * (0.26 if front_like else 0.24) - np.maximum(0.0, bite_noise) * width_base * 0.10
        local_width = np.clip(width_base * taper * width_wave + edge_jitter, width_base * 0.28, width_base * (1.02 if front_like else 1.18))

        corridor = (dist_to_line <= local_width) & terr_c & (field >= 0)
        # This is what prevents sea/neutral teleportation.
        corridor = _keep_components_touching(corridor, attacker_c, dilate_px=3)
        if not corridor.any():
            invalid_no_root += 1
            continue

        hostile = corridor & (control_c != attacker_id)
        if not hostile.any():
            invalid_no_enemy += 1
            continue

        # Start offensive progress at the first non-friendly pixel. A line may
        # begin deep in friendly rear without wasting 80 of 120 days marching
        # to its own frontier.
        first_hostile = float(np.min(field[hostile & (field >= 0)]))
        rear_allow = max(10.0, width_px * 0.36)
        forward = corridor & (field >= first_hostile - rear_allow)
        shifted = np.maximum(0.0, field - first_hostile)

        lateral = np.clip(dist_to_line / max(1.0, float(width_px)), 0.0, 1.6)
        flank_lag = min(28.0, max(4.0, width_base * (0.17 if front_like else 0.20)))
        shaped = shifted + (lateral ** 1.72) * flank_lag

        rview = reach_all[y0:y1, x0:x1]
        hview = hostile_all[y0:y1, x0:x1]
        pview = progress[y0:y1, x0:x1]

        rview |= forward
        hostile_forward = forward & (control_c != attacker_id)
        hview |= hostile_forward
        update = forward & ((pview < 0) | (shaped < pview))
        pview[update] = shaped[update]

    if not hostile_all.any():
        side = "Кефирстана" if attacker_id == CONTROL_KEFIR else "Йогуртстана"
        details = []
        if invalid_no_root:
            details.append(f"линий без связи со своей территорией: {invalid_no_root}")
        if invalid_no_enemy:
            details.append(f"линий, не выходящих в чужую/серую территорию: {invalid_no_enemy}")
        extra = (" (" + "; ".join(details) + ")") if details else ""
        raise ValueError(f"приказ {side} не создает наступления{extra}")

    if invalid_no_root:
        warnings.append(f"отброшено линий без связи со своей территорией: {invalid_no_root}")
    if invalid_no_enemy:
        warnings.append(f"отброшено линий, не выходящих к противнику: {invalid_no_enemy}")

    # Apply multi-scale resistance only inside the operational area.
    ty, tx = np.nonzero(reach_all)
    padn = int(max(20, gray_width_px + 8))
    nx0 = max(0, int(tx.min()) - padn)
    nx1 = min(w, int(tx.max()) + padn + 1)
    ny0 = max(0, int(ty.min()) - padn)
    ny1 = min(h, int(ty.max()) + padn + 1)
    shape_c = (ny1 - ny0, nx1 - nx0)

    n1 = _smooth_noise(shape_c, seed ^ 0x13579BDF, 170)
    n2 = _smooth_noise(shape_c, seed ^ 0x5F3759DF, 54)
    n3 = _smooth_noise(shape_c, seed ^ 0xC0FFEE, 19)
    n4 = _smooth_noise(shape_c, seed ^ 0xDEAD771, 8)
    resistance_c = (0.44 * n1 + 0.28 * n2 + 0.18 * n3 + 0.10 * n4).astype(np.float32)

    arrival = progress.copy()
    rc = reach_all[ny0:ny1, nx0:nx1]
    pc = progress[ny0:ny1, nx0:nx1]
    ac = arrival[ny0:ny1, nx0:nx1]
    noise_amp = max(0.0, float(front_noise_px))
    ac[rc] = pc[rc] + resistance_c[rc] * noise_amp
    ac[rc] = np.maximum(ac[rc], 0.0)

    max_progress = float(np.max(arrival[hostile_all]))
    max_progress = max(1.0, max_progress)

    gray_w = float(np.clip(gray_width_px, 3.0, 10.0))
    gray_noise = _smooth_noise(shape_c, seed ^ 0x9E3779B9, 33)
    gray_width_map = np.full(control.shape, gray_w, dtype=np.float32)
    gview = gray_width_map[ny0:ny1, nx0:nx1]
    gview[rc] = np.clip(gray_w + gray_noise[rc] * 2.8, 3.0, 10.0)
    # Gray scope includes a thin halo beyond the requested corridor so the
    # active front still exists at the final endpoint of an operation.
    halo_iter = max(1, int(np.ceil(10.0 + 2.0)))
    halo = cv2.dilate(reach_all.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=halo_iter).astype(bool)
    gray_scope = halo & territory & (control != attacker_id)

    return AttackPlan(
        arrival=arrival,
        reach=reach_all,
        gray_scope=gray_scope,
        gray_width_map=gray_width_map,
        attacker_id=attacker_id,
        defender_id=defender_id,
        days=days,
        gray_width_px=gray_w,
        thresholds=_daily_progress_curve(max_progress, days, seed),
        max_progress=max_progress,
        bbox=(nx0, ny0, nx1, ny1),
        warnings=list(dict.fromkeys(warnings)),
    )


def _state_for_day(start_control: np.ndarray, plan: AttackPlan, day: int) -> tuple[np.ndarray, np.ndarray]:
    day = max(1, min(plan.days, int(day)))
    threshold = float(plan.thresholds[day - 1])

    # Claim includes friendly pixels inside the active corridor. That matters in
    # simultaneous turns: it lets a drawn counteroffensive contest land that the
    # enemy captures earlier in the same 120-day turn.
    claimed = plan.reach & (plan.arrival >= 0) & (plan.arrival <= threshold)
    gray = _actual_front_band(claimed, plan.gray_scope, plan.gray_width_map, plan.bbox)
    return claimed, gray


def iter_attack_states(start_control: np.ndarray, plan: AttackPlan, homeland: np.ndarray | None = None, territory: np.ndarray | None = None, city_mask: np.ndarray | None = None) -> Iterator[np.ndarray]:
    for day in range(1, plan.days + 1):
        claimed, gray = _state_for_day(start_control, plan, day)
        control = start_control.copy()
        control[claimed] = plan.attacker_id
        # Stable attacker control wins over its own fringe.
        gray &= ~claimed
        control[gray] = CONTROL_CONTESTED
        if homeland is not None and territory is not None and city_mask is not None:
            control = _collapse_isolated_pockets(control, homeland, territory, city_mask)
        yield control


def apply_attack(start_control: np.ndarray, plan: AttackPlan, homeland: np.ndarray | None = None, territory: np.ndarray | None = None, city_mask: np.ndarray | None = None) -> np.ndarray:
    last = start_control.copy()
    for last in iter_attack_states(start_control, plan, homeland=homeland, territory=territory, city_mask=city_mask):
        pass
    return last


def iter_turn_states(
    start_control: np.ndarray,
    kefir_plan: AttackPlan | None,
    yogurt_plan: AttackPlan | None,
    days: int,
    clash_window_days: int = 2,
    homeland: np.ndarray | None = None,
    territory: np.ndarray | None = None,
    city_mask: np.ndarray | None = None,
) -> Iterator[np.ndarray]:
    """Simulate both sides on the same dates with no first-calculated-side bias."""
    if not kefir_plan and not yogurt_plan:
        for _ in range(days):
            yield start_control.copy()
        return

    kd = kefir_plan.arrival_day() if kefir_plan else None
    yd = yogurt_plan.arrival_day() if yogurt_plan else None

    empty = np.zeros(start_control.shape, dtype=bool)
    for day in range(1, days + 1):
        kc, kg = _state_for_day(start_control, kefir_plan, min(day, kefir_plan.days)) if kefir_plan else (empty, empty)
        yc, yg = _state_for_day(start_control, yogurt_plan, min(day, yogurt_plan.days)) if yogurt_plan else (empty, empty)

        control = start_control.copy()
        both = kc & yc
        k_win = kc & ~yc
        y_win = yc & ~kc
        clash = np.zeros(start_control.shape, dtype=bool)

        if both.any() and kd is not None and yd is not None:
            diff = kd.astype(np.int32) - yd.astype(np.int32)
            k_earlier = both & (diff <= -clash_window_days)
            y_earlier = both & (diff >= clash_window_days)
            close = both & ~(k_earlier | y_earlier)
            k_win |= k_earlier
            y_win |= y_earlier
            clash |= close

        control[k_win] = CONTROL_KEFIR
        control[y_win] = CONTROL_YOGURT

        # Stable claims beat the other side's mere 8-9 px front fringe.
        gray = (kg | yg | clash) & ~(k_win | y_win)
        control[gray] = CONTROL_CONTESTED
        if homeland is not None and territory is not None and city_mask is not None:
            control = _collapse_isolated_pockets(control, homeland, territory, city_mask)
        yield control


def apply_turn(
    start_control: np.ndarray,
    kefir_plan: AttackPlan | None,
    yogurt_plan: AttackPlan | None,
    days: int,
    homeland: np.ndarray | None = None,
    territory: np.ndarray | None = None,
    city_mask: np.ndarray | None = None,
) -> np.ndarray:
    last = start_control.copy()
    for last in iter_turn_states(start_control, kefir_plan, yogurt_plan, days, homeland=homeland, territory=territory, city_mask=city_mask):
        pass
    return last
