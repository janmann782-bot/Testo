from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine import build_attack_plan, iter_attack_states, iter_turn_states, load_rgb, city_mask_from_rgba
from palette import CONTROL_KEFIR, CONTROL_YOGURT
from storage import SessionPaths, ensure_session_schema, list_operations, load_meta
from visual import load_optional_visual_layers, render_visual, resize_cities, resize_terrain


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def stamp_date(frame: np.ndarray, day: date) -> np.ndarray:
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    text = day.strftime("%d.%m.%Y")
    font = _font(max(20, frame.shape[1] // 70))
    sw = max(2, frame.shape[1] // 1500)
    x = max(18, frame.shape[1] // 55)
    y = max(14, frame.shape[1] // 65)
    draw.text((x, y), text, font=font, fill=(255, 255, 255), stroke_width=sw, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def apply_gentle_curve(frame: np.ndarray, bg: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    """Apply a gentle bent/perspective export look inspired by mapping edits.

    Keeps output size unchanged, with a subtle keystone squeeze and curved
    vertical displacement. The map sits on a light background instead of
    stretching into black corners.
    """
    h, w = frame.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    xn = (xx - cx) / max(1.0, cx)
    yn = (yy - cy) / max(1.0, cy)

    # Slight horizontal pinch toward top/bottom and a vertical bow.
    scale_x = 1.0 - 0.055 * np.abs(yn)
    src_x = cx + (xx - cx) / np.maximum(scale_x, 0.82)
    src_y = yy - (10.0 * (xn ** 2) - 3.0) + 12.0 * np.maximum(0.0, yn) * (np.abs(xn) ** 1.35)

    warped = cv2.remap(frame, src_x.astype(np.float32), src_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bg)

    # Mask outer corners slightly to get the beveled/curved feel from the ref.
    mask = np.ones((h, w), dtype=np.uint8) * 255
    cut = int(round(w * 0.065))
    rise = int(round(h * 0.11))
    poly = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - rise],
        [w - cut, h - 1],
        [cut, h - 1],
        [0, h - rise],
    ], dtype=np.int32)
    mask[:] = 0
    cv2.fillConvexPoly(mask, poly, 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.2, sigmaY=1.2)
    out = np.full_like(warped, bg, dtype=np.uint8)
    a = (mask.astype(np.float32) / 255.0)[..., None]
    out = np.clip(np.rint(out * (1.0 - a) + warped.astype(np.float32) * a), 0, 255).astype(np.uint8)
    return out


def export_mp4(
    p: SessionPaths,
    output: Path,
    fps: int = 30,
    max_width: int = 1920,
    stamp_dates: bool = True,
    curved: bool = False,
) -> None:
    """Replay native simulation exactly, resize only the rendered control layer.

    Earlier builds re-simulated at export resolution for speed. That was fast,
    but geometry could differ slightly after scaling. This version keeps the
    authoritative simulation at native resolution and only downsamples each
    finished control state before rendering the MP4 frame. /map, replay, undo
    and export therefore share one history instead of parallel universes.
    """
    ensure_session_schema(p)
    base_native = load_rgb(p.base)
    homeland_native = np.load(p.homeland).astype(np.uint8)
    territory_native = np.load(p.territory).astype(bool)
    control = np.load(p.initial_control).astype(np.uint8)
    nh, nw = base_native.shape[:2]
    terrain_native, cities_native = load_optional_visual_layers(p.terrain, p.cities, (nh, nw))
    city_mask_native = city_mask_from_rgba(cities_native)

    if max_width > 0 and nw > max_width:
        scale = max_width / nw
        out_w = max(2, int(round(nw * scale / 2) * 2))
        out_h = max(2, int(round(nh * scale / 2) * 2))
    else:
        out_w = nw - (nw % 2)
        out_h = nh - (nh % 2)

    if (out_w, out_h) != (nw, nh):
        base = cv2.resize(base_native, (out_w, out_h), interpolation=cv2.INTER_AREA)
        homeland = cv2.resize(homeland_native, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        territory = _resize_mask(territory_native, (out_w, out_h))
        terrain = resize_terrain(terrain_native, (out_w, out_h))
        cities = resize_cities(cities_native, (out_w, out_h))
    else:
        base = base_native
        homeland = homeland_native
        territory = territory_native
        terrain = terrain_native
        cities = cities_native

    meta = load_meta(p)
    current_day = date.fromisoformat(meta["start_date"])

    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def push(native_state: np.ndarray, d: date) -> None:
        if (out_w, out_h) != (nw, nh):
            state = cv2.resize(native_state, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            state = native_state
        frame = render_visual(base, homeland, state, territory, terrain=terrain, cities=cities)
        if curved:
            frame = apply_gentle_curve(frame)
        if stamp_dates:
            frame = stamp_date(frame, d)
        assert proc.stdin is not None
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    try:
        push(control, current_day)
        for opdir in list_operations(p):
            spec = json.loads((opdir / "spec.json").read_text(encoding="utf-8"))
            days = int(spec["days"])
            kind = spec["kind"]

            if kind == "wait":
                for _ in range(days):
                    current_day += timedelta(days=1)
                    push(control, current_day)
                continue

            if kind == "attack":
                attacker = CONTROL_KEFIR if spec["side"] == "kefir" else CONTROL_YOGURT
                order_name = spec.get("order_name", "order")
                plan = build_attack_plan(
                    control,
                    territory_native,
                    _mask(opdir / f"{order_name}.png"),
                    attacker,
                    days,
                    width_px=int(spec.get("width_px", 110)),
                    gray_width_px=float(spec.get("gray_width_px", 9.0)),
                    front_noise_px=float(spec.get("front_noise_px", 11.0)),
                    seed=int(spec["seed"]),
                )
                for state in iter_attack_states(control, plan, homeland=homeland_native, territory=territory_native, city_mask=city_mask_native):
                    control = state
                    current_day += timedelta(days=1)
                    push(control, current_day)
                continue

            if kind == "turn":
                kp = yp = None
                if spec.get("kefir"):
                    s = spec["kefir"]
                    kp = build_attack_plan(
                        control,
                        territory_native,
                        _mask(opdir / "kefir.png"),
                        CONTROL_KEFIR,
                        days,
                        width_px=int(s.get("width_px", 110)),
                        gray_width_px=float(s.get("gray_width_px", 9.0)),
                        front_noise_px=float(s.get("front_noise_px", 11.0)),
                        seed=int(s["seed"]),
                    )
                if spec.get("yogurt"):
                    s = spec["yogurt"]
                    yp = build_attack_plan(
                        control,
                        territory_native,
                        _mask(opdir / "yogurt.png"),
                        CONTROL_YOGURT,
                        days,
                        width_px=int(s.get("width_px", 110)),
                        gray_width_px=float(s.get("gray_width_px", 9.0)),
                        front_noise_px=float(s.get("front_noise_px", 11.0)),
                        seed=int(s["seed"]),
                    )
                for state in iter_turn_states(control, kp, yp, days, homeland=homeland_native, territory=territory_native, city_mask=city_mask_native):
                    control = state
                    current_day += timedelta(days=1)
                    push(control, current_day)
                continue

            raise ValueError(f"unknown operation kind: {kind}")
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg закрыл поток раньше времени") from exc
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {stderr[-4000:]}")
