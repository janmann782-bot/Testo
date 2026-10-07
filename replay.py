from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from engine import apply_attack, apply_turn, build_attack_plan, city_mask_from_rgba
from palette import CONTROL_KEFIR, CONTROL_YOGURT
from storage import SessionPaths, ensure_session_schema, list_operations
from visual import load_optional_visual_layers


def _mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127


def replay_control(p: SessionPaths) -> np.ndarray:
    ensure_session_schema(p)
    territory = np.load(p.territory).astype(bool)
    homeland = np.load(p.homeland).astype(np.uint8)
    control = np.load(p.initial_control).astype(np.uint8)
    h, w = territory.shape
    _, cities = load_optional_visual_layers(p.terrain, p.cities, (h, w))
    city_mask = city_mask_from_rgba(cities)

    for opdir in list_operations(p):
        spec = json.loads((opdir / "spec.json").read_text(encoding="utf-8"))
        kind = spec["kind"]
        if kind == "wait":
            continue

        if kind == "attack":
            side = spec["side"]
            attacker = CONTROL_KEFIR if side == "kefir" else CONTROL_YOGURT
            order_name = spec.get("order_name", "order")
            plan = build_attack_plan(
                control,
                territory,
                _mask(opdir / f"{order_name}.png"),
                attacker_id=attacker,
                days=int(spec["days"]),
                width_px=int(spec.get("width_px", 110)),
                gray_width_px=float(spec.get("gray_width_px", 9.0)),
                front_noise_px=float(spec.get("front_noise_px", 10.0)),
                seed=int(spec["seed"]),
            )
            control = apply_attack(control, plan, homeland=homeland, territory=territory, city_mask=city_mask)
            continue

        if kind == "turn":
            days = int(spec["days"])
            kp = yp = None
            if spec.get("kefir"):
                s = spec["kefir"]
                kp = build_attack_plan(
                    control,
                    territory,
                    _mask(opdir / "kefir.png"),
                    CONTROL_KEFIR,
                    days,
                    width_px=int(s.get("width_px", 110)),
                    gray_width_px=float(s.get("gray_width_px", 9.0)),
                    front_noise_px=float(s.get("front_noise_px", 10.0)),
                    seed=int(s["seed"]),
                )
            if spec.get("yogurt"):
                s = spec["yogurt"]
                yp = build_attack_plan(
                    control,
                    territory,
                    _mask(opdir / "yogurt.png"),
                    CONTROL_YOGURT,
                    days,
                    width_px=int(s.get("width_px", 110)),
                    gray_width_px=float(s.get("gray_width_px", 9.0)),
                    front_noise_px=float(s.get("front_noise_px", 10.0)),
                    seed=int(s["seed"]),
                )
            control = apply_turn(control, kp, yp, days, homeland=homeland, territory=territory, city_mask=city_mask)
            continue

        raise ValueError(f"unknown operation kind: {kind}")

    return control


# Old name kept so existing imports/deploy scripts do not explode during update.
replay_owner = replay_control
