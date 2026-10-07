from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from engine import initial_state, load_rgb

SCHEMA_VERSION = 2


@dataclass(slots=True)
class SessionPaths:
    root: Path
    base: Path
    terrain: Path
    cities: Path
    homeland: Path
    initial_control: Path
    current_control: Path
    territory: Path
    meta: Path
    operations: Path
    # Legacy v1 paths, retained only for migration.
    legacy_initial_owner: Path
    legacy_current_owner: Path


def paths_for(data_dir: Path, chat_id: int) -> SessionPaths:
    root = data_dir / str(chat_id)
    return SessionPaths(
        root=root,
        base=root / "base.png",
        terrain=root / "terrain.png",
        cities=root / "cities.png",
        homeland=root / "homeland.npy",
        initial_control=root / "initial_control.npy",
        current_control=root / "current_control.npy",
        territory=root / "territory.npy",
        meta=root / "meta.json",
        operations=root / "operations",
        legacy_initial_owner=root / "initial_owner.npy",
        legacy_current_owner=root / "current_owner.npy",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def create_session(
    data_dir: Path,
    chat_id: int,
    base_path: Path,
    start_date: str = "2057-01-01",
    *,
    terrain_path: Path | None = None,
    cities_path: Path | None = None,
) -> SessionPaths:
    # Validate date before destroying an existing session.
    date.fromisoformat(start_date)
    with Image.open(base_path) as im:
        if im.format != "PNG":
            raise ValueError("базовую карту присылай PNG как файл; JPEG портит точные цвета")
        if im.width < 64 or im.height < 64:
            raise ValueError("карта слишком маленькая")
        if im.width > 8192 or im.height > 8192 or im.width * im.height > 35_000_000:
            raise ValueError("карта слишком большая; максимум 8192×8192 и 35 млн пикселей")
        base_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    homeland, control, territory = initial_state(base_rgb, reject_contested=True)

    p = paths_for(data_dir, chat_id)
    tmp_root = p.root.with_name(p.root.name + ".new")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(base_rgb, mode="RGB").save(tmp_root / "base.png", optimize=True)

        if terrain_path is not None:
            with Image.open(terrain_path) as tim:
                if tim.format != "PNG":
                    raise ValueError("physical layer must be PNG")
                if tim.size != (base_rgb.shape[1], base_rgb.shape[0]):
                    raise ValueError("physical layer size must match base map")
                tim.convert("RGB").save(tmp_root / "terrain.png", optimize=True)

        if cities_path is not None:
            with Image.open(cities_path) as cim:
                if cim.format != "PNG":
                    raise ValueError("city layer must be PNG")
                if cim.size != (base_rgb.shape[1], base_rgb.shape[0]):
                    raise ValueError("city layer size must match base map")
                rgba = cim.convert("RGBA")
                alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
                if int(alpha.max()) == 0:
                    raise ValueError("city layer is fully transparent")
                if int(alpha.min()) == 255:
                    raise ValueError("city layer needs transparency; opaque background would cover the map")
                rgba.save(tmp_root / "cities.png", optimize=True)

        atomic_save_npy(tmp_root / "homeland.npy", homeland)
        atomic_save_npy(tmp_root / "initial_control.npy", control)
        atomic_save_npy(tmp_root / "current_control.npy", control)
        atomic_save_npy(tmp_root / "territory.npy", territory)
        (tmp_root / "operations").mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            tmp_root / "meta.json",
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "start_date": start_date,
                    "elapsed_days": 0,
                    "next_operation": 1,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        backup = p.root.with_name(p.root.name + ".bak")
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        had_old = p.root.exists()
        if had_old:
            os.replace(p.root, backup)
        try:
            os.replace(tmp_root, p.root)
        except Exception:
            if had_old and backup.exists() and not p.root.exists():
                os.replace(backup, p.root)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
    return p


def load_meta(p: SessionPaths) -> dict:
    return json.loads(p.meta.read_text(encoding="utf-8"))


def save_meta(p: SessionPaths, meta: dict) -> None:
    _atomic_write_text(p.meta, json.dumps(meta, ensure_ascii=False, indent=2))


def ensure_session_schema(p: SessionPaths) -> None:
    if not p.meta.exists():
        return
    meta = load_meta(p)
    if int(meta.get("schema_version", 1)) >= SCHEMA_VERSION and p.homeland.exists() and p.current_control.exists():
        return

    if not p.base.exists():
        raise ValueError("сессия повреждена: нет base.png")

    base = load_rgb(p.base)
    homeland, derived_control, territory = initial_state(base, reject_contested=False)

    initial_control = derived_control
    current_control = derived_control.copy()
    if p.legacy_initial_owner.exists():
        initial_control = np.load(p.legacy_initial_owner).astype(np.uint8)
    if p.legacy_current_owner.exists():
        current_control = np.load(p.legacy_current_owner).astype(np.uint8)

    if initial_control.shape != homeland.shape or current_control.shape != homeland.shape:
        raise ValueError("старая сессия повреждена: размеры массивов не совпадают")

    atomic_save_npy(p.homeland, homeland)
    atomic_save_npy(p.initial_control, initial_control)
    atomic_save_npy(p.current_control, current_control)
    atomic_save_npy(p.territory, territory)
    meta["schema_version"] = SCHEMA_VERSION
    save_meta(p, meta)


def load_state(p: SessionPaths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ensure_session_schema(p)
    homeland = np.load(p.homeland).astype(np.uint8)
    control = np.load(p.current_control).astype(np.uint8)
    territory = np.load(p.territory).astype(bool)
    if homeland.shape != control.shape or homeland.shape != territory.shape:
        raise ValueError("сессия повреждена: homeland/control/territory разного размера")
    return homeland, control, territory


def current_date(p: SessionPaths) -> date:
    meta = load_meta(p)
    return date.fromisoformat(meta["start_date"]) + timedelta(days=int(meta["elapsed_days"]))


def append_operation(p: SessionPaths, spec: dict, masks: dict[str, np.ndarray] | None = None) -> Path:
    """Atomically commit an operation directory and then metadata."""
    meta = load_meta(p)
    idx = int(meta["next_operation"])
    opdir = p.operations / f"{idx:04d}"
    tmpdir = p.operations / f".{idx:04d}.tmp"
    if opdir.exists():
        raise FileExistsError(f"operation {idx} already exists")
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=False)

    try:
        if masks:
            for name, mask in masks.items():
                Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(tmpdir / f"{name}.png", optimize=True)
        _atomic_write_text(tmpdir / "spec.json", json.dumps(spec, ensure_ascii=False, indent=2))
        os.replace(tmpdir, opdir)
    finally:
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)

    meta["next_operation"] = idx + 1
    meta["elapsed_days"] = int(meta["elapsed_days"]) + int(spec["days"])
    try:
        save_meta(p, meta)
    except Exception:
        # Do not leave a committed operation that metadata does not know about.
        shutil.rmtree(opdir, ignore_errors=True)
        raise
    return opdir


def list_operations(p: SessionPaths) -> list[Path]:
    if not p.operations.exists():
        return []
    return sorted(x for x in p.operations.iterdir() if x.is_dir() and x.name.isdigit())


def remove_last_operation(p: SessionPaths) -> bool:
    ops = list_operations(p)
    if not ops:
        return False
    last = ops[-1]
    spec = json.loads((last / "spec.json").read_text(encoding="utf-8"))
    tomb = last.with_name("." + last.name + ".undo")
    if tomb.exists():
        shutil.rmtree(tomb, ignore_errors=True)
    os.replace(last, tomb)
    meta = load_meta(p)
    meta["next_operation"] = max(1, int(meta["next_operation"]) - 1)
    meta["elapsed_days"] = max(0, int(meta["elapsed_days"]) - int(spec["days"]))
    try:
        save_meta(p, meta)
    except Exception:
        os.replace(tomb, last)
        raise
    shutil.rmtree(tomb, ignore_errors=True)
    return True
