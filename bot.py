from __future__ import annotations

import asyncio
import os
import re
import secrets
from collections import defaultdict
from pathlib import Path

import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message
from PIL import Image

from engine import (
    apply_attack,
    apply_turn,
    build_attack_plan,
    city_mask_from_rgba,
    extract_order_masks,
    load_rgb,
    save_rgb,
    validate_annotation_against_reference,
)
from exporter import export_mp4
from canon import WAR_END, WAR_NAME, WAR_START, canon_hint
from palette import (
    CONTESTED,
    CONTROL_CONTESTED,
    CONTROL_KEFIR,
    CONTROL_YOGURT,
    HOME_KEFIR,
    HOME_YOGURT,
    KEFIR_HOME,
    KEFIR_OCCUPATION,
    ORDER_KEFIR,
    ORDER_YOGURT,
    SIDE_TO_CONTROL,
    YOGURT_HOME,
    YOGURT_OCCUPATION,
    rgb_hex,
)
from replay import replay_control
from visual import load_optional_visual_layers, render_visual
from storage import (
    append_operation,
    atomic_save_npy,
    create_session,
    current_date,
    load_meta,
    load_state,
    paths_for,
    remove_last_operation,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DEMO_MAP = ROOT / "yogurtstan_base_control.png"
DEMO_TERRAIN = ROOT / "yogurtstan_terrain.png"
DEMO_CITIES = ROOT / "yogurtstan_cities.png"
OWNER_ID = int(os.getenv("OWNER_ID", "0") or 0)
PENDING: dict[int, dict] = {}
LOCKS: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def allowed(message: Message) -> bool:
    return not OWNER_ID or (message.from_user and message.from_user.id == OWNER_ID)


def _parse_int(value: str, lo: int, hi: int) -> int:
    n = int(value)
    if not (lo <= n <= hi):
        raise ValueError(f"значение должно быть от {lo} до {hi}")
    return n


def _seed() -> int:
    return secrets.randbelow(2_000_000_000 - 1) + 1


def help_text() -> str:
    return (
        "Aurelia War Mapper ULTRA v5\n\n"
        "Главный режим:\n"
        "1) /demo\n"
        "2) /map или /map4k\n"
        "2.5) /canon [kefir|yogurt]\n"
        "3) /turn 60 110\n"
        "4) На СВЕЖЕЙ карте рисуй линии и пришли PNG как файл:\n"
        f"   {rgb_hex(ORDER_KEFIR)} — Кефирстан\n"
        f"   {rgb_hex(ORDER_YOGURT)} — Йогуртстан\n"
        "   Линии могут быть кривыми, горизонтальными, диагональными и несколькими\n"
        "5) /wait 30\n"
        "6) /undo\n"
        "7) /doctor — проверить и при необходимости восстановить историю\n"
        "8) /terrain и /cities — заменить визуальные слои\n"
        "9) /export 30 1920, /export4k 30 или /exportcurve 30 1920\n\n"
        "Серая зона — рваная кромка 3-10 px на исходной карте.\n"
        "Демо Йогуртстанской войны стартует с 28.06.2057 — первого дня боевых действий.\n"
        "Родная земля и текущий контроль хранятся отдельно, поэтому оккупация красится правильно.\n"
        "Физический слой лежит под политическими цветами, города всегда поверх.\n"
        "Цвета: /colors"
    )


def _resize_rgb_to_width(arr: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or width <= 0 or arr.shape[1] == width:
        return arr
    h, w = arr.shape[:2]
    if w == width:
        return arr
    scale = width / float(w)
    oh = max(2, int(round(h * scale / 2) * 2))
    return np.asarray(Image.fromarray(arr, mode="RGB").resize((width, oh), resample=Image.Resampling.LANCZOS), dtype=np.uint8)


async def send_current(message: Message, width: int | None = None) -> None:
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap.")
        return
    try:
        homeland, control, territory = load_state(p)
        base = load_rgb(p.base)
        terrain, cities = load_optional_visual_layers(p.terrain, p.cities, base.shape[:2])
        rendered = render_visual(base, homeland, control, territory, terrain=terrain, cities=cities)
        rendered = _resize_rgb_to_width(rendered, width)
        out = p.root / ("current_render_4k.png" if width else "current_render.png")
        save_rgb(out, rendered)
    except Exception as e:
        await message.answer(f"Сессия повреждена или не мигрировалась: {e}")
        return
    await message.answer_document(FSInputFile(out), caption=f"Текущая дата: {current_date(p).strftime('%d.%m.%Y')}")


async def cmd_start(message: Message):
    if allowed(message):
        await message.answer(help_text())


async def cmd_colors(message: Message):
    if not allowed(message):
        return
    await message.answer(
        "Палитра:\n"
        f"Родной Кефирстан: {rgb_hex(KEFIR_HOME)}\n"
        f"Родной Йогуртстан: {rgb_hex(YOGURT_HOME)}\n"
        f"Кефирская оккупация Йогуртстана: {rgb_hex(KEFIR_OCCUPATION)}\n"
        f"Йогуртстанская оккупация Кефирстана: {rgb_hex(YOGURT_OCCUPATION)}\n"
        f"Серая зона: {rgb_hex(CONTESTED)}\n"
        f"Приказ Кефирстана: {rgb_hex(ORDER_KEFIR)}\n"
        f"Приказ Йогуртстана: {rgb_hex(ORDER_YOGURT)}"
    )


async def cmd_demo(message: Message):
    if not allowed(message):
        return
    async with LOCKS[message.chat.id]:
        try:
            create_session(
                DATA_DIR,
                message.chat.id,
                DEMO_MAP,
                start_date="2057-06-28",
                terrain_path=DEMO_TERRAIN,
                cities_path=DEMO_CITIES,
            )
        except Exception as e:
            await message.answer(f"Не смог создать демо: {e}")
            return
        PENDING.pop(message.chat.id, None)
    await message.answer("Карта загружена. Это каноническое демо Йогуртстанской войны: старт 28.06.2057, физический слой и города уже подключены.")
    await send_current(message)


async def cmd_newmap(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    start_date = "2057-01-01"
    if len(parts) >= 2:
        try:
            # create_session validates ISO date too, but fail early here.
            from datetime import date
            date.fromisoformat(parts[1])
            start_date = parts[1]
        except ValueError:
            await message.answer("Дата нужна как YYYY-MM-DD, например /newmap 2057-01-01")
            return
    PENDING[message.chat.id] = {"kind": "newmap", "start_date": start_date}
    await message.answer(
        "Пришли базовую PNG-карту как ФАЙЛ. Серой зоны на стартовой карте быть не должно: "
        "у серого пикселя невозможно угадать родину. Оккупационные цвета разрешены."
    )


async def cmd_terrain(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap")
        return
    PENDING[message.chat.id] = {"kind": "terrain"}
    await message.answer("Пришли физический слой PNG как ФАЙЛ. Размер должен один в один совпадать с картой.")


async def cmd_cities(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap")
        return
    PENDING[message.chat.id] = {"kind": "cities"}
    await message.answer("Пришли слой городов PNG с прозрачным фоном как ФАЙЛ. Города будут рисоваться поверх фронта и физики.")


async def cmd_layers(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Нет сессии.")
        return
    await message.answer(
        "Визуальные слои:\n"
        f"Физика: {'есть' if p.terrain.exists() else 'нет'}\n"
        f"Города: {'есть' if p.cities.exists() else 'нет'}\n"
        "Физика смешивается с политическими цветами только внутри территорий; серая зона остается плоской."
    )


async def cmd_clearterrain(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    async with LOCKS[message.chat.id]:
        p.terrain.unlink(missing_ok=True)
        PENDING.pop(message.chat.id, None)
    await message.answer("Физический слой убран.")
    await send_current(message)


async def cmd_clearcities(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    async with LOCKS[message.chat.id]:
        p.cities.unlink(missing_ok=True)
        PENDING.pop(message.chat.id, None)
    await message.answer("Слой городов убран.")
    await send_current(message)


async def cmd_turn(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    if len(parts) < 2:
        await message.answer("Формат: /turn 60 [width_px]. Например /turn 120 110")
        return
    try:
        days = _parse_int(parts[1], 1, 5000)
        width = 110 if len(parts) < 3 else _parse_int(parts[2], 8, 800)
    except (ValueError, TypeError) as e:
        await message.answer(f"Кривые параметры: {e}")
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap")
        return
    PENDING[message.chat.id] = {
        "kind": "turn",
        "days": days,
        "width_px": width,
        "gray_width_px": 9.0,
        "front_noise_px": 11.0,
    }
    await message.answer(
        f"Ход: {days} дней. Возьми СВЕЖИЙ /map и рисуй поверх него:\n"
        f"{rgb_hex(ORDER_KEFIR)} — Кефирстан\n"
        f"{rgb_hex(ORDER_YOGURT)} — Йогуртстан\n"
        "Кривая/горизонтальная/ломаная линия нормальна. Несколько линий тоже. PNG как файл."
    )


async def cmd_attack(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    if len(parts) < 3:
        await message.answer("Формат: /attack kefir|yogurt DAYS [width_px]")
        return
    side_raw = parts[1].lower()
    if side_raw not in SIDE_TO_CONTROL:
        await message.answer("Сторона: kefir или yogurt.")
        return
    try:
        days = _parse_int(parts[2], 1, 5000)
        width = 110 if len(parts) < 4 else _parse_int(parts[3], 8, 800)
    except (ValueError, TypeError) as e:
        await message.answer(f"Кривые параметры: {e}")
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap")
        return
    attacker = SIDE_TO_CONTROL[side_raw]
    side = "kefir" if attacker == CONTROL_KEFIR else "yogurt"
    PENDING[message.chat.id] = {
        "kind": "attack",
        "side": side,
        "days": days,
        "width_px": width,
        "gray_width_px": 9.0,
        "front_noise_px": 11.0,
    }
    color = rgb_hex(ORDER_KEFIR if side == "kefir" else ORDER_YOGURT)
    await message.answer(
        f"{side}: {days} дней. Нарисуй линию {color} на СВЕЖЕМ /map от своей зоны в сторону цели и пришли PNG как файл."
    )




async def cmd_canon(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    side = None
    parts = (message.text or "").split()
    if len(parts) >= 2:
        raw = parts[1].lower()
        if raw in ("kefir", "yogurt"):
            side = raw
    if not p.meta.exists():
        await message.answer(canon_hint(WAR_START, side=side))
        return
    await message.answer(canon_hint(current_date(p), side=side))


async def cmd_wait(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала /demo или /newmap")
        return
    parts = (message.text or "").split()
    if len(parts) != 2:
        await message.answer("Формат: /wait 30")
        return
    try:
        days = _parse_int(parts[1], 1, 5000)
    except ValueError as e:
        await message.answer(f"Нужно нормальное число дней: {e}")
        return
    async with LOCKS[message.chat.id]:
        append_operation(p, {"kind": "wait", "days": days})
        PENDING.pop(message.chat.id, None)
    await message.answer(f"Промотал {days} дней. Дата: {current_date(p).strftime('%d.%m.%Y')}")


async def cmd_map(message: Message):
    if allowed(message):
        await send_current(message)


async def cmd_map4k(message: Message):
    if allowed(message):
        await send_current(message, width=3840)


async def cmd_status(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Нет войны. /demo")
        return
    try:
        homeland, control, territory = load_state(p)
        meta = load_meta(p)
        k_occ = int(((homeland == HOME_YOGURT) & (control == CONTROL_KEFIR)).sum())
        y_occ = int(((homeland == HOME_KEFIR) & (control == CONTROL_YOGURT)).sum())
        gray = int((territory & (control == CONTROL_CONTESTED)).sum())
    except Exception as e:
        await message.answer(f"Сессия сломана: {e}")
        return
    await message.answer(
        f"Старт: {meta['start_date']}\n"
        f"Прошло дней: {meta['elapsed_days']}\n"
        f"Текущая дата: {current_date(p).strftime('%d.%m.%Y')}\n"
        f"Ходов: {int(meta['next_operation']) - 1}\n"
        f"Кефирская оккупация: {k_occ:,} px\n"
        f"Йогуртстанская оккупация: {y_occ:,} px\n"
        f"Серая зона: {gray:,} px"
    )


async def cmd_cancel(message: Message):
    if not allowed(message):
        return
    if PENDING.pop(message.chat.id, None):
        await message.answer("Ожидавшийся приказ отменен.")
    else:
        await message.answer("Нечего отменять.")


async def cmd_undo(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    async with LOCKS[message.chat.id]:
        if not p.meta.exists() or not remove_last_operation(p):
            await message.answer("Откатывать нечего.")
            return
        try:
            control = replay_control(p)
            atomic_save_npy(p.current_control, control)
        except Exception as e:
            await message.answer(f"Операция удалена, но переигровка сломалась: {e}")
            return
        PENDING.pop(message.chat.id, None)
    await message.answer(f"Последний ход похоронен. Дата снова {current_date(p).strftime('%d.%m.%Y')}.")
    await send_current(message)


async def cmd_doctor(message: Message):
    if not allowed(message):
        return
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Нет сессии для проверки.")
        return
    await message.answer("Переигрываю историю и сверяю состояние. Это может занять немного времени.")
    loop = asyncio.get_running_loop()
    async with LOCKS[message.chat.id]:
        try:
            homeland, cached, territory = load_state(p)
            base = load_rgb(p.base)
            load_optional_visual_layers(p.terrain, p.cities, base.shape[:2])
            replayed = await loop.run_in_executor(None, lambda: replay_control(p))
            diff = int(np.count_nonzero(cached != replayed))
            if diff:
                atomic_save_npy(p.current_control, replayed)
                repaired = True
            else:
                repaired = False
            if homeland.shape != territory.shape or homeland.shape != replayed.shape:
                raise ValueError("размеры state-массивов не совпадают")
            meta = load_meta(p)
        except Exception as e:
            await message.answer(f"Doctor нашел тяжелую поломку и не стал выдумывать лечение: {e}")
            return
    if repaired:
        await message.answer(
            f"Нашел рассинхрон на {diff:,} пикселях и восстановил current_control из журнала операций. "
            f"Дата: {current_date(p).strftime('%d.%m.%Y')}"
        )
    else:
        await message.answer(
            f"Сессия здорова: replay совпадает с кешем, операций {int(meta['next_operation']) - 1}, "
            f"дата {current_date(p).strftime('%d.%m.%Y')}."
        )


async def _run_export(message: Message, fps: int, width: int, curved: bool = False) -> None:
    p = paths_for(DATA_DIR, message.chat.id)
    if not p.meta.exists():
        await message.answer("Сначала создай войну.")
        return
    suffix = "_4k" if width >= 3800 else ""
    curve_suffix = "_curve" if curved else ""
    out = p.root / f"war_export{curve_suffix}{suffix}.mp4"
    curve_text = ", curved mode" if curved else ""
    await message.answer(f"Рендерю {load_meta(p)['elapsed_days']} дней: {fps} fps, до {width}px{curve_text}.")
    loop = asyncio.get_running_loop()
    async with LOCKS[message.chat.id]:
        try:
            await loop.run_in_executor(None, lambda: export_mp4(p, out, fps=fps, max_width=width, stamp_dates=True, curved=curved))
        except Exception as e:
            await message.answer(f"Экспорт сдох: {e}")
            return
    caption = "Война целиком: стартовый кадр + 1 кадр на каждый прошедший день."
    if curved:
        caption += " Экспорт в слегка изогнутом режиме."
    try:
        await message.answer_video(FSInputFile(out), caption=caption, supports_streaming=True)
    except Exception:
        await message.answer_document(FSInputFile(out), caption=caption)


async def cmd_export(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    try:
        fps = 30 if len(parts) < 2 else _parse_int(parts[1], 1, 60)
        width = 1920 if len(parts) < 3 else _parse_int(parts[2], 640, 4096)
    except ValueError as e:
        await message.answer(f"Формат: /export 30 1920. {e}")
        return
    await _run_export(message, fps, width)


async def cmd_export4k(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    try:
        fps = 30 if len(parts) < 2 else _parse_int(parts[1], 1, 60)
    except ValueError as e:
        await message.answer(f"Формат: /export4k 30. {e}")
        return
    await _run_export(message, fps, 3840)


async def cmd_exportcurve(message: Message):
    if not allowed(message):
        return
    parts = (message.text or "").split()
    try:
        fps = 30 if len(parts) < 2 else _parse_int(parts[1], 1, 60)
        width = 1920 if len(parts) < 3 else _parse_int(parts[2], 640, 4096)
    except ValueError as e:
        await message.answer(f"Формат: /exportcurve 30 1920. {e}")
        return
    await _run_export(message, fps, width, curved=True)


async def _download_png_document(message: Message, bot: Bot, root: Path) -> Path | None:
    if not message.document:
        return None
    name = message.document.file_name or "upload.png"
    if not name.lower().endswith(".png"):
        return None
    if message.document.file_size and message.document.file_size > 60 * 1024 * 1024:
        return None
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    await bot.download(message.document, destination=tmp)
    return tmp


def _validate_png_dimensions(path: Path, expected: tuple[int, int] | None = None) -> tuple[int, int]:
    with Image.open(path) as im:
        if im.format != "PNG":
            raise ValueError("нужен настоящий PNG, а не JPEG с усами .png")
        w, h = im.size
        if w < 64 or h < 64:
            raise ValueError("картинка слишком маленькая")
        if w > 8192 or h > 8192 or w * h > 35_000_000:
            raise ValueError("картинка слишком большая; лимит 35 млн пикселей")
        if expected and (w, h) != expected:
            raise ValueError(f"размер должен быть {expected[0]}×{expected[1]}, а пришло {w}×{h}")
        return w, h


async def handle_document(message: Message, bot: Bot):
    if not allowed(message) or not message.document:
        return
    pending = PENDING.get(message.chat.id)
    if not pending:
        return

    tmp = await _download_png_document(message, bot, DATA_DIR / str(message.chat.id) / "incoming")
    if tmp is None:
        await message.answer("Нужен PNG как ФАЙЛ. Не фото, не JPEG, не скриншот через мясорубку Telegram.")
        return

    async with LOCKS[message.chat.id]:
        # Pending may have been consumed while this document was waiting for the lock.
        pending = PENDING.get(message.chat.id)
        if not pending:
            return

        if pending["kind"] == "newmap":
            try:
                _validate_png_dimensions(tmp)
                create_session(DATA_DIR, message.chat.id, tmp, start_date=pending.get("start_date", "2057-01-01"))
            except Exception as e:
                await message.answer(f"Карту не принял: {e}")
                return
            PENDING.pop(message.chat.id, None)
            await message.answer("Карта принята. Родная принадлежность сохранена отдельно от контроля.")
            await send_current(message)
            return

        p = paths_for(DATA_DIR, message.chat.id)
        if not p.meta.exists():
            await message.answer("Сначала /demo или /newmap")
            return

        if pending["kind"] in {"terrain", "cities"}:
            base = load_rgb(p.base)
            try:
                _validate_png_dimensions(tmp, expected=(base.shape[1], base.shape[0]))
                with Image.open(tmp) as im:
                    if pending["kind"] == "terrain":
                        layer = im.convert("RGB")
                        dest = p.terrain
                    else:
                        layer = im.convert("RGBA")
                        alpha = np.asarray(layer.getchannel("A"), dtype=np.uint8)
                        if int(alpha.max()) == 0:
                            raise ValueError("слой городов полностью прозрачный")
                        if int(alpha.min()) == 255:
                            raise ValueError("у слоя городов нет прозрачности; непрозрачный фон закроет всю карту")
                        dest = p.cities
                    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
                    layer.save(tmp_dest, format="PNG", optimize=True)
                    os.replace(tmp_dest, dest)
            except Exception as e:
                await message.answer(f"Слой не принял: {e}")
                return
            PENDING.pop(message.chat.id, None)
            await message.answer("Слой установлен. Возьми новый /map перед следующим приказом.")
            await send_current(message)
            return

        base = load_rgb(p.base)
        try:
            _validate_png_dimensions(tmp, expected=(base.shape[1], base.shape[0]))
            annotated = load_rgb(tmp)
            homeland, control, territory = load_state(p)
        except Exception as e:
            await message.answer(f"Не смог прочитать карту: {e}")
            return

        terrain, cities = load_optional_visual_layers(p.terrain, p.cities, base.shape[:2])
        city_mask = city_mask_from_rgba(cities)
        reference = render_visual(base, homeland, control, territory, terrain=terrain, cities=cities)
        masks = extract_order_masks(annotated)
        ok, reason = validate_annotation_against_reference(annotated, reference, masks)
        if not ok:
            await message.answer(reason or "Это не последний кадр карты.")
            return

        days = int(pending["days"])
        width = int(pending["width_px"])
        gray = float(pending["gray_width_px"])
        noise = float(pending["front_noise_px"])

        if pending["kind"] == "attack":
            attacker = CONTROL_KEFIR if pending["side"] == "kefir" else CONTROL_YOGURT
            order = masks[attacker]
            if not order.any():
                await message.answer("Не вижу линии нужной стороны точного цвета. Приказ пока не записан.")
                return
            seed = _seed()
            try:
                plan = build_attack_plan(control, territory, order, attacker, days, width, gray, noise, seed)
                final_control = apply_attack(control, plan, homeland=homeland, territory=territory, city_mask=city_mask)
            except ValueError as e:
                await message.answer(f"Не смог построить наступление: {e}")
                return

            spec = {
                "kind": "attack",
                "side": pending["side"],
                "days": days,
                "width_px": width,
                "gray_width_px": gray,
                "front_noise_px": noise,
                "seed": seed,
                "order_name": "order",
            }
            try:
                append_operation(p, spec, {"order": order})
                atomic_save_npy(p.current_control, final_control)
            except Exception as e:
                # History is authoritative. If the cache write failed after the
                # operation commit, rebuild it immediately.
                try:
                    atomic_save_npy(p.current_control, replay_control(p))
                except Exception:
                    pass
                await message.answer(f"Не смог надежно записать ход: {e}")
                return

            PENDING.pop(message.chat.id, None)
            msg = f"Наступление проведено за {days} дней. Дата: {current_date(p).strftime('%d.%m.%Y')}"
            if plan.warnings:
                msg += "\nПредупреждения:\n- " + "\n- ".join(plan.warnings)
            await message.answer(msg)
            await send_current(message)
            return

        if pending["kind"] == "turn":
            has_k = bool(masks[CONTROL_KEFIR].any())
            has_y = bool(masks[CONTROL_YOGURT].any())
            if not has_k and not has_y:
                await message.answer("Не вижу ни розовой, ни зеленой линии. Генштаб прислал чистую карту, мощно.")
                return

            kp = yp = None
            spec = {"kind": "turn", "days": days, "kefir": None, "yogurt": None}
            save_masks: dict[str, np.ndarray] = {}
            errors: list[str] = []
            warnings: list[str] = []

            if has_k:
                ks = _seed()
                try:
                    kp = build_attack_plan(control, territory, masks[CONTROL_KEFIR], CONTROL_KEFIR, days, width, gray, noise, ks)
                    spec["kefir"] = {"width_px": width, "gray_width_px": gray, "front_noise_px": noise, "seed": ks}
                    save_masks["kefir"] = masks[CONTROL_KEFIR]
                    warnings += ["Кефирстан: " + w for w in kp.warnings]
                except ValueError as e:
                    errors.append(f"Кефирстан: {e}")

            if has_y:
                ys = _seed()
                try:
                    yp = build_attack_plan(control, territory, masks[CONTROL_YOGURT], CONTROL_YOGURT, days, width, gray, noise, ys)
                    spec["yogurt"] = {"width_px": width, "gray_width_px": gray, "front_noise_px": noise, "seed": ys}
                    save_masks["yogurt"] = masks[CONTROL_YOGURT]
                    warnings += ["Йогуртстан: " + w for w in yp.warnings]
                except ValueError as e:
                    errors.append(f"Йогуртстан: {e}")

            # Atomic semantics: if the user drew an order for a side and it is
            # invalid, reject the whole turn. No half-turn surprise bullshit.
            if errors:
                await message.answer("Ход НЕ записан. Исправь линии:\n- " + "\n- ".join(errors))
                return

            try:
                final_control = apply_turn(control, kp, yp, days, homeland=homeland, territory=territory, city_mask=city_mask)
                append_operation(p, spec, save_masks)
                atomic_save_npy(p.current_control, final_control)
            except Exception as e:
                try:
                    atomic_save_npy(p.current_control, replay_control(p))
                except Exception:
                    pass
                await message.answer(f"Ход не удалось надежно записать: {e}")
                return

            PENDING.pop(message.chat.id, None)
            msg = f"Ход на {days} дней проведен. Дата: {current_date(p).strftime('%d.%m.%Y')}"
            if warnings:
                msg += "\nПредупреждения:\n- " + "\n- ".join(warnings)
            await message.answer(msg)
            await send_current(message)


async def main() -> None:
    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set BOT_TOKEN environment variable")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bot = Bot(token=token)
    dp = Dispatcher()
    dp.message.register(cmd_start, CommandStart())
    dp.message.register(cmd_demo, Command("demo"))
    dp.message.register(cmd_newmap, Command("newmap"))
    dp.message.register(cmd_terrain, Command("terrain"))
    dp.message.register(cmd_cities, Command("cities"))
    dp.message.register(cmd_layers, Command("layers"))
    dp.message.register(cmd_clearterrain, Command("clearterrain"))
    dp.message.register(cmd_clearcities, Command("clearcities"))
    dp.message.register(cmd_turn, Command("turn"))
    dp.message.register(cmd_attack, Command("attack"))
    dp.message.register(cmd_wait, Command("wait"))
    dp.message.register(cmd_canon, Command("canon"))
    dp.message.register(cmd_map, Command("map"))
    dp.message.register(cmd_map4k, Command("map4k"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(cmd_doctor, Command("doctor"))
    dp.message.register(cmd_colors, Command("colors"))
    dp.message.register(cmd_cancel, Command("cancel"))
    dp.message.register(cmd_undo, Command("undo"))
    dp.message.register(cmd_export, Command("export"))
    dp.message.register(cmd_export4k, Command("export4k"))
    dp.message.register(cmd_exportcurve, Command("exportcurve"))
    dp.message.register(handle_document, F.document)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
