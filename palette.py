from __future__ import annotations

import numpy as np

# Native / homeland colors from the original Yogurtstan-war mapping palette.
KEFIR_HOME = np.array([0x4A, 0x00, 0x00], dtype=np.uint8)       # #4A0000
YOGURT_HOME = np.array([0x00, 0x13, 0x4A], dtype=np.uint8)      # #00134A

# Occupation colors. Kefir occupation is the color supplied for the project.
# Yogurt occupation did not have a canonical value in the old bot, so it gets a
# separate configurable blue instead of lying and painting occupied Kefir land
# as native Yogurt territory.
KEFIR_OCCUPATION = np.array([0xB1, 0x28, 0x23], dtype=np.uint8)  # #B12823
YOGURT_OCCUPATION = np.array([0x1F, 0x52, 0xA6], dtype=np.uint8) # #1F52A6

CONTESTED = np.array([0x70, 0x77, 0x80], dtype=np.uint8)        # #707780
LEGACY_HIGHLIGHT = np.array([0xFF, 0xA7, 0xA4], dtype=np.uint8)  # #FFA7A4, old temporary highlight
NEUTRAL = np.array([0x9D, 0x9D, 0x9D], dtype=np.uint8)          # #9D9D9D

# Command colors. Input only; never rendered to the final map.
ORDER_KEFIR = np.array([0xFF, 0x00, 0xFF], dtype=np.uint8)       # #FF00FF
ORDER_YOGURT = np.array([0x00, 0xFF, 0x66], dtype=np.uint8)      # #00FF66

# Immutable homeland layer.
HOME_NONE = 0
HOME_KEFIR = 1
HOME_YOGURT = 2

# Mutable control layer.
CONTROL_NONE = 0
CONTROL_KEFIR = 1
CONTROL_YOGURT = 2
CONTROL_CONTESTED = 3

SIDE_TO_CONTROL = {
    "kefir": CONTROL_KEFIR,
    "k": CONTROL_KEFIR,
    "кефир": CONTROL_KEFIR,
    "кефирстан": CONTROL_KEFIR,
    "yogurt": CONTROL_YOGURT,
    "y": CONTROL_YOGURT,
    "йогурт": CONTROL_YOGURT,
    "йогуртстан": CONTROL_YOGURT,
}

# Compatibility aliases for older modules / saved specs.
KEFIR = KEFIR_HOME
YOGURT = YOGURT_HOME
GRAY = CONTESTED
OWNER_NONE = CONTROL_NONE
OWNER_KEFIR = CONTROL_KEFIR
OWNER_YOGURT = CONTROL_YOGURT
OWNER_GRAY = CONTROL_CONTESTED
SIDE_TO_OWNER = SIDE_TO_CONTROL


def rgb_hex(color: np.ndarray) -> str:
    r, g, b = (int(x) for x in color)
    return f"#{r:02X}{g:02X}{b:02X}"
