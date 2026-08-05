"""Touch-typing finger maps for supported keyboard layouts."""

from __future__ import annotations


US_ANSI_QWERTY = "US ANSI QWERTY"


def _build_us_ansi() -> dict[str, str]:
    groups = {
        "Left pinky": "`1qaz",
        "Left ring": "2wsx",
        "Left middle": "3edc",
        "Left index": "45rtfgvb",
        "Right index": "67yuhjnm",
        "Right middle": "8ik,",
        "Right ring": "9ol.",
        "Right pinky": "0-p;/=[]\\'",
    }
    mapping = {
        character: finger
        for finger, characters in groups.items()
        for character in characters
    }
    mapping.update(
        {
            "space": "Thumb",
            "tab": "Left pinky",
            "caps_lock": "Left pinky",
            "shift": "Left pinky",
            "shift_l": "Left pinky",
            "shift_r": "Right pinky",
            "backspace": "Right pinky",
            "enter": "Right pinky",
        }
    )
    return mapping


LAYOUTS = {US_ANSI_QWERTY: _build_us_ansi()}


def normalize_key_label(label: str) -> str:
    if len(label) >= 3 and label[0] == label[-1] == "'":
        value = label[1:-1]
        if len(value) == 1:
            return "space" if value == " " else value.lower()
    if label == " ":
        return "space"
    return label.lower()


def expected_finger(label: str, layout: str = US_ANSI_QWERTY) -> str:
    mapping = LAYOUTS.get(layout, LAYOUTS[US_ANSI_QWERTY])
    normalized = normalize_key_label(label)
    if normalized in mapping:
        return mapping[normalized]
    if normalized and normalized[0] in mapping:
        return mapping[normalized[0]]
    return "Unmapped"
