from typing import Dict, Literal

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


Color = Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"]

COLOR_INDEX: Dict[Color, int] = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}


def colorize(s: str, color: Color) -> str:
    return COLOR_SEQ % COLOR_INDEX[color] + s + RESET_SEQ
