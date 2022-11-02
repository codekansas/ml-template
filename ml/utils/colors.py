from typing import Dict, Literal, Tuple

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


Color = Literal["black", "red", "green", "yellow", "blue", "magenta", "cyan", "white", "grey"]

COLOR_INDEX: Dict[Color, int] = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "grey": 90,
}


def get_colorize_parts(color: Color) -> Tuple[str, str]:
    return COLOR_SEQ % COLOR_INDEX[color], RESET_SEQ


def colorize(s: str, color: Color) -> str:
    start, end = get_colorize_parts(color)
    return start + s + end
