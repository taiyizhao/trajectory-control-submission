from __future__ import annotations

import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
OUT = ROOT / "README.pdf"

W, H = 1240, 1754
M = 82
CONTENT_W = W - 2 * M
BG = "white"
INK = "#172129"
RULE = "#d8e1e6"
CODE_BG = "#f5f7f9"
TABLE_BORDER = "#cfd9df"
TABLE_HEAD = "#edf3f6"
TABLE_ALT = "#fbfcfd"


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F = {
    "h1": load_font("segoeuib.ttf", 40),
    "h2": load_font("segoeuib.ttf", 29),
    "h3": load_font("segoeuib.ttf", 23),
    "body": load_font("segoeui.ttf", 20),
    "mono": load_font("consola.ttf", 17),
    "table": load_font("segoeui.ttf", 16),
    "table_bold": load_font("segoeuib.ttf", 16),
    "page": load_font("segoeui.ttf", 14),
}

pages: list[Image.Image] = []
img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)
y = M
page_num = 1


def text_w(text: str, font) -> int:
    return draw.textbbox((0, 0), text, font=font)[2]


def line_h(font, extra: int = 6) -> int:
    box = draw.textbbox((0, 0), "Ag", font=font)
    return box[3] - box[1] + extra


BODY_LH = line_h(F["body"], 8)
MONO_LH = line_h(F["mono"], 6)
TABLE_LH = line_h(F["table"], 5)


def clean_inline(text: str) -> str:
    return text.strip().replace("`", "").replace("**", "")


def finish_page() -> None:
    global img, draw, y, page_num
    footer = f"- {page_num} -"
    draw.text(((W - text_w(footer, F["page"])) / 2, H - 50), footer, font=F["page"], fill="#687782")
    pages.append(img)
    page_num += 1
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    y = M


def ensure(height: int) -> None:
    if y + height > H - M:
        finish_page()


def wrap_pixel(text: str, font, max_w: int) -> list[str]:
    text = clean_inline(text)
    if not text:
        return [""]
    out: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else current + " " + word
        if text_w(candidate, font) <= max_w:
            current = candidate
            continue
        if current:
            out.append(current)
            current = ""
        if text_w(word, font) <= max_w:
            current = word
            continue
        chunk = ""
        for ch in word:
            if text_w(chunk + ch, font) <= max_w:
                chunk += ch
            else:
                if chunk:
                    out.append(chunk)
                chunk = ch
        current = chunk
    if current:
        out.append(current)
    return out or [""]


def draw_wrapped(text: str, prefix: str = "") -> None:
    global y
    prefix_w = text_w(prefix, F["body"]) if prefix else 0
    wrapped = wrap_pixel(text, F["body"], CONTENT_W - prefix_w)
    ensure(len(wrapped) * BODY_LH + 4)
    for i, line in enumerate(wrapped):
        if prefix and i == 0:
            draw.text((M, y), prefix, font=F["body"], fill=INK)
        draw.text((M + prefix_w, y), line, font=F["body"], fill=INK)
        y += BODY_LH


def draw_code_line(text: str) -> None:
    global y
    chunks = textwrap.wrap(text, width=104, replace_whitespace=False, drop_whitespace=False) or [""]
    for chunk in chunks:
        ensure(MONO_LH + 8)
        draw.rectangle((M - 8, y - 3, W - M + 8, y + MONO_LH + 1), fill=CODE_BG)
        draw.text((M, y), chunk, font=F["mono"], fill=INK)
        y += MONO_LH


def is_sep_row(row: str) -> bool:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    block: list[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        block.append(lines[i])
        i += 1
    rows: list[list[str]] = []
    for row in block:
        if is_sep_row(row):
            continue
        rows.append([clean_inline(c) for c in row.strip().strip("|").split("|")])
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]
    return rows, i


def column_widths(rows: list[list[str]]) -> list[int]:
    n = len(rows[0])
    if n == 2:
        first = int(CONTENT_W * 0.36)
        return [first, CONTENT_W - first]
    if n == 3:
        widths = [int(CONTENT_W * 0.44), int(CONTENT_W * 0.25)]
        return widths + [CONTENT_W - sum(widths)]
    if n == 4:
        widths = [int(CONTENT_W * 0.28), int(CONTENT_W * 0.22), int(CONTENT_W * 0.22)]
        return widths + [CONTENT_W - sum(widths)]
    if n == 6:
        weights = [0.18, 0.16, 0.14, 0.15, 0.17]
        widths = [int(CONTENT_W * w) for w in weights]
        return widths + [CONTENT_W - sum(widths)]
    base = CONTENT_W // n
    widths = [base] * n
    widths[-1] += CONTENT_W - sum(widths)
    return widths


def draw_table(rows: list[list[str]]) -> None:
    global y
    widths = column_widths(rows)
    pad_x, pad_y = 7, 7
    y += 3
    for r, row in enumerate(rows):
        fonts = [F["table_bold"] if r == 0 else F["table"]] * len(row)
        wrapped = [wrap_pixel(cell, fonts[c], widths[c] - 2 * pad_x) for c, cell in enumerate(row)]
        row_h = max(len(cell) for cell in wrapped) * TABLE_LH + 2 * pad_y
        ensure(row_h + 4)
        bg = TABLE_HEAD if r == 0 else (TABLE_ALT if r % 2 == 0 else BG)
        x = M
        for c, cell in enumerate(wrapped):
            draw.rectangle((x, y, x + widths[c], y + row_h), fill=bg, outline=TABLE_BORDER, width=1)
            ty = y + pad_y
            for line in cell:
                draw.text((x + pad_x, ty), line, font=fonts[c], fill=INK)
                ty += TABLE_LH
            x += widths[c]
        y += row_h
    y += 10


def main() -> None:
    global y
    lines = README.read_text(encoding="utf-8").splitlines()
    in_code = False
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            y += 5 if in_code else 9
            i += 1
            continue
        if in_code:
            draw_code_line(raw.rstrip())
            i += 1
            continue
        if not stripped:
            y += 8
            i += 1
            continue
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            draw_table(rows)
            continue
        if stripped.startswith("# "):
            title = clean_inline(stripped[2:])
            ensure(76)
            draw.text((M, y), title, font=F["h1"], fill=INK)
            y += line_h(F["h1"], 14)
            draw.line((M, y, W - M, y), fill=RULE, width=2)
            y += 18
            i += 1
            continue
        if stripped.startswith("## "):
            title = clean_inline(stripped[3:])
            ensure(62)
            y += 8
            draw.text((M, y), title, font=F["h2"], fill=INK)
            y += line_h(F["h2"], 9)
            draw.line((M, y, W - M, y), fill="#e4ebef", width=1)
            y += 10
            i += 1
            continue
        if stripped.startswith("### "):
            ensure(44)
            draw.text((M, y), clean_inline(stripped[4:]), font=F["h3"], fill=INK)
            y += line_h(F["h3"], 10)
            i += 1
            continue
        if stripped.startswith("- "):
            draw_wrapped(stripped[2:], prefix="- ")
            i += 1
            continue
        draw_wrapped(stripped)
        i += 1

    if y > M:
        finish_page()
    pages[0].save(OUT, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    print(OUT)


if __name__ == "__main__":
    main()
