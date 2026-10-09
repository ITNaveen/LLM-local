"""Hindi text as transparent pictures (Pillow + raqm shaping), laid over the video by ffmpeg.

This draws Devanagari correctly (e.g. the ि sign in विराट goes before its consonant) with
any ffmpeg build, so headlines, the title, subtitles and the thumbnail never depend on a
special ffmpeg ('ffmpeg-full' / libass)."""

import re
from functools import lru_cache
from pathlib import Path

from .config import FONTS_DIR

BOLD = FONTS_DIR / "NotoSansDevanagari-Bold.ttf"
LATIN = FONTS_DIR / "NotoSans-Bold.ttf"        # English words inside Hindi lines ("CJP", "vs")
YELLOW, WHITE, RED = (255, 214, 0, 255), (255, 255, 255, 255), (214, 31, 38, 255)
LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9'’&.\-]*")


@lru_cache(maxsize=1)
def available():
    try:
        from PIL import features
        return bool(features.check("raqm")) and BOLD.exists()
    except Exception:  # noqa: BLE001 - Pillow missing or broken
        return False


@lru_cache(maxsize=128)
def _font(size, latin=False):
    from PIL import ImageFont
    path = LATIN if latin and LATIN.exists() else BOLD
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.RAQM)


def runs(text):
    """Split a line into (piece, is_latin): the Hindi font has no English letters."""
    out, pos = [], 0
    for m in LATIN_RUN.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def line_width(text, size, stroke=0):
    return sum(_font(size, latin).getlength(piece) for piece, latin in runs(text)) + 2 * stroke


def draw_line(draw, x, baseline, text, size, fill, stroke=0, stroke_fill=None):
    for piece, latin in runs(text):
        font = _font(size, latin)
        draw.text((x, baseline), piece, font=font, fill=fill, anchor="ls",
                  stroke_width=stroke, stroke_fill=stroke_fill)
        x += font.getlength(piece)


def wrap(text, size, max_width, stroke=0, max_lines=3):
    """Greedy word wrap; returns lines (None if it needs more than max_lines)."""
    lines, cur = [], ""
    for word in (text or "").split():
        trial = f"{cur} {word}".strip()
        if not cur or line_width(trial, size, stroke) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines if len(lines) <= max_lines else None


def fit(text, max_width, size, min_size, stroke=0, max_lines=3):
    """Largest font size (from size down to min_size) at which the text fits."""
    s = size
    while s > min_size:
        lines = wrap(text, s, max_width, stroke, max_lines)
        if lines:
            return s, lines
        s = int(s * 0.9)
    return min_size, wrap(text, min_size, max_width, stroke, 99) or [text]


def _canvas(W, H):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _metrics(size):
    asc, desc = _font(size).getmetrics()
    return asc, desc


def _lines_height(size, n, gap):
    asc, desc = _metrics(size)
    return n * (asc + desc) + (n - 1) * gap


def _block(draw, lines, size, cx, top, fill, stroke=0, stroke_fill=None, gap=0, fills=None):
    asc, desc = _metrics(size)
    y = top
    for i, line in enumerate(lines):
        w = line_width(line, size, stroke)
        draw_line(draw, cx - w / 2 + stroke, y + asc, line, size,
                  (fills[i % len(fills)] if fills else fill), stroke, stroke_fill)
        y += asc + desc + gap
    return y


def headline(path, W, H, text):
    """On-screen story text over (darkened, moving) footage: white on a dark box with a red
    news-style accent bar."""
    img, d = _canvas(W, H)
    size, lines = fit(text, int(W * 0.78), int(H * 0.075), int(H * 0.042))
    gap = int(H * 0.012)
    th = _lines_height(size, len(lines), gap)
    tw = max(line_width(ln, size) for ln in lines)
    pad_x, pad_y = int(H * 0.035), int(H * 0.028)
    top = (H - th) / 2
    box = (W / 2 - tw / 2 - pad_x, top - pad_y, W / 2 + tw / 2 + pad_x, top + th + pad_y * 1.4)
    d.rounded_rectangle(box, radius=int(H * 0.012), fill=(0, 0, 0, 190))
    d.rectangle((box[0], box[1], box[0] + int(H * 0.012), box[3]), fill=RED)
    _block(d, lines, size, W / 2, top, WHITE, gap=gap)
    img.save(path)
    return path


def title(path, W, H, text):
    """The film's title: huge yellow letters with a thick black outline."""
    img, d = _canvas(W, H)
    stroke = max(2, int(H * 0.009))
    size, lines = fit(text, int(W * 0.86), int(H * 0.115), int(H * 0.06), stroke, 2)
    gap = int(H * 0.01)
    th = _lines_height(size, len(lines), gap)
    top = (H - th) / 2
    d.rectangle((0, top - H * 0.04, W, top + th + H * 0.06), fill=(0, 0, 0, 120))
    _block(d, lines, size, W / 2, top, YELLOW, stroke, (0, 0, 0, 255), gap)
    img.save(path)
    return path


def subtitle_band(H):
    """Height of the strip at the bottom of the frame that holds subtitles."""
    return int(H * 0.26) // 2 * 2


def subtitle(path, W, H, text):
    """Narration subtitle: white, black outline, on a soft dark strip. The picture is only the
    bottom strip of the frame (subtitle_band(H) high); it is laid at the bottom."""
    band = subtitle_band(H)
    img, d = _canvas(W, band)
    if text:
        stroke = max(1, int(H * 0.004))
        size, lines = fit(text, int(W * 0.84), int(H * 0.05), int(H * 0.034), stroke, 2)
        gap = int(H * 0.006)
        th = _lines_height(size, len(lines), gap)
        bottom = band - int(H * 0.07)
        top = bottom - th
        tw = max(line_width(ln, size, stroke) for ln in lines)
        pad = int(H * 0.014)
        d.rounded_rectangle((W / 2 - tw / 2 - pad * 2, top - pad, W / 2 + tw / 2 + pad * 2,
                             bottom + pad * 1.6), radius=pad, fill=(0, 0, 0, 140))
        _block(d, lines, size, W / 2, top, WHITE, stroke, (0, 0, 0, 255), gap)
    img.save(path)
    return path


def thumbnail(frame_png, out_jpg, text, W=1280, H=720):
    """YouTube thumbnail: the hero frame, punchier colours, giant yellow + white Hindi words."""
    from PIL import Image, ImageEnhance
    if frame_png and Path(frame_png).exists():
        base = Image.open(frame_png).convert("RGB")
        scale = max(W / base.width, H / base.height)
        base = base.resize((int(base.width * scale + 0.5), int(base.height * scale + 0.5)))
        left, top = (base.width - W) // 2, (base.height - H) // 2
        base = base.crop((left, top, left + W, top + H))
        base = ImageEnhance.Contrast(ImageEnhance.Color(base).enhance(1.35)).enhance(1.15)
    else:
        base = Image.new("RGB", (W, H), (20, 20, 24))
    img, d = _canvas(W, H)
    stroke, gap = 9, 6
    size, lines = fit(" ".join((text or "").split()), int(W * 0.92), 150, 70, stroke, 2)
    th = _lines_height(size, len(lines), gap)
    top = H - th - 46
    d.rectangle((0, top - 30, W, H), fill=(0, 0, 0, 110))
    _block(d, lines, size, W / 2, top, YELLOW, stroke, (0, 0, 0, 255), gap, fills=[YELLOW, WHITE])
    out = Image.alpha_composite(base.convert("RGBA"), img).convert("RGB")
    out.save(out_jpg, quality=92)
    return out_jpg
