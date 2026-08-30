"""Generate the project icon and social preview from one set of geometry constants.

    python scripts/make_icon.py

The mark is the product in one glyph: a tile of cells, and one cell that landed
outside it. That is `1025-contiguous-float32` — the minimal failing case the
shrinker reports for a kernel that masks against `(n // BLOCK) * BLOCK` instead
of `n`. The tile is fine. The element past its edge is the bug.

Colours track the mkdocs Material palette (primary black, accent deep orange)
so the docs site, the README and the favicon are one system.

SVG is the source of truth for anything that scales. The PNGs are rendered from
the same constants below rather than from the SVG, so the two cannot drift
without this file changing.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Geometry, in a 64x64 viewBox ──────────────────────────────────────────
VIEW = 64.0
CELL = 9.0
GAP = 2.5
TILE_X, TILE_Y = 7.0, 16.0
ROWS = COLS = 3
BOUNDARY_X = 43.0
STRAY_X, STRAY_Y = 48.0, 27.5
RADIUS = 13.0        # background corner radius
CELL_RADIUS = 1.6

# ── Colour ────────────────────────────────────────────────────────────────
INK = "#0F1115"      # background, near-black to match `primary: black`
CELL_FG = "#E9EDF2"  # the tile: what the kernel got right
ACCENT = "#FF5722"   # Material deep orange: the element past the edge
EDGE = "#39404D"     # the tile boundary itself

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"


def _cells() -> list[tuple[float, float]]:
    return [
        (TILE_X + c * (CELL + GAP), TILE_Y + r * (CELL + GAP))
        for r in range(ROWS)
        for c in range(COLS)
    ]


def svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
        f'width="64" height="64" role="img" aria-label="shapesandstrides">',
        f'<rect width="64" height="64" rx="{RADIUS}" fill="{INK}"/>',
        f'<line x1="{BOUNDARY_X}" y1="{TILE_Y - 3}" x2="{BOUNDARY_X}" '
        f'y2="{TILE_Y + ROWS * CELL + (ROWS - 1) * GAP + 3}" '
        f'stroke="{EDGE}" stroke-width="1.4" stroke-linecap="round"/>',
    ]
    for x, y in _cells():
        parts.append(
            f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
            f'rx="{CELL_RADIUS}" fill="{CELL_FG}"/>'
        )
    parts.append(
        f'<rect x="{STRAY_X}" y="{STRAY_Y}" width="{CELL}" height="{CELL}" '
        f'rx="{CELL_RADIUS}" fill="{ACCENT}"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _draw_mark(d: ImageDraw.ImageDraw, ox: float, oy: float, k: float,
               background: bool = True) -> None:
    """Draw the mark at scale `k`, origin (ox, oy). k = px per viewBox unit."""
    def box(x, y, w, h):
        return [ox + x * k, oy + y * k, ox + (x + w) * k, oy + (y + h) * k]

    if background:
        d.rounded_rectangle(box(0, 0, VIEW, VIEW), radius=RADIUS * k, fill=INK)
    y0 = TILE_Y - 3
    y1 = TILE_Y + ROWS * CELL + (ROWS - 1) * GAP + 3
    d.line(
        [ox + BOUNDARY_X * k, oy + y0 * k, ox + BOUNDARY_X * k, oy + y1 * k],
        fill=EDGE, width=max(1, round(1.4 * k)),
    )
    for x, y in _cells():
        d.rounded_rectangle(box(x, y, CELL, CELL), radius=CELL_RADIUS * k, fill=CELL_FG)
    d.rounded_rectangle(box(STRAY_X, STRAY_Y, CELL, CELL),
                        radius=CELL_RADIUS * k, fill=ACCENT)


def png(size: int, path: Path, ss: int = 4) -> None:
    img = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    _draw_mark(ImageDraw.Draw(img), 0, 0, (size * ss) / VIEW)
    img.resize((size, size), Image.LANCZOS).save(path)


def _font(name: str, size: int):
    for p in (f"C:/Windows/Fonts/{name}", f"/usr/share/fonts/truetype/dejavu/{name}"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def social(path: Path, ss: int = 2) -> None:
    """1280x640 — what renders when the repo is shared.

    The right panel is the real output of
    `shapesandstrides verify examples/verified_kernels.py`, trimmed to fit. A
    picture of the tool doing its job beats a picture of its name.

    Every string is measured against the space it has before it is drawn, and
    `_fit` raises rather than let text run under the panel. An earlier revision
    silently clipped the wordmark.
    """
    W, H = 1280 * ss, 640 * ss
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)

    M = 88 * ss
    PANEL_L = 672 * ss
    COL = PANEL_L - 40 * ss - M          # usable width of the left column

    def _fit(text, font, what):
        w = d.textlength(text, font=font)
        if w > COL:
            raise ValueError(
                f"social preview: {what} is {w / ss:.0f}px wide, column is "
                f"{COL / ss:.0f}px. Shorten it or drop the font size."
            )
        return text

    # ── left column ──
    mark = 84 * ss
    glyph = Image.new("RGBA", (mark, mark), (0, 0, 0, 0))
    _draw_mark(ImageDraw.Draw(glyph), 0, 0, mark / VIEW, background=False)
    img.paste(glyph, (M, 146 * ss), glyph)

    wf = _font("segoeuib.ttf", 50 * ss)
    d.text((M, 258 * ss), _fit("shapesandstrides", wf, "wordmark"), font=wf, fill=CELL_FG)

    tf = _font("segoeui.ttf", 30 * ss)
    for i, line in enumerate(["Honest correctness and timing",
                              "for Triton kernels."]):
        d.text((M, (332 + i * 42) * ss), _fit(line, tf, "tagline"), font=tf, fill="#9BA6B4")

    d.line([(M, 448 * ss), (M + 108 * ss, 448 * ss)], fill=ACCENT, width=5 * ss)

    sf = _font("segoeui.ttf", 25 * ss)
    for i, line in enumerate(["Every number carries a confidence interval,",
                              "a sample count, and a quality tier."]):
        d.text((M, (484 + i * 36) * ss), _fit(line, sf, "support"), font=sf, fill="#6B7787")

    # ── right panel: real verify output, verbatim ──
    PANEL_R = (1280 - 64) * ss
    d.rounded_rectangle([PANEL_L, 132 * ss, PANEL_R, 508 * ss],
                        radius=14 * ss, fill="#161A21")
    cx = PANEL_L + 30 * ss
    inner = PANEL_R - 30 * ss - cx
    mono = _font("CascadiaMono.ttf", 16 * ss)

    def _mono(text, y, colour):
        w = d.textlength(text, font=mono)
        if w > inner:
            raise ValueError(
                f"social preview: panel line {text!r} is {w / ss:.0f}px, "
                f"panel is {inner / ss:.0f}px. Kernel names are not abbreviated "
                f"here on purpose — shrink the font, not the truth."
            )
        d.text((cx, y * ss), text, font=mono, fill=colour)

    _mono("$ shapesandstrides verify examples/", 164, "#6B7787")
    rows = [
        ("fused_add                     CORRECT     16/16", CELL_FG),
        ("fused_mul                     CORRECT     16/16", CELL_FG),
        ("fused_add_autotuned           CORRECT     16/16", CELL_FG),
        ("fused_add_drops_tail          INCORRECT    5/16", ACCENT),
        ("fused_add_assumes_contiguous  INCORRECT   15/16", ACCENT),
        ("rowsum                        INCORRECT    0/16", ACCENT),
    ]
    for i, (line, colour) in enumerate(rows):
        _mono(line, 212 + i * 32, colour)
    _mono("6 kernel(s) on device=cuda, 3 failed", 420, "#9BA6B4")
    _mono("minimal case: 1025-contiguous-float32", 456, "#6B7787")

    img.resize((1280, 640), Image.LANCZOS).save(path)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "icon.svg").write_text(svg(), encoding="utf-8")
    for s in (512, 192, 64, 32):
        png(s, ASSETS / f"icon-{s}.png")
    png(32, ASSETS / "favicon.png")
    social(ASSETS / "social-preview.png")
    for f in sorted(ASSETS.iterdir()):
        print(f"{f.relative_to(ASSETS.parent.parent)}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
