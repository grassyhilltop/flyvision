# wingdetector.py — fly + wing detection library and still-image analyzer.
#
# This file is the shared "fly biology + drawing" layer used by both:
#   - the still-image analyzer below (`python wingdetector.py --image …`)
#   - the live tracker in `run_realtime_tracker.py`
#
# Capabilities:
#   - SAM 3.1 (MLX) predictor wrapper for fly + wing detection.
#   - Per-wing measurement via PCA on the mask pixels:
#       major / minor axis lengths (px and mm) + area (px and mm²).
#   - cv2 drawing primitives reused by the live tracker (alpha fill,
#     contour, halo dim-line, extension ticks, ROI scale bar, HUD pills).
#   - PIL drawing primitives used by the still-image analyzer (rounded
#     pills with TTF text + the mm² superscript glyph + alpha pixels).
#   - CSV logger (one row per fly, padded comma format that lines up in
#     macOS Quick Look and parses cleanly into Excel).
#
# Calibration: D. melanogaster body length ~2.5 mm.  The default
# `--px-per-mm 120` is a placeholder; the live tracker's [c] key
# refines it from a clicked reference, and every saved CSV row carries
# a `calibrated` yes/no flag so downstream analysis can tell apart
# default-measured rows from properly-calibrated ones.

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mlx_vlm.utils import load_model
from mlx_vlm.models.sam3_1.generate import Sam3Predictor, predict_multi
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor

FLY_PROMPT = "fruit fly"
WING_PROMPT = "wing"

DEFAULT_PX_PER_MM = 120.0  # placeholder until a reference object exists

FONT_CANDIDATES = [
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

# Monospace font — used for dimension labels so numeric columns align.
MONO_FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
]


def _pick(candidates, size):
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def pick_font(size: int) -> ImageFont.ImageFont:
    return _pick(FONT_CANDIDATES, size)


def pick_mono_font(size: int) -> ImageFont.ImageFont:
    return _pick(MONO_FONT_CANDIDATES, size)


# ---------- SAM helpers ----------

def load_predictor(local_path: Path, score_threshold: float) -> Sam3Predictor:
    model = load_model(local_path)
    processor = Sam31Processor.from_pretrained(local_path)
    return Sam3Predictor(model, processor, score_threshold=score_threshold)


def ensure_mask(mask_raw: np.ndarray, W: int, H: int) -> np.ndarray:
    if mask_raw.shape != (H, W):
        m = np.array(
            Image.fromarray(mask_raw.astype(np.float32)).resize((W, H))
        ) > 0
    else:
        m = mask_raw > 0
    return m.astype(bool)


def mask_centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def mask_top_point(mask: np.ndarray):
    """Topmost (min-y) mask pixel — good anchor for a confidence chip."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    i = int(np.argmin(ys))
    return np.array([int(xs[i]), int(ys[i])], dtype=float)


def mask_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ---------- measurement ----------

def pca_wing_stats(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    if not (np.all(np.isfinite(eigvals)) and np.all(np.isfinite(eigvecs))):
        return None
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    major_vec, minor_vec = eigvecs[:, 0], eigvecs[:, 1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        major_proj = centered @ major_vec
        minor_proj = centered @ minor_vec
    if not (np.all(np.isfinite(major_proj)) and np.all(np.isfinite(minor_proj))):
        return None
    return {
        "center": center,
        "major_vec": major_vec,
        "minor_vec": minor_vec,
        "major_axis": (
            center + major_vec * major_proj.min(),
            center + major_vec * major_proj.max(),
        ),
        "minor_axis": (
            center + minor_vec * minor_proj.min(),
            center + minor_vec * minor_proj.max(),
        ),
        "major_length_px": float(major_proj.max() - major_proj.min()),
        "minor_length_px": float(minor_proj.max() - minor_proj.min()),
        "area_px": int(mask.sum()),
    }


def assign_wings_to_flies(fly_masks, wing_masks) -> list[int]:
    if not fly_masks:
        return [-1] * len(wing_masks)
    # Skip any fly with a degenerate (all-zero) mask — mask_centroid returns
    # None for those, which breaks np.stack.  Keep the original fly index so
    # the returned assignments stay aligned with fly_masks.
    entries = [(i, mask_centroid(m)) for i, m in enumerate(fly_masks)]
    entries = [(i, c) for i, c in entries if c is not None]
    if not entries:
        return [-1] * len(wing_masks)
    fly_idx = [e[0] for e in entries]
    fly_cs = np.stack([e[1] for e in entries])
    out = []
    for wm in wing_masks:
        wc = mask_centroid(wm)
        if wc is None:
            out.append(-1)
            continue
        d = np.linalg.norm(fly_cs - wc, axis=1)
        out.append(fly_idx[int(np.argmin(d))])
    return out


# ---------- drawing primitives ----------

def blend_fill(img_bgr, mask, color_bgr, alpha):
    overlay = img_bgr.copy()
    overlay[mask] = color_bgr
    return cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)


def draw_contour(img_bgr, mask, color_bgr, thickness=2):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(img_bgr, contours, -1, color_bgr, thickness, lineType=cv2.LINE_AA)


def _filled_triangle(canvas, tip, direction, color, size):
    u = np.asarray(direction, dtype=np.float64)
    n = np.array([-u[1], u[0]])
    tip = np.asarray(tip, dtype=np.float64)
    back = tip - u * size
    left = back + n * (size * 0.45)
    right = back - n * (size * 0.45)
    pts = np.array([tip, left, right], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color, lineType=cv2.LINE_AA)


def dim_line_with_halo(canvas, a, b, color, thickness, head_size):
    """Dim line parallel to a feature with filled arrowheads + thin white halo."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a
    L = np.linalg.norm(v)
    if L < 1e-6:
        return
    u = v / L
    ai, bi = tuple(a.astype(int)), tuple(b.astype(int))
    cv2.line(canvas, ai, bi, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.line(canvas, ai, bi, color, thickness, cv2.LINE_AA)
    _filled_triangle(canvas, a, -u, (255, 255, 255), head_size + 1)
    _filled_triangle(canvas, b, u, (255, 255, 255), head_size + 1)
    _filled_triangle(canvas, a, -u, color, head_size)
    _filled_triangle(canvas, b, u, color, head_size)


def extension_tick(canvas, inner, outer, color, thickness, gap=3):
    """Short perpendicular tick at the alignment point (gap near the feature)."""
    inner = np.asarray(inner, dtype=np.float64)
    outer = np.asarray(outer, dtype=np.float64)
    v = outer - inner
    L = np.linalg.norm(v)
    if L < 1e-6:
        return
    u = v / L
    start = inner + u * gap
    end = outer + u * 3
    si, ei = tuple(start.astype(int)), tuple(end.astype(int))
    cv2.line(canvas, si, ei, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    cv2.line(canvas, si, ei, color, thickness, cv2.LINE_AA)


def composite_layer(canvas, layer, alpha):
    """Blend `layer` onto `canvas` only at pixels where `layer` has drawn
    content, so untouched areas of the canvas are preserved."""
    mask = layer.any(axis=2)
    if not mask.any():
        return canvas
    blended = cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0)
    out = canvas.copy()
    out[mask] = blended[mask]
    return out


def outward_normal(axis_vec, wing_center, reference):
    n = np.array([-axis_vec[1], axis_vec[0]], dtype=np.float64)
    if np.dot(wing_center - reference, n) < 0:
        n = -n
    return n


def downward_normal(axis_vec):
    """Perpendicular to `axis_vec`, biased toward +y (below in image space)."""
    n = np.array([-axis_vec[1], axis_vec[0]], dtype=np.float64)
    if n[1] < 0 or (abs(n[1]) < 1e-6 and n[0] < 0):
        n = -n
    return n


def place_centered_pill(pil_img, text, cx, top_y, font, accent_rgb,
                        W, H, pad=8, radius=8, border=2):
    """Draw a pill with its top edge at `top_y`, horizontally centred on `cx`.

    Clamps horizontally + vertically so the pill stays inside the frame.
    Returns (rect, pill_width, pill_height).
    """
    draw = ImageDraw.Draw(pil_img, "RGBA")
    tw, th = measure_text_bbox(draw, text, font)
    pill_w = tw + 2 * pad
    pill_h = th + 2 * pad
    text_x = cx - tw / 2
    text_y = top_y + pad
    # clamp horizontally
    text_x = max(2 + pad, min(W - 2 - pad - tw, text_x))
    # clamp vertically
    text_y = max(2 + pad, min(H - 2 - pad - th, text_y))
    rect = draw_label_pill(pil_img, text, (text_x, text_y), font, accent_rgb,
                           pad=pad, radius=radius, border=border)
    return rect, pill_w, pill_h


def place_left_pill(pil_img, text, left_x, top_y, font, accent_rgb,
                    W, H, pad=8, radius=8, border=2):
    """Pill with its LEFT edge at `left_x`, top edge at `top_y`."""
    draw = ImageDraw.Draw(pil_img, "RGBA")
    tw, th = measure_text_bbox(draw, text, font)
    text_x = left_x + pad
    text_y = top_y + pad
    text_x = max(2 + pad, min(W - 2 - pad - tw, text_x))
    text_y = max(2 + pad, min(H - 2 - pad - th, text_y))
    rect = draw_label_pill(pil_img, text, (text_x, text_y), font, accent_rgb,
                           pad=pad, radius=radius, border=border)
    return rect


# ---------- label + scale bar ----------

def measure_text_bbox(draw, text, font):
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_label_pill(pil_img, text, xy, font, accent_rgb,
                    pad=8, radius=8, border=2, bg_alpha=120):
    # Draw on a transparent overlay and alpha_composite onto pil_img so
    # the semi-transparent background actually blends with the photo.
    # ImageDraw on an RGBA base REPLACES pixels (including the destination
    # alpha), so after .convert("RGB") the bg becomes fully opaque black —
    # that's the "opacity missing" bug in earlier revisions.
    tw, th = measure_text_bbox(ImageDraw.Draw(pil_img), text, font)
    x, y = xy
    rect = [x - pad, y - pad, x + tw + pad, y + th + pad]
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    bg = (0, 0, 0, bg_alpha)
    outline = (*accent_rgb, 255)
    if hasattr(odraw, "rounded_rectangle"):
        odraw.rounded_rectangle(rect, radius=radius, fill=bg,
                                outline=outline, width=border)
    else:
        odraw.rectangle(rect, fill=bg, outline=outline, width=border)
    odraw.multiline_text((x, y), text, font=font,
                         fill=(255, 255, 255, 255), spacing=3)
    pil_img.alpha_composite(overlay)
    return rect


def rect_overlaps_occ(rect, occ, W, H):
    x0, y0, x1, y1 = rect
    x0c = max(0, int(x0)); y0c = max(0, int(y0))
    x1c = min(W, int(x1)); y1c = min(H, int(y1))
    if x1c <= x0c or y1c <= y0c:
        return False
    return bool(occ[y0c:y1c, x0c:x1c].any())


def rect_overlaps_any(rect, boxes):
    x0, y0, x1, y1 = rect
    for b in boxes:
        bx0, by0, bx1, by1 = b
        if not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1):
            return True
    return False


def place_label(pil_img, text, anchor, direction, font, accent_rgb, occ,
                placed_boxes, W, H, pad=8, radius=8, border=2,
                avoid_occ=True, max_tries=18):
    """Try to place the label at `anchor`; if it collides with masks or other
    labels, nudge along `direction` until clear or we give up."""
    draw = ImageDraw.Draw(pil_img, "RGBA")
    tw, th = measure_text_bbox(draw, text, font)
    ax, ay = float(anchor[0]), float(anchor[1])
    dx, dy = float(direction[0]), float(direction[1])
    dn = (dx * dx + dy * dy) ** 0.5
    if dn < 1e-6:
        dx, dy = 0.0, -1.0
    else:
        dx, dy = dx / dn, dy / dn
    step = max(10, int(round(H / 80)))

    for _ in range(max_tries):
        rect = (ax - pad, ay - pad, ax + tw + pad, ay + th + pad)
        inside = (rect[0] >= 2 and rect[1] >= 2
                  and rect[2] <= W - 2 and rect[3] <= H - 2)
        occ_ok = (not avoid_occ) or (not rect_overlaps_occ(rect, occ, W, H))
        if inside and occ_ok and not rect_overlaps_any(rect, placed_boxes):
            break
        ax += dx * step
        ay += dy * step

    ax = max(2 + pad, min(W - tw - 2 - pad, ax))
    ay = max(2 + pad, min(H - th - 2 - pad, ay))
    rect = draw_label_pill(pil_img, text, (ax, ay), font, accent_rgb,
                           pad=pad, radius=radius, border=border)
    placed_boxes.append(tuple(rect))


def draw_scale_bar(pil_img, W, H, px_per_mm, bar_mm=1.0, margin=24):
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = pick_font(max(16, int(round(H / 55.0))))
    sub_font = pick_font(max(12, int(round(H / 85.0))))

    bar_px = int(round(bar_mm * px_per_mm))
    pad = 14
    label = f"{bar_mm:g} mm"
    sub = f"assumed {px_per_mm:g} px/mm"
    lw, lh = measure_text_bbox(draw, label, font)
    sw, sh = measure_text_bbox(draw, sub, sub_font)
    inner_w = max(bar_px, lw, sw)
    box_w = inner_w + 2 * pad
    box_h = lh + sh + 40

    bx2 = W - margin
    bx1 = bx2 - box_w
    by1 = margin
    by2 = by1 + box_h
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=10,
                               fill=(0, 0, 0, 190))
    else:
        draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, 190))

    y_bar = by1 + pad + 4
    x_bar1 = bx1 + (box_w - bar_px) // 2
    x_bar2 = x_bar1 + bar_px
    draw.line([(x_bar1, y_bar), (x_bar2, y_bar)], fill=(255, 255, 255, 255), width=4)
    tick = 8
    draw.line([(x_bar1, y_bar - tick), (x_bar1, y_bar + tick)],
              fill=(255, 255, 255, 255), width=4)
    draw.line([(x_bar2, y_bar - tick), (x_bar2, y_bar + tick)],
              fill=(255, 255, 255, 255), width=4)

    draw.text((bx1 + (box_w - lw) / 2, y_bar + 10),
              label, font=font, fill=(255, 255, 255, 255))
    draw.text((bx1 + (box_w - sw) / 2, y_bar + 10 + lh + 4),
              sub, font=sub_font, fill=(220, 220, 220, 255))


# ---------- realtime HUD helpers (cv2-based, fast) ----------
#
# These are used by run_realtime_tracker.py — they live here so both files
# render in the same visual language.  Distinct from the PIL-based
# `place_*_pill` / `draw_scale_bar` above, which are slower but produce
# crisper TTF text + the mm² glyph for static screenshots.

def fill_rect_alpha(canvas, x0, y0, x1, y1, color, alpha):
    """Blend a solid color into canvas[y0:y1, x0:x1] at the given alpha."""
    H, W = canvas.shape[:2]
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(W, int(x1)); y1 = min(H, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def hud_text(img, text, xy, scale=0.6, color=(255, 255, 255), thick=1):
    """Black-haloed white text — readable over any background."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def centered_pill(img, text, cx, cy, scale, thick, accent_bgr,
                  pad_x=4, pad_y=2, bg_alpha=0.55):
    """Pill (rounded box + text) horizontally centered on `cx`."""
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x0 = cx - tw // 2 - pad_x
    y0 = cy - th - pad_y
    x1 = cx + tw // 2 + pad_x
    y1 = cy + bl + pad_y
    fill_rect_alpha(img, x0, y0, x1, y1, (0, 0, 0), bg_alpha)
    cv2.rectangle(img, (x0, y0), (x1, y1), accent_bgr, 1, cv2.LINE_AA)
    cv2.putText(img, text, (cx - tw // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                cv2.LINE_AA)


def left_pill(img, text, left_x, cy, scale, thick, accent_bgr,
              pad_x=4, pad_y=2, bg_alpha=0.55):
    """Pill with its LEFT edge at `left_x` and baseline near `cy`.

    Used to stack the wing-confidence chip flush-left with the wing
    measurement box, rather than centring it on the wing centroid —
    so the chip + box read as one stacked unit.
    """
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x0 = left_x
    y0 = cy - th - pad_y
    x1 = left_x + tw + 2 * pad_x
    y1 = cy + bl + pad_y
    fill_rect_alpha(img, x0, y0, x1, y1, (0, 0, 0), bg_alpha)
    cv2.rectangle(img, (x0, y0), (x1, y1), accent_bgr, 1, cv2.LINE_AA)
    cv2.putText(img, text, (left_x + pad_x, cy),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                cv2.LINE_AA)


def labeled_field(bar, label, value, x, y, active=False, cursor=False):
    """Render `label: [value]` starting at (x,y); return x after the field.

    The value sits in a small bordered box so the user can see they're
    in an editable field.  `active=True` brightens the border; `cursor=True`
    appends an underscore caret.
    """
    scale = 0.55
    thick = 1
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(bar, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (180, 180, 180), thick, cv2.LINE_AA)
    x_box = x + lw + 8
    display = value + ("_" if cursor else "")
    (vw, vh), _ = cv2.getTextSize(display if display else " ",
                                  cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = 6
    bx0 = x_box
    by0 = y - vh - pad
    bx1 = x_box + vw + 2 * pad
    by1 = y + pad
    border_col = (80, 200, 255) if active else (110, 110, 120)
    bg_col = (48, 48, 58) if active else (36, 36, 42)
    cv2.rectangle(bar, (bx0, by0), (bx1, by1), bg_col, -1)
    cv2.rectangle(bar, (bx0, by0), (bx1, by1), border_col, 1, cv2.LINE_AA)
    cv2.putText(bar, display, (bx0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240),
                thick, cv2.LINE_AA)
    return bx1


def draw_realtime_scale_bar(canvas, px_per_mm, bar_mm=1.0, margin=16):
    """Compact OpenCV scale bar in the top-right.  Bar length = bar_mm × px/mm
    in *image* pixels; the caller is expected to have scaled px/mm to display
    coords if it's drawing on a downscaled canvas (so 1 mm in the source maps
    to the right number of screen pixels)."""
    H, W = canvas.shape[:2]
    bar_px = int(round(bar_mm * px_per_mm))
    bar_px = max(10, min(bar_px, W // 3))

    label = f"{bar_mm:g} mm"
    sub = f"{px_per_mm:g} px/mm"
    lscale = max(0.45, H / 1700.0)
    sscale = max(0.35, H / 2200.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, sscale, 1)
    inner_w = max(bar_px, tw, sw)
    pad = max(10, int(H / 120))
    box_w = inner_w + 2 * pad
    box_h = 14 + th + sh + 22

    bx1 = W - margin
    bx0 = bx1 - box_w
    by0 = margin
    by1 = by0 + box_h
    if bx0 < margin:
        return
    fill_rect_alpha(canvas, bx0, by0, bx1, by1, (0, 0, 0), 0.55)
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (200, 200, 200), 1,
                  cv2.LINE_AA)

    bar_y = by0 + 14
    bar_x1 = bx0 + (box_w - bar_px) // 2
    bar_x2 = bar_x1 + bar_px
    cv2.line(canvas, (bar_x1, bar_y), (bar_x2, bar_y),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (bar_x1, bar_y - 4), (bar_x1, bar_y + 4),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (bar_x2, bar_y - 4), (bar_x2, bar_y + 4),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (bx0 + (box_w - tw) // 2, bar_y + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, lscale, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, sub, (bx0 + (box_w - sw) // 2,
                              bar_y + th + sh + 14),
                cv2.FONT_HERSHEY_SIMPLEX, sscale, (220, 220, 220), 1,
                cv2.LINE_AA)


def _format_mmss(seconds):
    if seconds is None:
        return "--:--"
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------- CSV logging ----------
#
# Padded-comma format: each value is right-padded with spaces to a fixed
# per-column width before a comma is written.  This gives:
#   - macOS Quick Look (monospace, no tab interpretation): commas line up
#     vertically across rows, so headers and data form a visual grid.
#   - Excel: parses commas normally; numeric cells auto-strip whitespace.
#     String cells keep the padding but Excel left-aligns strings anyway,
#     so the visible result is fine.
#
# Column order: short numerics first (so the alignment doesn't break when
# Quick Look hits a long string field), free-form strings last.

CSV_COLUMNS = [
    "fly_id", "fly_score",
    "L_wing_score", "L_wing_L_px", "L_wing_W_px", "L_wing_A_px",
    "L_wing_L_mm", "L_wing_W_mm", "L_wing_A_mm2",
    "R_wing_score", "R_wing_L_px", "R_wing_W_px", "R_wing_A_px",
    "R_wing_L_mm", "R_wing_W_mm", "R_wing_A_mm2",
    "px_per_mm", "calibrated", "mode", "video_time_s",
    "fly_prompt", "wing_prompt",
    "saved_at", "session", "image_file", "source",
]

CSV_COL_WIDTHS = {
    "fly_id":        6,
    "fly_score":     9,
    "L_wing_score": 12,
    "L_wing_L_px":  11,
    "L_wing_W_px":  11,
    "L_wing_A_px":  11,
    "L_wing_L_mm": 11,
    "L_wing_W_mm": 11,
    "L_wing_A_mm2": 12,
    "R_wing_score": 12,
    "R_wing_L_px":  11,
    "R_wing_W_px":  11,
    "R_wing_A_px":  11,
    "R_wing_L_mm": 11,
    "R_wing_W_mm": 11,
    "R_wing_A_mm2": 12,
    "px_per_mm":     9,
    "calibrated":   10,   # 'yes' / 'no' — whether user ran [c] this session
    "mode":          9,   # 'no-labels'
    "video_time_s": 12,
    "fly_prompt":   24,
    "wing_prompt":  24,
    "saved_at":     19,   # 'YYYY-MM-DD HH:MM:SS'
    "session":      32,
    "image_file":   40,
    "source":       40,
}


def _pad_csv(value, width):
    """Right-pad `value`'s string form with spaces to `width`.  Empty fields
    (missing wing side, or None) become `width` spaces so the comma still
    lands in the right column.  Values longer than `width` are written in
    full — better to break alignment for one cell than to truncate data."""
    s = "" if value is None or value == "" else str(value)
    if len(s) < width:
        s = s + " " * (width - len(s))
    return s


class CsvLogger:
    """Append-only writer.  Writes the header row on first append; later
    appends are pure rows."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = (self.path.exists()
                              and self.path.stat().st_size > 0)

    def append(self, rows):
        if not rows:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f, delimiter=",")
            if not self._wrote_header:
                w.writerow([_pad_csv(c, CSV_COL_WIDTHS[c])
                            for c in CSV_COLUMNS])
                self._wrote_header = True
            for r in rows:
                w.writerow([_pad_csv(r.get(c, ""), CSV_COL_WIDTHS[c])
                            for c in CSV_COLUMNS])


def _wing_row_fields(stats, score, px_per_mm):
    """Pack wing PCA stats into the dict shape build_fly_rows expects."""
    if stats is None:
        return {}
    L_px = int(round(stats["major_length_px"]))
    W_px = int(round(stats["minor_length_px"]))
    A_px = int(stats["area_px"])
    return {
        "score": round(float(score), 4),
        "L_px": L_px, "W_px": W_px, "A_px": A_px,
        "L_mm": round(L_px / px_per_mm, 4),
        "W_mm": round(W_px / px_per_mm, 4),
        "A_mm2": round(A_px / (px_per_mm ** 2), 6),
    }


def build_fly_rows(result, fly_prompt, wing_prompt, W, H, mode, roi,
                   px_per_mm, base_fly_id, context):
    """Derive one-row-per-fly CSV dicts from a prediction result.

    `context` carries the non-measurement fields (session, saved_at,
    fly_prompt, wing_prompt, calibrated, …).  `base_fly_id` is the first
    fly_id to assign; subsequent flies get base_fly_id+1, +2, etc.

    In focus mode, only the ROI-anchored fly survives (matching the
    annotator).  Returns (rows, n_flies_assigned) so callers can decide
    whether to advance their global fly_id counter.
    """
    if result is None:
        return [], 0

    fly_masks, fly_scores = [], []
    wing_masks, wing_scores = [], []
    for mask_raw, score, label in zip(result.masks, result.scores, result.labels):
        m = ensure_mask(mask_raw, W, H)
        if label == fly_prompt:
            fly_masks.append(m); fly_scores.append(float(score))
        elif label == wing_prompt:
            wing_masks.append(m); wing_scores.append(float(score))

    wing_to_fly = assign_wings_to_flies(fly_masks, wing_masks)

    if mode == "focus" and roi is not None:
        rx, ry, rr = roi
        best_i, best_d = -1, float("inf")
        for i, m in enumerate(fly_masks):
            c = mask_centroid(m)
            if c is None:
                continue
            d = float(np.hypot(c[0] - rx, c[1] - ry))
            if d < rr and d < best_d:
                best_i, best_d = i, d
        if best_i < 0:
            return [], 0
        keep = [i for i, f in enumerate(wing_to_fly) if f == best_i]
        wing_masks = [wing_masks[i] for i in keep]
        wing_scores = [wing_scores[i] for i in keep]
        fly_masks = [fly_masks[best_i]]
        fly_scores = [fly_scores[best_i]]
        wing_to_fly = [0] * len(wing_masks)

    rows = []
    for i, fm in enumerate(fly_masks):
        fc = mask_centroid(fm)
        if fc is None:
            continue
        fly_cx = float(fc[0])
        left = right = None
        for w_idx, wm in enumerate(wing_masks):
            if wing_to_fly[w_idx] != i:
                continue
            wc = mask_centroid(wm)
            if wc is None:
                continue
            stats = pca_wing_stats(wm)
            fields = _wing_row_fields(stats, wing_scores[w_idx], px_per_mm)
            if not fields:
                continue
            # Wing side is decided by centroid-x relative to the fly body
            # centroid: smaller x = Left.  When only one wing is detected
            # the other side stays blank; both blanks are valid cells.
            if float(wc[0]) < fly_cx:
                if left is None:
                    left = fields
                elif right is None:
                    right = fields
            else:
                if right is None:
                    right = fields
                elif left is None:
                    left = fields

        row = dict(context)
        row["fly_id"] = base_fly_id + len(rows)
        row["fly_score"] = round(fly_scores[i], 4)
        row["px_per_mm"] = round(px_per_mm, 4)
        for side, fields in (("L_wing", left), ("R_wing", right)):
            if not fields:
                continue
            row[f"{side}_score"] = fields["score"]
            row[f"{side}_L_px"] = fields["L_px"]
            row[f"{side}_W_px"] = fields["W_px"]
            row[f"{side}_A_px"] = fields["A_px"]
            row[f"{side}_L_mm"] = fields["L_mm"]
            row[f"{side}_W_mm"] = fields["W_mm"]
            row[f"{side}_A_mm2"] = fields["A_mm2"]
        rows.append(row)

    return rows, len(rows)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True,
                    help="path to a fly photo (PNG/JPG)")
    ap.add_argument("--model", default="model/sam3.1-bf16",
                    help="path to local SAM 3.1 (MLX) model directory")
    ap.add_argument("--output-dir", default="../data/output",
                    help="folder for the saved annotated PNG")
    ap.add_argument("--score-threshold", type=float, default=0.3)
    ap.add_argument("--fly-alpha", type=float, default=0.10)
    ap.add_argument("--wing-alpha", type=float, default=0.16)
    ap.add_argument("--px-per-mm", type=float, default=DEFAULT_PX_PER_MM,
                    help="assumed calibration; override when a reference is in-frame")
    args = ap.parse_args()

    predictor = load_predictor(Path(args.model), args.score_threshold)
    pil_image = Image.open(args.image).convert("RGB")
    W, H = pil_image.size

    print(f"Loaded {args.image} ({W}x{H}); running SAM 3.1…")
    result = predict_multi(predictor, pil_image, [FLY_PROMPT, WING_PROMPT])

    fly_masks, fly_scores = [], []
    wing_masks, wing_scores = [], []
    for mask_raw, score, label in zip(result.masks, result.scores, result.labels):
        m = ensure_mask(mask_raw, W, H)
        if label == FLY_PROMPT:
            fly_masks.append(m); fly_scores.append(float(score))
        elif label == WING_PROMPT:
            wing_masks.append(m); wing_scores.append(float(score))

    print(f"  flies: {len(fly_masks)}  wings: {len(wing_masks)}")
    wing_to_fly = assign_wings_to_flies(fly_masks, wing_masks)

    n = max(len(fly_masks), 1)
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n)
    fly_rgb = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]
    fly_bgr = [(b, g, r) for (r, g, b) in fly_rgb]

    canvas = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # faint fly fills + thin outlines
    for i, m in enumerate(fly_masks):
        canvas = blend_fill(canvas, m, fly_bgr[i], args.fly_alpha)
    for i, m in enumerate(fly_masks):
        draw_contour(canvas, m, fly_bgr[i], thickness=1)

    # wing fills + halo + coloured outline
    for w_idx, m in enumerate(wing_masks):
        f_idx = wing_to_fly[w_idx]
        color = fly_bgr[f_idx] if f_idx >= 0 else (255, 255, 255)
        canvas = blend_fill(canvas, m, color, args.wing_alpha)
    for w_idx, m in enumerate(wing_masks):
        f_idx = wing_to_fly[w_idx]
        color = fly_bgr[f_idx] if f_idx >= 0 else (255, 255, 255)
        draw_contour(canvas, m, (255, 255, 255), thickness=4)
        draw_contour(canvas, m, color, thickness=2)

    # --- dimension arrows per wing, BELOW the contour (perp with +y bias) ---
    head_size = max(9, int(round(H / 140.0)))
    dim_thickness = 1
    tick_thickness = 1
    dim_alpha = 0.65  # subtlety — let the photo show through

    # draw dim lines onto a black overlay, then blend onto canvas
    dim_layer = np.zeros_like(canvas)
    wing_dim_info = []  # keep per-wing data for label placement after compositing

    for w_idx, m in enumerate(wing_masks):
        stats = pca_wing_stats(m)
        if stats is None:
            wing_dim_info.append(None)
            continue
        f_idx = wing_to_fly[w_idx]
        out_n = downward_normal(stats["major_vec"])
        A, B = stats["major_axis"]

        dim_offset = (
            stats["minor_length_px"] / 2.0
            + head_size * 1.4
            + max(10.0, H / 80.0)
        )
        A_dim = A + out_n * dim_offset
        B_dim = B + out_n * dim_offset
        color = fly_bgr[f_idx] if f_idx >= 0 else (40, 40, 40)

        extension_tick(dim_layer, A, A_dim, color, tick_thickness)
        extension_tick(dim_layer, B, B_dim, color, tick_thickness)
        dim_line_with_halo(dim_layer, A_dim, B_dim, color, dim_thickness, head_size)
        wing_dim_info.append(stats)

    canvas = composite_layer(canvas, dim_layer, dim_alpha)

    # PIL for text + scale bar
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
    label_font = pick_mono_font(max(14, int(round(H / 56.0))))
    chip_font = pick_mono_font(max(13, int(round(H / 66.0))))
    pad_label, pad_chip = 8, 5
    gap = max(6, int(round(H / 220.0)))
    measure_draw = ImageDraw.Draw(pil_canvas, "RGBA")

    # --- Wings: dim label + chip, centred on wing centroid x.
    # Default layout: chip on top, dim label below chip, arrows still below the wing.
    # Flip to below the wing if there isn't room above.
    for w_idx, m in enumerate(wing_masks):
        stats = wing_dim_info[w_idx]
        if stats is None:
            continue
        f_idx = wing_to_fly[w_idx]
        accent_rgb = fly_rgb[f_idx] if f_idx >= 0 else (255, 255, 255)

        wc = mask_centroid(m)
        bb = mask_bbox(m)
        if wc is None or bb is None:
            continue
        xmin, ymin, xmax, ymax = bb
        cx = int(round(wc[0]))

        L_mm = stats["major_length_px"] / args.px_per_mm
        W_mm = stats["minor_length_px"] / args.px_per_mm
        A_mm2 = stats["area_px"] / (args.px_per_mm ** 2)
        L_px = int(round(stats["major_length_px"]))
        W_px = int(round(stats["minor_length_px"]))
        A_px = int(stats["area_px"])
        px_w = max(len(str(L_px)), len(str(W_px)), len(str(A_px)))
        dim_text = (
            f"L  ~{L_mm:>5.2f} mm    ({L_px:>{px_w}d} px)\n"
            f"W  ~{W_mm:>5.2f} mm    ({W_px:>{px_w}d} px)\n"
            f"A  ~{A_mm2:>5.2f} mm²   ({A_px:>{px_w}d} px)"
        )
        chip_text = f"wing {wing_scores[w_idx] * 100:.0f}%"

        dim_tw, dim_th = measure_text_bbox(measure_draw, dim_text, label_font)
        chip_tw, chip_th = measure_text_bbox(measure_draw, chip_text, chip_font)
        dim_ph = dim_th + 2 * pad_label
        chip_ph = chip_th + 2 * pad_chip

        # stack height needed above the wing
        stack_h = chip_ph + gap + dim_ph + gap
        if ymin - stack_h >= 2:
            # Place above: chip on top, dim just above wing
            dim_top = ymin - gap - dim_ph
            chip_top = dim_top - gap - chip_ph
        else:
            # Flip: chip above dim, both below wing
            dim_top = ymax + gap
            chip_top = dim_top + dim_ph + gap

        dim_rect, _, _ = place_centered_pill(
            pil_canvas, dim_text, cx, dim_top, label_font,
            accent_rgb, W, H, pad=pad_label, radius=8, border=2,
        )
        # Left-align the wing chip to the dim pill's left edge so the two
        # stacked elements line up vertically — reduces visual jitter.
        place_left_pill(pil_canvas, chip_text, dim_rect[0], chip_top,
                        chip_font, accent_rgb, W, H,
                        pad=pad_chip, radius=6, border=1)

    # --- Fly chip: centred on fly centroid x, below the bbox by default ---
    for i, m in enumerate(fly_masks):
        c = mask_centroid(m)
        bb = mask_bbox(m)
        if c is None or bb is None:
            continue
        xmin, ymin, xmax, ymax = bb
        cx = int(round(c[0]))

        chip_text = f"fruit fly #{i + 1} {fly_scores[i] * 100:.0f}%"
        chip_tw, chip_th = measure_text_bbox(measure_draw, chip_text, chip_font)
        chip_ph = chip_th + 2 * pad_chip

        chip_top = ymax + gap
        if chip_top + chip_ph > H - 2:
            chip_top = max(2, ymin - gap - chip_ph)

        place_centered_pill(pil_canvas, chip_text, cx, chip_top, chip_font,
                            fly_rgb[i], W, H, pad=pad_chip, radius=6, border=1)

    draw_scale_bar(pil_canvas, W, H, px_per_mm=args.px_per_mm, bar_mm=1.0)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = outdir / f"wingdetection_v3_{stem}_{ts}.png"
    pil_canvas.convert("RGB").save(outpath)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
