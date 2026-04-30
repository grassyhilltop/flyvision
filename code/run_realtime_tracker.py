# run_realtime_tracker.py — Fly Vision Realtime
#
# Single-window OpenCV app that tracks fruit flies live with SAM 3.1
# (MLX) and runs a higher-resolution wing-measurement analysis on demand.
#
# Workflow:
#   - Open a video file (or webcam): live tracker runs at the requested
#     `--resolution` (default 224 px = ~20-30 FPS on Apple Silicon).
#     The fly contour + score chip is drawn per detection; in focus mode
#     a translucent ROI ring picks one fly out of the scene.
#   - Press SPACE to pause.  Press [a] to run analyze: a higher-resolution
#     `predict_multi` pass with the wing prompt as well, drawing PCA
#     major/minor axes + L/W/A measurement boxes per wing.
#   - Press [c] to calibrate: click two points on a known reference,
#     enter mm value → px_per_mm updates and the scale bar reflects it.
#     Every saved CSV row carries a `calibrated` yes/no flag.
#   - Press [s] to save: writes an annotated PNG plus appends one CSV
#     row per fly.  In focus mode each saved fly gets the next
#     session-global fly_id (1, 2, 3, … walking through individuals
#     one at a time); in full mode each save records 1..N scene-local.
#
# Architecture:
#   - Reader thread: reads frames from cv2.VideoCapture, posts to a
#     freshest-frame slot for inference + a FIFO queue for display.
#   - Inference thread: always grabs the freshest frame, runs SAM
#     detection (and tracker propagation if --resolution >= 1008).
#   - Display thread (main): pulls from FIFO, composes the annotated
#     frame + control bar, handles keyboard + mouse.
#
# All fly/wing-specific helpers (mask handling, PCA measurement,
# drawing primitives, CSV) live in `wingdetector.py` so the still-image
# analyzer there shares the same visual + data conventions.

import argparse
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_vlm.generate import wired_limit
from mlx_vlm.models.sam3.generate import (
    DetectionResult,
    Sam3Predictor,
    SimpleTracker,
    _filter_by_regions,
)
from mlx_vlm.models.sam3_1.generate import (
    _detect_with_backbone,
    _get_backbone_features,
    _init_tracker_memory,
    _propagate_tracker,
    predict_multi,
)
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
from mlx_vlm.utils import get_model_path, load_model

# All fly/wing-specific helpers + CSV machinery live in the wingdetector
# library, which also has its own still-image CLI for analyzing single
# PNGs (`python wingdetector.py --image …`).
from wingdetector import (
    FLY_PROMPT,
    WING_PROMPT,
    CsvLogger,
    blend_fill,
    build_fly_rows,
    centered_pill,
    composite_layer,
    dim_line_with_halo,
    downward_normal,
    draw_contour,
    ensure_mask,
    extension_tick,
    fill_rect_alpha,
    hud_text,
    labeled_field,
    left_pill,
    mask_bbox,
    mask_centroid,
    pca_wing_stats,
    _format_mmss,
)


VERSION = "0.1"
WINDOW_TITLE = "Fly Vision Realtime"
BAR_H = 60   # 2 rows: prompts/badge/metrics on top, keys + thr/res/mode below

MODES = ["focus", "full", "no-labels"]
MODE_DESCRIPTIONS = {
    "focus": "focus (ROI)",
    "full": "full",
    "no-labels": "chips off",
}

STATUS_COLORS = {  # BGR
    "DETECT":    (60, 200, 255),
    "TRACK":     (90, 220, 90),
    "PAUSED":    (140, 140, 150),
    "ANALYZING": (255, 180, 60),
    "ANALYZED":  (255, 220, 100),
    "INIT":      (160, 160, 160),
}

KEY_LEFT = {2, 81, 63234, 65361, 2424832}
KEY_RIGHT = {3, 83, 63235, 65363, 2555904}


def status_for(analyze_state, paused, live_mode):
    if analyze_state == "analyzing":
        return "ANALYZING"
    if analyze_state == "showing":
        return "ANALYZED"
    if paused:
        return "PAUSED"
    if live_mode == "detect":
        return "DETECT"
    if live_mode == "track":
        return "TRACK"
    return "INIT"


# ---------------------------------------------------------------------------
# Bar rendering
# ---------------------------------------------------------------------------

def render_control_bar(W_total, fly_prompt, wing_prompt, mode_label, status,
                       fps, infer_ms, n_obj, threshold, resolution,
                       editing=False, edit_field=0, edit_buffer=("", ""),
                       calib_mode=None, calib_buffer=""):
    """Two-row bar.

    Row 1: prompt fields (left)            [BADGE]  N.N FPS (M ms)  obj K  (right)
    Row 2: keys / contextual hint (left)              Thresh X%  Res Ypx  Mode: …  (right)

    The status badge sits at the leftmost edge of the right metadata cluster,
    so all numeric/state info groups together cognitively.
    """
    bar = np.full((BAR_H, W_total, 3), (26, 26, 32), dtype=np.uint8)
    row1_y = 22
    row2_y = BAR_H - 14

    # ----- Row 1 left: prompt fields (or calibration mm-entry) -----
    if calib_mode == "mm":
        labeled_field(bar, "Reference length (mm):", calib_buffer,
                      16, row1_y, active=True, cursor=True)
    else:
        fly_disp = edit_buffer[0] if editing else fly_prompt
        wing_disp = edit_buffer[1] if editing else wing_prompt
        # Field labels describe ROLE (what the system does with this
        # prompt), not the prompt content — so the labels stay stable
        # while users experiment with different prompt strings.
        x_after_fly = labeled_field(
            bar, "Detecting:", fly_disp, 16, row1_y,
            active=(editing and edit_field == 0),
            cursor=(editing and edit_field == 0))
        labeled_field(bar, "Measuring:", wing_disp, x_after_fly + 24, row1_y,
                      active=(editing and edit_field == 1),
                      cursor=(editing and edit_field == 1))

    # ----- Row 1 right: BADGE + metrics in one right-aligned cluster -----
    badge_color = STATUS_COLORS.get(status, STATUS_COLORS["INIT"])
    badge_font = 0.55
    badge_thick = 1
    (bw, bh), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX,
                                  badge_font, badge_thick)
    pad_x_b, pad_y_b = 8, 4
    badge_w = bw + 2 * pad_x_b
    badge_h = bh + 2 * pad_y_b
    nums = f"{fps:.1f} FPS ({infer_ms:.0f} ms)   obj {n_obj}"
    (sw, _), _ = cv2.getTextSize(nums, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    spacing = 24   # gap between badge and FPS text — more breathing room
    cluster_w = badge_w + spacing + sw
    cluster_x = W_total - cluster_w - 16

    badge_x0 = cluster_x
    badge_y0 = max(2, row1_y - bh - pad_y_b + 2)
    badge_x1 = badge_x0 + badge_w
    badge_y1 = badge_y0 + badge_h
    fill_rect_alpha(bar, badge_x0, badge_y0, badge_x1, badge_y1,
                    badge_color, 0.92)
    cv2.putText(bar, status, (badge_x0 + pad_x_b, row1_y),
                cv2.FONT_HERSHEY_SIMPLEX, badge_font, (0, 0, 0),
                badge_thick, cv2.LINE_AA)
    cv2.putText(bar, nums, (badge_x1 + spacing, row1_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
                cv2.LINE_AA)

    # ----- Row 2 left: keys (or contextual hint) -----
    if calib_mode in ("p1", "p2"):
        hint = {
            "p1": "Calibration: click extreme point 1   (Esc cancel)",
            "p2": "Calibration: click extreme point 2   (Esc cancel)",
        }[calib_mode]
        cv2.putText(bar, hint, (16, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 255), 1,
                    cv2.LINE_AA)
    elif calib_mode == "mm":
        hint = "Type reference length in mm   (Enter commits, Esc cancels)"
        cv2.putText(bar, hint, (16, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 255), 1,
                    cv2.LINE_AA)
    elif editing:
        hint = ("Editing prompts:  [Tab] field  [Backspace] del  "
                "[Enter] commit  [Esc] cancel")
        cv2.putText(bar, hint, (16, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 220, 255), 1,
                    cv2.LINE_AA)
    else:
        # Tightened: smaller font, double-space between groups, [<->] arrow.
        keys = ("[a] analyze  [SPACE] pause  [<->] step  [s] save  "
                "[c] calib  [e] edit  [m] mode  [9/0] ROI  "
                "[-/=] thr  [q] quit")
        cv2.putText(bar, keys, (16, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 200), 1,
                    cv2.LINE_AA)

    # ----- Row 2 right: settings.  Always visible. -----
    thr_pct = int(round(threshold * 100))
    right = f"Thresh {thr_pct}%   Res {resolution}px   Mode: {mode_label}"
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(bar, right, (W_total - rw - 16, row2_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1,
                cv2.LINE_AA)

    cv2.line(bar, (0, BAR_H - 1), (W_total, BAR_H - 1), (70, 70, 80), 1,
             cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Mask bucketing + assignment helpers
# ---------------------------------------------------------------------------

def _assign_wings_to_flies(fly_masks, wing_masks):
    if not fly_masks:
        return [-1] * len(wing_masks)
    cs, idxs = [], []
    for i, m in enumerate(fly_masks):
        c = mask_centroid(m)
        if c is None:
            continue
        cs.append(c); idxs.append(i)
    if not cs:
        return [-1] * len(wing_masks)
    cs = np.stack(cs)
    out = []
    for wm in wing_masks:
        wc = mask_centroid(wm)
        if wc is None:
            out.append(-1)
            continue
        d = np.linalg.norm(cs - wc, axis=1)
        out.append(idxs[int(np.argmin(d))])
    return out


# ---------------------------------------------------------------------------
# Annotation pipeline (mostly unchanged from v4; ROI ring now translucent)
# ---------------------------------------------------------------------------

def annotate_v6(frame_bgr, result, fly_prompt, wing_prompt, mode, roi,
                px_per_mm, disp_W, disp_H,
                fly_alpha=0.10, wing_alpha=0.16, dim_alpha=0.65,
                roi_alpha=0.65,
                calib_p1=None, calib_p2=None,
                next_fly_id_for_focus=1):
    H, W = frame_bgr.shape[:2]
    scale_x = disp_W / W
    scale_y = disp_H / H
    iso_scale = (scale_x + scale_y) / 2.0

    # Bucket masks
    fly_masks, fly_scores = [], []
    wing_masks, wing_scores = [], []
    if result is not None:
        for m_raw, s, lab in zip(result.masks, result.scores, result.labels):
            if m_raw is None:
                continue
            m = ensure_mask(np.asarray(m_raw), W, H)
            if not m.any():
                continue
            if lab == wing_prompt:
                wing_masks.append(m); wing_scores.append(float(s))
            else:
                fly_masks.append(m); fly_scores.append(float(s))

    wing_to_fly = _assign_wings_to_flies(fly_masks, wing_masks)

    # Focus filter
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
        if best_i >= 0:
            keep_w = [j for j, f in enumerate(wing_to_fly) if f == best_i]
            wing_masks = [wing_masks[j] for j in keep_w]
            wing_scores = [wing_scores[j] for j in keep_w]
            fly_masks = [fly_masks[best_i]]
            fly_scores = [fly_scores[best_i]]
            wing_to_fly = [0] * len(wing_masks)
        else:
            fly_masks, fly_scores = [], []
            wing_masks, wing_scores = [], []
            wing_to_fly = []

    n = max(len(fly_masks), 1)
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n)
    fly_rgb = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]
    fly_bgr = [(b, g, r) for (r, g, b) in fly_rgb]

    # --- Native: fills + contours ---
    native = frame_bgr.copy()
    for i, m in enumerate(fly_masks):
        native = blend_fill(native, m, fly_bgr[i], fly_alpha)
    for i, m in enumerate(fly_masks):
        draw_contour(native, m, fly_bgr[i], thickness=1)
    for j, m in enumerate(wing_masks):
        f_idx = wing_to_fly[j]
        color = fly_bgr[f_idx] if f_idx >= 0 else (255, 255, 255)
        native = blend_fill(native, m, color, wing_alpha)
    for j, m in enumerate(wing_masks):
        f_idx = wing_to_fly[j]
        color = fly_bgr[f_idx] if f_idx >= 0 else (255, 255, 255)
        draw_contour(native, m, (255, 255, 255), thickness=2)
        draw_contour(native, m, color, thickness=1)

    if (disp_W, disp_H) != (W, H):
        canvas = cv2.resize(native, (disp_W, disp_H),
                            interpolation=cv2.INTER_AREA)
    else:
        canvas = native

    # --- Display: translucent ROI ring + crosshair ---
    if mode == "focus" and roi is not None:
        rx_d = int(round(roi[0] * scale_x))
        ry_d = int(round(roi[1] * scale_y))
        rr_d = int(round(roi[2] * iso_scale))
        roi_layer = np.zeros_like(canvas)
        cv2.circle(roi_layer, (rx_d, ry_d), rr_d, (255, 255, 255), 5,
                   cv2.LINE_AA)
        cv2.circle(roi_layer, (rx_d, ry_d), rr_d, (80, 220, 60), 3,
                   cv2.LINE_AA)
        cv2.drawMarker(roi_layer, (rx_d, ry_d), (255, 255, 255),
                       cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA)
        cv2.drawMarker(roi_layer, (rx_d, ry_d), (80, 220, 60),
                       cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        canvas = composite_layer(canvas, roi_layer, roi_alpha)

    # --- Display: calibration markers (always visible if set) ---
    if calib_p1 is not None:
        p1d = (int(round(calib_p1[0] * scale_x)),
               int(round(calib_p1[1] * scale_y)))
        cv2.drawMarker(canvas, p1d, (0, 255, 255),
                       cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        if calib_p2 is not None:
            p2d = (int(round(calib_p2[0] * scale_x)),
                   int(round(calib_p2[1] * scale_y)))
            cv2.line(canvas, p1d, p2d, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(canvas, p2d, (0, 255, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)

    # --- Display: wing dim lines (anchors scaled from native) ---
    show_dims = (mode != "no-labels") and bool(wing_masks)
    wing_render = []
    if show_dims:
        head_size = max(7, int(round(disp_H / 140.0)))
        dim_layer = np.zeros_like(canvas)
        for j, m in enumerate(wing_masks):
            stats = pca_wing_stats(m)
            if stats is None:
                continue
            f_idx = wing_to_fly[j]
            color = fly_bgr[f_idx] if f_idx >= 0 else (40, 40, 40)
            A_n, B_n = stats["major_axis"]
            out_n = downward_normal(stats["major_vec"])
            dim_offset = (stats["minor_length_px"] / 2.0
                          + head_size / max(iso_scale, 1e-6) * 1.4
                          + max(10.0, H / 80.0))
            A_dn = A_n + out_n * dim_offset
            B_dn = B_n + out_n * dim_offset
            scl = np.array([scale_x, scale_y])
            extension_tick(dim_layer, A_n * scl, A_dn * scl, color, 1)
            extension_tick(dim_layer, B_n * scl, B_dn * scl, color, 1)
            dim_line_with_halo(dim_layer, A_dn * scl, B_dn * scl,
                               color, 1, head_size)
            wing_render.append({"j": j, "stats": stats, "color": color,
                                "score": wing_scores[j]})
        canvas = composite_layer(canvas, dim_layer, dim_alpha)

    # --- Display: chips + scale bar ---
    chip_font = max(0.5, disp_H / 1200.0)
    label_font = max(0.45, disp_H / 1400.0)
    pad_box = 4
    gap = max(4, disp_H // 220)

    if show_dims:
        for wv in wing_render:
            stats = wv["stats"]
            j = wv["j"]
            bb = mask_bbox(wing_masks[j])
            if bb is None:
                continue
            xmin, ymin, xmax, ymax = bb
            cx_d = int(round(stats["center"][0] * scale_x))
            ymin_d = int(round(ymin * scale_y))
            ymax_d = int(round(ymax * scale_y))
            L_mm = stats["major_length_px"] / px_per_mm
            W_mm = stats["minor_length_px"] / px_per_mm
            A_mm2 = stats["area_px"] / (px_per_mm ** 2)
            lines = [
                f"L ~{L_mm:5.2f} mm ({int(round(stats['major_length_px'])):4d} px)",
                f"W ~{W_mm:5.2f} mm ({int(round(stats['minor_length_px'])):4d} px)",
                f"A ~{A_mm2:5.2f} mm^2",
            ]
            sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, label_font, 1)
                     for l in lines]
            line_tw = max(sz[0][0] for sz in sizes)
            line_th = sizes[0][0][1]
            spacing = line_th + 6
            box_w = line_tw + 2 * pad_box
            box_h = spacing * len(lines) + pad_box

            chip_text = f'"{wing_prompt}" {wv["score"] * 100:.0f}%'
            (_, cth), cbl = cv2.getTextSize(chip_text,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            chip_font, 1)
            chip_h = cth + cbl + 4
            stack_h = chip_h + gap + box_h + gap
            if ymin_d - stack_h >= 2:
                dim_top = ymin_d - gap - box_h
                chip_cy = dim_top - gap - cbl - 2
            else:
                dim_top = ymax_d + gap
                chip_cy = dim_top + box_h + gap + cth

            x0 = int(cx_d - box_w // 2)
            x0 = max(2, min(disp_W - box_w - 2, x0))
            y0 = int(dim_top)
            y0 = max(2, min(disp_H - box_h - 2, y0))
            x1, y1 = x0 + box_w, y0 + box_h
            fill_rect_alpha(canvas, x0, y0, x1, y1, (0, 0, 0), 0.55)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), wv["color"], 1,
                          cv2.LINE_AA)
            for i, line in enumerate(lines):
                cv2.putText(canvas, line,
                            (x0 + pad_box,
                             y0 + pad_box + (i + 1) * spacing - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, label_font,
                            (255, 255, 255), 1, cv2.LINE_AA)
            left_pill(canvas, chip_text, x0, int(chip_cy), chip_font, 1,
                      wv["color"])

    if mode != "no-labels":
        for i, m in enumerate(fly_masks):
            c = mask_centroid(m)
            bb = mask_bbox(m)
            if c is None or bb is None:
                continue
            xmin, ymin, xmax, ymax = bb
            cx_d = int(round(c[0] * scale_x))
            ymin_d = int(round(ymin * scale_y))
            ymax_d = int(round(ymax * scale_y))
            # focus-mode label is the next session-global fly_id (the ID
            # this fly will be saved under).  Other modes show the
            # 1-indexed scene position so the user gets a count.
            if mode == "focus":
                fly_label = f"#{next_fly_id_for_focus}"
            else:
                fly_label = f"#{i + 1}"
            chip_text = f'"{fly_prompt}" {fly_label} {fly_scores[i] * 100:.0f}%'
            (_, cth), cbl = cv2.getTextSize(chip_text,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            chip_font, 1)
            chip_cy = ymax_d + gap + cth
            if chip_cy + cbl + 2 > disp_H - 2:
                chip_cy = max(cth + 2, ymin_d - gap - cbl - 2)
            centered_pill(canvas, chip_text, cx_d, int(chip_cy),
                          chip_font, 1, fly_bgr[i])

    # Scale bar
    H_d, W_d = canvas.shape[:2]
    bar_px = max(10, min(W_d // 3,
                          int(round(1.0 * px_per_mm * iso_scale))))
    label = "1 mm"
    sub = f"{px_per_mm:g} px/mm"
    lscale = max(0.42, H_d / 1700.0)
    sscale = max(0.34, H_d / 2200.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, lscale, 1)
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, sscale, 1)
    inner_w = max(bar_px, tw, sw)
    pad = max(8, int(H_d / 130))
    box_w = inner_w + 2 * pad
    box_h = 14 + th + sh + 18
    bx1 = W_d - 14
    bx0 = bx1 - box_w
    by0 = 14
    by1 = by0 + box_h
    if bx0 >= 14:
        fill_rect_alpha(canvas, bx0, by0, bx1, by1, (0, 0, 0), 0.55)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (200, 200, 200), 1,
                      cv2.LINE_AA)
        bar_y = by0 + 12
        bar_x1 = bx0 + (box_w - bar_px) // 2
        bar_x2 = bar_x1 + bar_px
        cv2.line(canvas, (bar_x1, bar_y), (bar_x2, bar_y),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (bar_x1, bar_y - 4), (bar_x1, bar_y + 4),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (bar_x2, bar_y - 4), (bar_x2, bar_y + 4),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (bx0 + (box_w - tw) // 2, bar_y + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, lscale, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, sub, (bx0 + (box_w - sw) // 2,
                                   bar_y + th + sh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, sscale, (220, 220, 220), 1,
                    cv2.LINE_AA)

    return canvas, len(fly_masks), len(wing_masks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def track_video_realtime_v6(
    video_path: str,
    fly_prompt_init: str,
    wing_prompt_init: str,
    model_path: str,
    threshold: float = 0.5,
    nms_thresh: float = 0.5,
    boxes: Optional[str] = None,
    resolution: int = 224,
    analyze_resolution: int = 1008,
    detect_every: int = 1,
    recompute_backbone_every: int = 1,
    update_memory_every: int = 3,
    display_width: Optional[int] = 1280,
    px_per_mm_init: float = 120.0,
    focus_radius_frac: float = 0.30,
    output_dir: str = "../output",
):
    box_array = None
    if boxes is not None:
        box_list = []
        for b in boxes.split(";"):
            coords = [float(x) for x in b.split(",")]
            if len(coords) == 4:
                box_list.append(coords)
        if box_list:
            box_array = np.array(box_list)

    print(f"Loading model: {model_path}")
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam31Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)

    is_camera = str(video_path).isdigit()
    cap = cv2.VideoCapture(int(video_path) if is_camera else video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = (frame_count / fps) if (frame_count and fps > 0) else None

    # Mutable state shared between threads + UI ----------------------------
    prompts_state = {"fly": fly_prompt_init, "wing": wing_prompt_init,
                     "dirty": True}
    threshold_state = {"v": float(threshold)}
    px_per_mm_state = {"v": float(px_per_mm_init)}

    # Session identity for save / CSV
    source_path = (Path(video_path) if not is_camera else
                   Path(f"webcam{video_path}"))
    session_prefix = (datetime.now().strftime("%Y%m%d_%H%M%S")
                      + f"_{source_path.stem}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CsvLogger(data_dir / f"{session_prefix}.csv")
    save_counter = 0
    next_fly_id = 1

    print(f"Video: {fps:.1f} fps, {W}x{H}")
    print(f"Live prompt: {fly_prompt_init!r}, threshold {threshold}, "
          f"resolution {resolution}x{resolution}")
    print(f"Analyze prompts: [{fly_prompt_init!r}, {wing_prompt_init!r}], "
          f"resolution {analyze_resolution}")
    print(f"Optimized: detect every {detect_every}, "
          f"ViT every {recompute_backbone_every}, "
          f"memory update every {update_memory_every}")
    print(f"Output: {out_dir}/  CSV: {csv_logger.path}")

    if display_width and display_width < W:
        disp_W = int(display_width)
        disp_H = int(round(H * disp_W / W))
    else:
        disp_W, disp_H = W, H
    scale_x = disp_W / W
    scale_y = disp_H / H
    print(f"Display: {disp_W}x{disp_H} (native frame {W}x{H})")
    print("Keys: SPACE pause  <-/-> step  a analyze  s save  c calib  "
          "e edit  m mode  9/0 ROI  -/= thr  q quit")

    focus_radius = max(40.0, focus_radius_frac * H)
    roi = [W / 2.0, H / 2.0, focus_radius]
    mode_idx = 0  # focus

    frame_buffer = queue.Queue(maxsize=10)
    lock = threading.Lock()
    empty_result = DetectionResult(
        boxes=np.zeros((0, 4)),
        masks=np.zeros((0, H, W), dtype=np.uint8),
        scores=np.zeros((0,)),
        labels=[],
    )
    latest = {
        "result": empty_result,
        "n_obj": 0,
        "fps": 0.0,
        "infer_ms": 0.0,
        "mode": "init",
    }
    # latest_frame is the single-slot hand-off from reader/SPACE-handler
    # to the inference thread.  `is_snapshot` distinguishes a paused-
    # snapshot push (must be processed with all caches reset) from a
    # streaming push (uses the FPS-optimized cached backbone).
    latest_frame = {"bgr": None, "pos_s": None, "seq": 0,
                    "is_snapshot": False}
    paused_state = {"v": False}
    running = {"active": True}
    # Frame-to-result synchronization counter.  The reader and the
    # SPACE/arrow-step handlers bump `req_seq` when they push a frame;
    # the inference thread records the seq it consumed and writes
    # `done_seq` after it lands the result.  push_snapshot_and_wait
    # blocks until done_seq catches up to the snapshot's req_seq.
    sync = {"req_seq": 0, "done_seq": 0}

    frame_interval = 1.0 / fps if not is_camera else 0

    # --- snapshot sync helper ---
    def push_snapshot_and_wait(frame_bgr, pos_s, timeout=1.0):
        """Run a fresh, frame-aligned detection on `frame_bgr` and block
        until the result is in `latest["result"]`.

        Mechanism: tag the slot push with `is_snapshot=True`.  The
        inference thread sees the tag at consume-time and (a) resets
        ALL cross-frame state — backbone cache, encoder cache, tracker
        state, id tracker — before processing, and (b) leaves the
        caches empty afterwards so the post-pause first play frame is
        also computed fresh.  This sidesteps the race where the in-
        flight inference iteration's end-of-iter cache pre-fetch would
        otherwise overwrite an externally-driven cache reset.

        Pause is conceptually a fresh session: FPS optimizations don't
        apply, every snapshot starts from a clean state.
        """
        with lock:
            sync["req_seq"] += 1
            target_seq = sync["req_seq"]
            latest_frame["bgr"] = frame_bgr.copy()
            latest_frame["pos_s"] = pos_s
            latest_frame["seq"] = target_seq
            latest_frame["is_snapshot"] = True
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with lock:
                if sync["done_seq"] >= target_seq:
                    return True
            time.sleep(0.005)
        return False

    # --- reader ---
    def reader_loop():
        next_frame_time = time.perf_counter()
        while running["active"]:
            if paused_state["v"]:
                time.sleep(0.05)
                continue
            if not is_camera:
                now = time.perf_counter()
                if now < next_frame_time:
                    time.sleep(max(0, next_frame_time - now - 0.001))
                    continue
            ret, frame = cap.read()
            if not ret:
                if is_camera:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                next_frame_time = time.perf_counter()
                continue
            pos_s = (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                     if not is_camera else None)
            # Atomic push: re-check paused INSIDE the lock.  If the user
            # pressed SPACE after we cleared the top-of-loop check but
            # before we got here, the SPACE handler is racing to put a
            # snapshot in the slot — we must not clobber it.  The
            # in-flight cap.read()'d frame gets dropped instead; on
            # un-pause cap will pick up from the right position via the
            # frame_buffer drain (see SPACE handler), so no real frames
            # are lost in the user's perception.
            with lock:
                if paused_state["v"]:
                    continue
                sync["req_seq"] += 1
                latest_frame["bgr"] = frame
                latest_frame["pos_s"] = pos_s
                latest_frame["seq"] = sync["req_seq"]
                latest_frame["is_snapshot"] = False  # streaming push
                # Queue push lives inside the lock too so it's tied to
                # the slot decision (both happen, or neither does).
                if frame_buffer.full():
                    try:
                        frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                frame_buffer.put((frame, pos_s))
            if not is_camera:
                next_frame_time += frame_interval

    # --- inference (live tracker) ---
    def inference_loop():
        backbone_cache = {"features": None}
        encoder_cache = {}
        tracker_state = {"memory_bank": [], "n_objects": 0, "labels": []}
        inference_count = 0
        prop_count = 0
        id_tracker = SimpleTracker()
        local_prompts = [prompts_state["fly"]]

        while running["active"]:
            with lock:
                latest_bgr = latest_frame["bgr"]
                consumed_seq = latest_frame.get("seq", 0)
                # `is_snapshot` is True when push_snapshot_and_wait set it
                # (pause / arrow-step path); False for streaming reader
                # pushes.  We capture it under the lock and clear it so
                # any subsequent push starts fresh.
                is_snapshot = latest_frame.get("is_snapshot", False)
                latest_frame["bgr"] = None
                latest_frame["is_snapshot"] = False
            if latest_bgr is None:
                time.sleep(0.005)
                continue

            # On a snapshot, aggressively wipe ALL cross-frame state.
            # Pause is conceptually a fresh session — none of the FPS
            # optimizations (prefetched backbone features, tracker
            # memory, persistent ID tracker) should leak influence
            # from previous frames.  This is the deterministic
            # mental model the user expects: "press SPACE → process
            # this exact frame from scratch".
            if is_snapshot:
                backbone_cache["features"] = None
                encoder_cache.clear()
                tracker_state["memory_bank"] = []
                tracker_state["n_objects"] = 0
                tracker_state["labels"] = []
                id_tracker = SimpleTracker()
                prop_count = 0

            # Pick up prompt + threshold updates from the UI thread
            local_prompts = [prompts_state["fly"]]
            current_thr = threshold_state["v"]
            predictor.score_threshold = current_thr
            if prompts_state["dirty"]:
                # Prompt change → drop tracker memory; redetect fresh
                tracker_state["memory_bank"] = []
                tracker_state["n_objects"] = 0
                tracker_state["labels"] = []
                backbone_cache["features"] = None
                prompts_state["dirty"] = False

            t0 = time.perf_counter()
            frame_pil = Image.fromarray(cv2.cvtColor(latest_bgr,
                                                    cv2.COLOR_BGR2RGB))
            image_size = frame_pil.size
            inputs = predictor.processor.preprocess_image(frame_pil)
            pixel_values = mx.array(inputs["pixel_values"])

            if backbone_cache["features"] is None:
                backbone_cache["features"] = _get_backbone_features(
                    model, pixel_values)
            backbone_features = backbone_cache["features"]

            can_track = (
                resolution >= 1008
                and tracker_state["memory_bank"]
                and tracker_state["n_objects"] > 0
            )
            need_detect = (
                inference_count % detect_every == 0
                or not tracker_state["memory_bank"]
                or not can_track
            )

            if need_detect:
                encoder_cache.clear()
                result = _detect_with_backbone(
                    predictor, backbone_features, local_prompts,
                    image_size, current_thr, encoder_cache=encoder_cache,
                )
                if box_array is not None and len(result.scores) > 0:
                    result = _filter_by_regions(result, box_array)
                mode = "detect"
                if len(result.scores) > 0 and resolution >= 1008:
                    tracker_state["memory_bank"] = _init_tracker_memory(
                        model, backbone_features, list(result.masks))
                    tracker_state["n_objects"] = len(result.scores)
                    tracker_state["labels"] = (
                        result.labels if result.labels
                        else [local_prompts[0]] * len(result.scores))
                    prop_count = 0
                elif len(result.scores) > 0:
                    tracker_state["labels"] = (
                        result.labels if result.labels
                        else [local_prompts[0]] * len(result.scores))
            else:
                result, updated_bank = _propagate_tracker(
                    model, backbone_features,
                    tracker_state["memory_bank"],
                    tracker_state["n_objects"], image_size,
                )
                result.labels = tracker_state["labels"]
                prop_count += 1
                if prop_count % update_memory_every == 0:
                    tracker_state["memory_bank"] = updated_bank
                mode = "track"

            dt = time.perf_counter() - t0

            # End-of-iter backbone pre-fetch (the upstream FPS
            # optimization that pre-computes features for the NEXT
            # frame using THIS frame's pixels).  Skip on snapshot:
            # we don't want either the snapshot iteration OR the
            # following one to reuse stale features.  The next frame
            # (whether play resumes or another snapshot fires) will
            # find features=None and compute fresh.
            if not is_snapshot:
                next_count = inference_count + 1
                if (next_count % recompute_backbone_every == 0
                        or next_count % detect_every == 0):
                    backbone_cache["features"] = _get_backbone_features(
                        model, pixel_values)
            else:
                backbone_cache["features"] = None

            result = id_tracker.update(result)
            with lock:
                latest["result"] = result
                latest["n_obj"] = len(result.scores)
                latest["fps"] = 1.0 / max(dt, 1e-6)
                latest["infer_ms"] = dt * 1000.0
                latest["mode"] = mode
                # Mark this seq as done.  max() so a stale slow compute
                # never moves done_seq backwards if a fresher one already
                # landed (shouldn't happen given single-threaded inference,
                # but defensive against reordering).
                if consumed_seq > sync["done_seq"]:
                    sync["done_seq"] = consumed_seq
            inference_count += 1

    # --- mouse ---
    mouse_state = {"click": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["click"] = (x, y)

    # --- analyze + edit + calibration state ---
    analyze_state = {"v": "idle"}
    analyze_result = {"r": None}
    editing_state = {"v": False, "field": 0, "buf": ["", ""]}
    calib_state = {"mode": None, "p1": None, "p2": None, "buf": ""}
    # `calibrated` flips True the first time the user commits a [c]
    # calibration in this session.  Persisted into every CSV row so
    # downstream analysis can tell calibrated vs default measurements
    # apart at a glance.
    calibrated_state = {"v": False}

    def clear_analyze(reason=""):
        if analyze_state["v"] != "idle":
            analyze_state["v"] = "idle"
            analyze_result["r"] = None
            if reason:
                print(f"[analyze] cleared ({reason})")

    def run_analyze(frame_bgr):
        orig_size = processor.image_size
        try:
            processor.image_size = analyze_resolution
            predictor.score_threshold = threshold_state["v"]
            pil = Image.fromarray(cv2.cvtColor(frame_bgr,
                                               cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            res = predict_multi(predictor, pil,
                                [prompts_state["fly"], prompts_state["wing"]])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            n_fly = sum(1 for l in res.labels if l == prompts_state["fly"])
            n_wing = sum(1 for l in res.labels if l == prompts_state["wing"])
            print(f"[analyze] done in {dt_ms:.0f} ms — "
                  f"{n_fly} flies, {n_wing} wings")
            analyze_result["r"] = res
            analyze_state["v"] = "showing"
        except Exception as exc:
            print(f"[analyze] error: {exc}")
            analyze_state["v"] = "idle"
            analyze_result["r"] = None
        finally:
            processor.image_size = orig_size

    # --- main loop ---
    with wired_limit(model):
        threading.Thread(target=reader_loop, daemon=True).start()
        threading.Thread(target=inference_loop, daemon=True).start()

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, disp_W, BAR_H + disp_H)
        cv2.setMouseCallback(WINDOW_TITLE, on_mouse)

        last_frame_bgr = None
        last_pos_s = None

        # Annotation cache.  annotate_v6 is the expensive thing in the loop
        # (~30 ms at 1920x1080); when nothing relevant has changed (e.g. user
        # is typing characters into the prompt editor on a paused frame),
        # reuse the previous canvas and only re-render the cheap bar/HUD.
        # This is what makes prompt editing feel instant.
        ann_cache = {"key": None, "canvas": None, "n_fly": 0, "n_wing": 0}

        while True:
            mode = MODES[mode_idx]

            # Frame fetch / hold
            if paused_state["v"]:
                if last_frame_bgr is None:
                    key = cv2.waitKeyEx(50)
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        paused_state["v"] = not paused_state["v"]
                        clear_analyze("unpause")
                    continue
                frame_bgr = last_frame_bgr
                pos_s = last_pos_s
            else:
                try:
                    frame_bgr, pos_s = frame_buffer.get(timeout=0.05)
                    last_frame_bgr = frame_bgr
                    last_pos_s = pos_s
                except queue.Empty:
                    if last_frame_bgr is None:
                        key = cv2.waitKeyEx(1)
                        if key == ord("q"):
                            break
                        if key == ord(" "):
                            paused_state["v"] = not paused_state["v"]
                        continue
                    frame_bgr = last_frame_bgr
                    pos_s = last_pos_s

            if analyze_state["v"] == "showing" and analyze_result["r"] is not None:
                draw_result = analyze_result["r"]
            else:
                with lock:
                    draw_result = latest["result"]

            with lock:
                inf_fps = latest["fps"]
                inf_ms = latest["infer_ms"]
                live_mode = latest["mode"]

            ann_key = (
                id(frame_bgr),
                tuple(roi),
                mode,
                px_per_mm_state["v"],
                calib_state["p1"],
                calib_state["p2"],
                id(draw_result),
                next_fly_id,
                prompts_state["fly"],
                prompts_state["wing"],
            )
            if ann_key == ann_cache["key"] and ann_cache["canvas"] is not None:
                annotated = ann_cache["canvas"].copy()
                n_fly = ann_cache["n_fly"]
                n_wing = ann_cache["n_wing"]
            else:
                annotated, n_fly, n_wing = annotate_v6(
                    frame_bgr, draw_result,
                    prompts_state["fly"], prompts_state["wing"],
                    mode, tuple(roi), px_per_mm_state["v"],
                    disp_W, disp_H,
                    calib_p1=calib_state["p1"],
                    calib_p2=calib_state["p2"],
                    next_fly_id_for_focus=next_fly_id)
                ann_cache["key"] = ann_key
                ann_cache["canvas"] = annotated.copy()
                ann_cache["n_fly"] = n_fly
                ann_cache["n_wing"] = n_wing

            if pos_s is not None:
                hud_text(annotated,
                         f"t={_format_mmss(pos_s)} / "
                         f"{_format_mmss(duration_s)}",
                         (12, 28), scale=0.6)
            if n_wing > 0:
                hud_text(annotated, f"flies {n_fly}   wings {n_wing}",
                         (12, disp_H - 14), scale=0.6)
            else:
                hud_text(annotated, f"flies {n_fly}",
                         (12, disp_H - 14), scale=0.6)
            # Bottom-right version watermark — bumps with VERSION so saved
            # screenshots self-identify which build they came from.
            ver_text = f"(v{VERSION})"
            (vw, _), _ = cv2.getTextSize(ver_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 1)
            hud_text(annotated, ver_text,
                     (disp_W - vw - 12, disp_H - 14), scale=0.5,
                     color=(200, 200, 200))

            status = status_for(analyze_state["v"], paused_state["v"],
                                live_mode)
            bar = render_control_bar(
                disp_W, prompts_state["fly"], prompts_state["wing"],
                MODE_DESCRIPTIONS[mode], status,
                inf_fps, inf_ms, n_fly,
                threshold_state["v"], resolution,
                editing=editing_state["v"],
                edit_field=editing_state["field"],
                edit_buffer=tuple(editing_state["buf"]),
                calib_mode=calib_state["mode"],
                calib_buffer=calib_state["buf"])
            combined = np.vstack([bar, annotated])
            cv2.imshow(WINDOW_TITLE, combined)

            # ---- mouse: ROI move OR calibration point ----
            if mouse_state["click"] is not None:
                mx_px, my_px = mouse_state["click"]
                mouse_state["click"] = None
                if my_px >= BAR_H:
                    nx = mx_px / scale_x
                    ny = (my_px - BAR_H) / scale_y
                    if calib_state["mode"] == "p1":
                        calib_state["p1"] = (nx, ny)
                        calib_state["mode"] = "p2"
                        print(f"[calib] p1 = ({nx:.1f}, {ny:.1f})")
                    elif calib_state["mode"] == "p2":
                        calib_state["p2"] = (nx, ny)
                        calib_state["mode"] = "mm"
                        calib_state["buf"] = ""
                        print(f"[calib] p2 = ({nx:.1f}, {ny:.1f})")
                    elif mode == "focus":
                        roi[0] = float(nx); roi[1] = float(ny)
                        if analyze_state["v"] != "idle":
                            clear_analyze("ROI moved")

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue

            # --------- calibration mm-entry mode ---------
            if calib_state["mode"] == "mm":
                if key in (13, 10):  # Enter
                    try:
                        mm_val = float(calib_state["buf"])
                        if (mm_val <= 0 or calib_state["p1"] is None
                                or calib_state["p2"] is None):
                            raise ValueError
                        dx = calib_state["p2"][0] - calib_state["p1"][0]
                        dy = calib_state["p2"][1] - calib_state["p1"][1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        new_ppm = dist / mm_val
                        px_per_mm_state["v"] = new_ppm
                        calibrated_state["v"] = True
                        print(f"[calib] {dist:.2f} px = {mm_val} mm  →  "
                              f"px_per_mm = {new_ppm:.3f}")
                    except ValueError:
                        print(f"[calib] invalid input "
                              f"{calib_state['buf']!r} — cancelled")
                    calib_state["mode"] = None
                    calib_state["p1"] = None
                    calib_state["p2"] = None
                    calib_state["buf"] = ""
                elif key == 27:  # Esc
                    calib_state["mode"] = None
                    calib_state["p1"] = None
                    calib_state["p2"] = None
                    calib_state["buf"] = ""
                elif key in (8, 127):
                    if calib_state["buf"]:
                        calib_state["buf"] = calib_state["buf"][:-1]
                else:
                    ch = key & 0xFF
                    if 32 <= ch <= 126:
                        calib_state["buf"] += chr(ch)
                continue

            # --------- prompt editing mode ---------
            if editing_state["v"]:
                if key in (13, 10):
                    f = editing_state["buf"][0].strip() or prompts_state["fly"]
                    w = editing_state["buf"][1].strip() or prompts_state["wing"]
                    changed = (f != prompts_state["fly"]
                               or w != prompts_state["wing"])
                    prompts_state["fly"] = f
                    prompts_state["wing"] = w
                    if changed:
                        prompts_state["dirty"] = True
                        clear_analyze("prompts changed")
                        print(f"[prompts] fly={f!r}  wing={w!r}")
                    editing_state["v"] = False
                elif key == 27:
                    editing_state["v"] = False
                elif key == 9:  # Tab
                    editing_state["field"] = 1 - editing_state["field"]
                elif key in (8, 127):
                    f_idx = editing_state["field"]
                    if editing_state["buf"][f_idx]:
                        editing_state["buf"][f_idx] = (
                            editing_state["buf"][f_idx][:-1])
                else:
                    ch = key & 0xFF
                    if 32 <= ch <= 126:
                        f_idx = editing_state["field"]
                        editing_state["buf"][f_idx] = (
                            editing_state["buf"][f_idx] + chr(ch))
                continue

            # --------- normal-mode keys ---------
            # Arrow-step while paused.  Also waits for inference to land
            # the new frame's result so the first render after the step
            # shows correctly-aligned masks (same fix as the SPACE pause).
            if paused_state["v"] and key in KEY_LEFT and not is_camera:
                cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
                ret, fr = cap.read()
                if ret:
                    last_frame_bgr = fr
                    last_pos_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    push_snapshot_and_wait(fr, last_pos_s)
                    clear_analyze("frame step")
                continue
            if paused_state["v"] and key in KEY_RIGHT and not is_camera:
                ret, fr = cap.read()
                if ret:
                    last_frame_bgr = fr
                    last_pos_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    push_snapshot_and_wait(fr, last_pos_s)
                    clear_analyze("frame step")
                continue

            k = key & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                paused_state["v"] = not paused_state["v"]
                if paused_state["v"]:
                    # Pause-snapshot + sync: push the displayed frame to
                    # inference and BLOCK until the result lands for that
                    # exact frame.  Without the wait, the next render
                    # iteration would composite the new (held) frame with
                    # the still-stale result and flash mismatched masks
                    # for ~100 ms before inference catches up.
                    if last_frame_bgr is not None:
                        push_snapshot_and_wait(last_frame_bgr, last_pos_s)
                else:
                    # Drain the FIFO of any frames the reader pushed
                    # before pause — otherwise unpause briefly replays
                    # those old frames before catching up to "now".
                    while not frame_buffer.empty():
                        try:
                            frame_buffer.get_nowait()
                        except queue.Empty:
                            break
                    clear_analyze("unpause")
                print("[paused]" if paused_state["v"] else "[playing]")
            elif k == ord("m"):
                mode_idx = (mode_idx + 1) % len(MODES)
                print(f"mode = {MODES[mode_idx]}")
            elif k == ord("9"):
                roi[2] = max(40.0, roi[2] * 0.9)
            elif k == ord("0"):
                roi[2] = min(0.6 * H, roi[2] * 1.1)
            elif k in (ord("="), ord("+")):
                threshold_state["v"] = min(0.95, round(
                    threshold_state["v"] + 0.05, 2))
                print(f"thresh = {threshold_state['v']:.2f}")
            elif k == ord("-"):
                threshold_state["v"] = max(0.05, round(
                    threshold_state["v"] - 0.05, 2))
                print(f"thresh = {threshold_state['v']:.2f}")
            elif k == ord("e"):
                editing_state["v"] = True
                editing_state["field"] = 0
                editing_state["buf"] = [prompts_state["fly"],
                                        prompts_state["wing"]]
            elif k == ord("c"):
                calib_state["mode"] = "p1"
                calib_state["p1"] = None
                calib_state["p2"] = None
                calib_state["buf"] = ""
                print("[calib] click point 1")
            elif k == ord("a"):
                if last_frame_bgr is None:
                    print("[analyze] no frame to analyze yet")
                    continue
                if not paused_state["v"]:
                    paused_state["v"] = True
                    print("[paused]")
                analyze_state["v"] = "analyzing"
                pre_bar = render_control_bar(
                    disp_W, prompts_state["fly"], prompts_state["wing"],
                    MODE_DESCRIPTIONS[mode], "ANALYZING",
                    inf_fps, inf_ms, n_fly,
                    threshold_state["v"], resolution)
                cv2.imshow(WINDOW_TITLE, np.vstack([pre_bar, annotated]))
                cv2.waitKey(1)
                run_analyze(last_frame_bgr)
            elif k == ord("s"):
                save_counter += 1
                image_name = f"{session_prefix}_{save_counter:04d}.png"
                outpath = out_dir / image_name
                cv2.imwrite(str(outpath), combined)
                context = {
                    "session": session_prefix,
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_file": image_name,
                    "source": str(source_path),
                    "video_time_s": (round(pos_s, 3) if pos_s is not None
                                     else ""),
                    "mode": mode,
                    "fly_prompt": prompts_state["fly"],
                    "wing_prompt": prompts_state["wing"],
                    "calibrated": "yes" if calibrated_state["v"] else "no",
                }
                # Fly-id semantics by mode:
                #   focus      → the one focused fly gets the next session-
                #                global id; advance the global counter.
                #   full / no-labels
                #              → the saved scene gets local ids 1..N (a
                #                snapshot of "all flies in this frame");
                #                global counter UNCHANGED.
                if mode == "focus":
                    base_id = next_fly_id
                else:
                    base_id = 1
                rows, n = build_fly_rows(
                    draw_result, prompts_state["fly"],
                    prompts_state["wing"], W, H,
                    mode, tuple(roi), px_per_mm_state["v"],
                    base_id, context)
                csv_logger.append(rows)
                if mode == "focus":
                    next_fly_id += n
                    print(f"[save] wrote {outpath}  fly_id={base_id} "
                          f"(next focus id={next_fly_id})")
                else:
                    print(f"[save] wrote {outpath}  (+{n} rows, "
                          f"scene-local ids 1..{n})")

        running["active"] = False

    cap.release()
    cv2.destroyAllWindows()
    print("Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Fly Vision Realtime v{VERSION} — live SAM 3.1 tracker "
                    "with on-pause wing measurement"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="video path; omit for webcam (device 0)")
    parser.add_argument("--prompt", default=FLY_PROMPT,
                        help=f"live tracker prompt; default {FLY_PROMPT!r}")
    parser.add_argument("--wing-prompt", default=WING_PROMPT,
                        help=f"wing prompt for [a] analyze; default "
                             f"{WING_PROMPT!r}")
    # Paths default to running this script from the `code/` subdirectory:
    #   model      → ./model/sam3.1-bf16
    #   output     → ../data/output  (CSV under output/data/)
    parser.add_argument("--model",
                        default="model/sam3.1-bf16",
                        help="path to local SAM 3.1 (MLX) model directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--boxes", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=224,
                        help="SAM image_size for live tracking; <1008 disables "
                             "tracker propagation but keeps FPS high")
    parser.add_argument("--analyze-resolution", type=int, default=1008,
                        help="SAM image_size for [a] analyze passes; 1008 is "
                             "the SAM training resolution (sharpest masks)")
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--backbone-every", type=int, default=1)
    parser.add_argument("--memory-every", type=int, default=3)
    parser.add_argument("--display-width", type=int, default=1280,
                        help="display window width; native frame is preserved "
                             "for mask drawing, only resized for display")
    parser.add_argument("--px-per-mm", type=float, default=120.0,
                        help="initial pixel-to-mm calibration; refine with [c]")
    parser.add_argument("--focus-radius-frac", type=float, default=0.30,
                        help="focus-mode ROI radius as a fraction of frame H")
    parser.add_argument("--output-dir", default="../data/output",
                        help="folder for saved PNGs + CSV (data/ subfolder)")
    args = parser.parse_args()

    video = args.video if args.video is not None else "0"

    track_video_realtime_v6(
        video,
        args.prompt,
        args.wing_prompt,
        model_path=args.model,
        threshold=args.threshold,
        nms_thresh=args.nms_thresh,
        boxes=args.boxes,
        resolution=args.resolution,
        analyze_resolution=args.analyze_resolution,
        detect_every=args.detect_every,
        recompute_backbone_every=args.backbone_every,
        update_memory_every=args.memory_every,
        display_width=args.display_width,
        px_per_mm_init=args.px_per_mm,
        focus_radius_frac=args.focus_radius_frac,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
