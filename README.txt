Fly Vision  v0.1
================

Live fruit-fly + wing detection on Apple Silicon.  Tracks flies in real
time with SAM 3.1 (MLX) at ~20-30 FPS at 224 px on M-series hardware,
and runs a higher-resolution wing-measurement pass on demand.


QUICK START
-----------
From the repo root:

    cd code
    python run_realtime_tracker.py --video ../data/input/your_fly_video.mp4

That's it for the live tracker.  In the window:

    SPACE       pause / resume
    a           run wing analysis on the current paused frame
    s           save annotated PNG + append CSV row(s) for each fly
    c           calibrate px/mm (click 2 points on a known reference)
    e           edit the detection prompts (Tab between fly / wing)
    m           cycle view: focus (ROI) -> full -> chips off
    9 / 0       shrink / grow the focus-mode ROI
    -/=         decrease / increase score threshold
    <- / ->     step frames while paused
    click       move the focus-mode ROI center
    q           quit

Saved outputs land in `data/output/`:

    data/output/<session>_<NNNN>.png
    data/output/data/<session>.csv

Each CSV row is one fly with left/right wing measurements (px and mm)
plus a `calibrated` yes/no column so you can tell apart rows recorded
on the default 120 px/mm vs rows recorded after calibration.


ALSO INCLUDED: still-image analyzer
-----------------------------------
For a single fly photo:

    cd code
    python wingdetector.py --image ../data/input/your_fly_photo.png

Saves an annotated PNG to `data/output/` with PCA dimension arrows,
mm^2 label pills (PIL/TTF for proper superscripts), and a scale bar.
This is the same measurement pipeline the realtime tracker uses on
analyze, so the numbers match.


SETUP
-----
Tested on macOS (Apple Silicon).

1. Python 3.10 environment.

2. Install MLX VLM (provides SAM 3.1 weights loading + inference):

       pip install mlx-vlm opencv-python pillow matplotlib numpy

3. Download SAM 3.1 weights (MLX format) and place them under
   `code/model/sam3.1-bf16/`.  The model folder is gitignored due to
   size.  See `code/model/COPY LOCAL MODEL HERE.txt` for notes;
   `mlx-community/sam3.1-bf16` on Hugging Face is the working build.

   The default `--model` flag for both scripts is `model/sam3.1-bf16`
   relative to the `code/` directory; override with `--model /abs/path`.

4. Drop your videos into `data/input/` (or pass `--video <path>`).
   Webcam works too: omit `--video` and it'll open device 0.


PERFORMANCE NOTES
-----------------
- Live tracker FPS scales with `--resolution`.  Defaults to 224 px
  (~20-30 FPS) which keeps inference fast at the cost of mask
  smoothness.  At `--resolution 1008` SAM's tracker propagation kicks
  in, masks get smoother across frames, but per-frame cost rises.
- The [a] analyze pass is independent of live tracker resolution —
  defaults to 1008 px (SAM's training resolution) regardless of live
  `--resolution`, so wing measurements always come from the sharpest
  available masks.


REPO LAYOUT
-----------
    code/
        run_realtime_tracker.py     live tracker (this is the main app)
        wingdetector.py             fly + wing helpers + still-image analyzer
        model/                      drop SAM 3.1 weights here (gitignored)
    data/
        input/                      your videos go here (gitignored)
        output/                     saved PNGs + CSV land here
    examples/
        example_input_data/         small reference inputs
        example_output/             reference outputs
