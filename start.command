#!/bin/bash
#
# Double-click in Finder to launch Fly Vision Realtime on the bundled
# example video.  Pre-flight: SAM 3.1 model in code/model/sam3.1-bf16/
# and the project's Python deps installed (see README.md).
#
# Pass extra CLI flags by editing this file (or run the python command
# directly from a terminal).

set -euo pipefail
cd "$(dirname "$0")"

cd code
echo "Launching Fly Vision Realtime..."
python run_realtime_tracker.py \
    --video ../examples/example_input_data/testvideo1.mp4 \
    "$@"

echo
echo "Window closed.  Press any key to exit Terminal..."
read -n 1
