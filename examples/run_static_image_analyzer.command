#!/bin/bash
#
# Double-click in Finder to run the still-image wing analyzer on the
# bundled example fly photo.  Output PNG (with PCA dimension arrows +
# mm² label pills + scale bar) lands in data/output/.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
cd code
echo "Running wingdetector on example image..."
python wingdetector.py \
    --image ../examples/example_input_data/testimage1.jpg \
    "$@"

echo
echo "Output saved to data/output/.  Press any key to exit Terminal..."
read -n 1
