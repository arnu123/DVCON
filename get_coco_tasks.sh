#!/bin/bash
# TADOP — Download COCO-Tasks annotations
# Run this once on your machine from inside the tadop/ project folder.
# Takes ~2 minutes. No COCO images needed yet.

set -e  # stop on any error

echo "=== Step 1: Install git-lfs ==="
if command -v git-lfs &> /dev/null; then
    echo "git-lfs already installed, skipping."
else
    # Ubuntu / WSL
    if command -v apt &> /dev/null; then
        sudo apt install -y git-lfs
    # Mac with Homebrew
    elif command -v brew &> /dev/null; then
        brew install git-lfs
    else
        echo "ERROR: Can't detect package manager. Install git-lfs manually from https://git-lfs.com"
        exit 1
    fi
fi

git lfs install

echo ""
echo "=== Step 2: Clone COCO-Tasks annotations ==="
if [ -d "coco-tasks" ]; then
    echo "coco-tasks/ folder already exists, skipping clone."
else
    git clone -b cvpr2019 --depth 1 https://github.com/coco-tasks/dataset.git coco-tasks
fi

echo ""
echo "=== Step 3: Verify files ==="
EXPECTED=28  # 14 tasks x 2 splits (train + val)
COUNT=$(ls coco-tasks/annotations/task_*.json 2>/dev/null | wc -l)
echo "Found $COUNT annotation files (expected $EXPECTED)"
if [ "$COUNT" -ne "$EXPECTED" ]; then
    echo "WARNING: File count mismatch. git-lfs may not have downloaded the actual files."
    echo "Check: git lfs pull  (inside the coco-tasks/ directory)"
else
    echo "All good!"
fi

echo ""
echo "=== Step 4: Build the real LUT ==="
python sw/lut/build_affordance_lut.py \
    --coco-tasks-dir ./coco-tasks/annotations \
    --out-dir ./data/lut

echo ""
echo "Done. Real LUT is in data/lut/"