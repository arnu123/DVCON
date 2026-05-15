#!/bin/bash
# TADOP — Download COCO val2017 images (1 GB, one-time)
# Run from inside the Stage2/ project folder.

set -e

echo "=== COCO val2017 download ==="
echo "Size: ~1 GB compressed, ~6 GB uncompressed (5000 images)"
echo

if [ -d "coco_images/val2017" ]; then
    n_files=$(ls coco_images/val2017/*.jpg 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_files" -ge 4900 ]; then
        echo "Already have $n_files images in coco_images/val2017/, skipping."
        exit 0
    fi
fi

mkdir -p coco_images
cd coco_images

if [ ! -f "val2017.zip" ]; then
    echo "Downloading val2017.zip from cocodataset.org..."
    curl -L -o val2017.zip http://images.cocodataset.org/zips/val2017.zip
else
    echo "val2017.zip already present, skipping download."
fi

echo "Unzipping..."
unzip -q val2017.zip

echo "Verifying..."
n_files=$(ls val2017/*.jpg | wc -l | tr -d ' ')
echo "Found $n_files images (expected ~5000)."

if [ "$n_files" -lt 4900 ]; then
    echo "WARNING: image count looks low. Re-download may be needed."
    exit 1
fi

echo
echo "Optionally remove the zip to save 1 GB:"
echo "    rm coco_images/val2017.zip"
echo
echo "Done. Images are in coco_images/val2017/"
