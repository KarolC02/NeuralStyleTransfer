#!/bin/bash

# Define all combinations
combinations=(
  "content/me_hk.jpg styles/van_gogh.jpg"
)

# Fail fast on error
set -e

# Check files exist
for combo in "${combinations[@]}"; do
  read -r content style <<< "$combo"
  [[ -f "$content" ]] || { echo "Missing content image: $content" >&2; exit 1; }
  [[ -f "$style" ]] || { echo "Missing style image: $style" >&2; exit 1; }
done

# Run experiments
for combo in "${combinations[@]}"; do
  read -r content style <<< "$combo"
  content_base=$(basename "$content" .jpg)
  style_base=$(basename "$style" .jpg)

  for beta in 100 1000 10000 100000 10000000; do
    for lr in 0.001 0.01 0.1; do
      out_dir="generated_images_from_content_${content_base}_${style_base}_beta=$(printf "%.0e" $beta)_lr=${lr}"
      echo "Running: $content vs $style with beta=$beta lr=$lr"
      python nst.py --content "$content" --style "$style" \
        --output-dir "$out_dir" \
        --init content --beta $beta --lr $lr --epochs 20000 --save-every 1000
    done
  done
done
