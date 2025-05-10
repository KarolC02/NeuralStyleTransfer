#!/bin/bash

# Define all combinations
combinations=(
  "content/me_hk.jpg styles/van_gogh.jpg"
  "content/me.jpg styles/background.jpg"
)

# Make sure script fails fast on missing file
set -e

# Check files exist
for combo in "${combinations[@]}"; do
  read -r content style <<< "$combo"
  if [[ ! -f "$content" ]]; then
    echo "Missing content image: $content" >&2; exit 1
  fi
  if [[ ! -f "$style" ]]; then
    echo "Missing style image: $style" >&2; exit 1
  fi
done

# Run experiments
for combo in "${combinations[@]}"; do
  read -r content style <<< "$combo"
  content_base=$(basename "$content" .png)
  style_base=$(basename "$style" .jpg)

  # a) From noise, beta = 1,10,100,1000
  for beta in 1 10 100 1000; do
    out_dir="generated_images_from_noise_${content_base}_${style_base}_beta=$(printf "%.0e" $beta)"
    echo "Running noise init: $content vs $style with beta=$beta"
    python nst.py --content "$content" --style "$style" \
      --output-dir "$out_dir" \
      --init noise --beta $beta --save-every 500
  done

  # b) From content, beta = 1,100,10000,1000000 and 9
  for beta in 1 100 10000 1000000 9; do
    out_dir="generated_images_from_content_${content_base}_${style_base}_beta=$(printf "%.0e" $beta)"
    echo "Running content init: $content vs $style with beta=$beta"
    python nst.py --content "$content" --style "$style" \
      --output-dir "$out_dir" \
      --init content --beta $beta --save-every 500
  done
done
