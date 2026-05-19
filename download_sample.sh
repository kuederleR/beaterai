#!/bin/bash
set -e

# Try multiple sources for a dashcam/driving video with lane lines
# These are all public domain / CC0 licensed driving clips

URLS=(
    # Pexels driving video (direct CDN link) - highway driving with lanes
    "https://videos.pexels.com/video-files/2053100/2053100-sd_640_360_30fps.mp4"
    # Pexels city driving
    "https://videos.pexels.com/video-files/857195/857195-sd_640_360_25fps.mp4"
    # Another Pexels driving clip
    "https://videos.pexels.com/video-files/3048163/3048163-sd_640_360_25fps.mp4"
)

for url in "${URLS[@]}"; do
    echo "Trying: $url"
    if curl -L -f -o sample.mp4 "$url" 2>/dev/null; then
        size=$(stat -c%s sample.mp4 2>/dev/null || stat -f%z sample.mp4 2>/dev/null)
        if [ "$size" -gt 10000 ]; then
            echo "Downloaded sample.mp4 ($size bytes)"
            file sample.mp4
            exit 0
        else
            echo "File too small ($size bytes), trying next..."
            rm -f sample.mp4
        fi
    else
        echo "Download failed, trying next..."
    fi
done

# Fallback: generate a synthetic driving-like video with ffmpeg
echo "All downloads failed. Generating synthetic test video..."
ffmpeg -y -f lavfi \
    -i "color=c=0x333333:s=640x360:d=30,drawtext=text='YOLOPv2 Test - Replace with dashcam video':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2" \
    -c:v libx264 -pix_fmt yuv420p \
    sample.mp4
echo "Generated synthetic sample.mp4"
ls -lh sample.mp4
