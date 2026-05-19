#!/bin/bash
set -e

# Real dashcam driving videos from Pexels (free to use / Pexels License)
# These are actual forward-facing dashboard camera footage showing roads,
# lane markings, and traffic from the driver's perspective.

URLS=(
    # Pexels 31901299: "Busy Urban Traffic View From a Car" - real dashcam
    "https://videos.pexels.com/video-files/31901299/13528793_640_360_30fps.mp4"
    "https://videos.pexels.com/video-files/31901299/13528793_1280_720_30fps.mp4"
    "https://videos.pexels.com/video-files/31901299/13528793_1920_1080_30fps.mp4"
    # Pexels 2053100: Highway driving dashcam
    "https://videos.pexels.com/video-files/2053100/2053100-sd_640_360_30fps.mp4"
    "https://videos.pexels.com/video-files/2053100/2053100-hd_1280_720_25fps.mp4"
)

for url in "${URLS[@]}"; do
    echo "Trying: $url"
    if curl -L -f --connect-timeout 15 --max-time 180 -o sample.mp4 "$url" 2>/dev/null; then
        size=$(stat -c%s sample.mp4 2>/dev/null || stat -f%z sample.mp4 2>/dev/null)
        if [ "$size" -gt 100000 ]; then
            echo "SUCCESS: Downloaded sample.mp4 ($size bytes)"
            file sample.mp4 || true
            exit 0
        else
            echo "File too small ($size bytes), trying next..."
            rm -f sample.mp4
        fi
    else
        echo "Download failed, trying next..."
    fi
done

# Fallback: generate a synthetic test video with ffmpeg
echo "All downloads failed. Generating synthetic test video..."
ffmpeg -y -f lavfi \
    -i "color=c=0x333333:s=640x360:d=30,drawtext=text='YOLOPv2 Test - Replace with dashcam video':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2" \
    -c:v libx264 -pix_fmt yuv420p \
    sample.mp4
echo "Generated synthetic sample.mp4"
ls -lh sample.mp4
