#!/bin/bash
set -e

# Try multiple sources for first-person dashcam driving videos
# These are Pexels CDN direct links (public domain / Pexels License)

URLS=(
    # Pexels 5382495: Dashcam POV on expressway with lane lines and traffic
    "https://videos.pexels.com/video-files/5382495/5382495-sd_640_360_25fps.mp4"
    "https://videos.pexels.com/video-files/5382495/5382495-hd_1280_720_25fps.mp4"
    # Pexels 3048163: Car driving on highway
    "https://videos.pexels.com/video-files/3048163/3048163-sd_640_360_25fps.mp4"
    # Pexels 2659475: POV driving on road
    "https://videos.pexels.com/video-files/2659475/2659475-sd_640_360_25fps.mp4"
)

for url in "${URLS[@]}"; do
    echo "Trying: $url"
    if curl -L -f --connect-timeout 15 --max-time 120 -o sample.mp4 "$url" 2>/dev/null; then
        size=$(stat -c%s sample.mp4 2>/dev/null || stat -f%z sample.mp4 2>/dev/null)
        if [ "$size" -gt 50000 ]; then
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

# Fallback: generate a synthetic driving-like video with ffmpeg
echo "All downloads failed. Generating synthetic test video..."
ffmpeg -y -f lavfi \
    -i "color=c=0x333333:s=640x360:d=30,drawtext=text='YOLOPv2 Test - Replace with dashcam video':fontcolor=white:fontsize=20:x=(w-text_w)/2:y=(h-text_h)/2" \
    -c:v libx264 -pix_fmt yuv420p \
    sample.mp4
echo "Generated synthetic sample.mp4"
ls -lh sample.mp4
