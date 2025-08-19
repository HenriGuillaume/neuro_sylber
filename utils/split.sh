#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 input.wav"
  exit 1
fi

INPUT="$1"
BASENAME="${INPUT%.*}"

ffmpeg -i "$INPUT" -f segment -segment_time 60 -c copy "${BASENAME}_%03d.wav"
