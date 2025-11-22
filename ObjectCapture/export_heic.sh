#!/usr/bin/env bash
set -euo pipefail

# Path to Xcode's build output (Debug by default, use Release if you built with that configuration)
TOOL_PATH="$HOME/Library/Developer/Xcode/DerivedData/ObjectCaptureReconstruction-*/Build/Products/Debug/HEICDepthTool"

# Expand the glob to find the actual path
TOOL_PATH=$(echo $TOOL_PATH)

if [[ ! -x "$TOOL_PATH" ]]; then
  echo "[ERROR] HEICDepthTool not found at: $TOOL_PATH"
  echo "[INFO] Build the HEICDepthTool target in Xcode first (⌘B)"
  echo "[INFO] Or use: xcodebuild -project ObjectCaptureReconstruction.xcodeproj -scheme HEICDepthTool -configuration Debug"
  exit 1
fi

COLOR="/Users/maxlyu/Documents/nutritionverse-3d-dataset-estimation/test_objectcapture/images/view_000_img.png"
DEPTH="/Users/maxlyu/Documents/nutritionverse-3d-dataset-estimation/test_objectcapture/depth/view_000_depth_mm.png"
OUT="/Users/maxlyu/Documents/nutritionverse-3d-dataset-estimation/test_objectcapture/combined_new.heic"

# Depth is stored in millimeters → 1 unit = 0.001 meters
SCALE="0.001"

echo "[INFO] Using tool: $TOOL_PATH"

"$TOOL_PATH" \
  --color "$COLOR" \
  --depth-png16 "$DEPTH" \
  --depth-scale "$SCALE" \
  --out "$OUT"

echo "[INFO] Wrote: $OUT"
echo "[INFO] Verify with: exiftool -a -G1 -s -Depth* '$OUT'"
