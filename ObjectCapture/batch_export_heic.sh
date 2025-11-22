#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Batch export HEIC files with embedded depth from color + depth PNG pairs
# ============================================================================

# Parse command line arguments
IMAGE_DIR="${1:-}"
DEPTH_DIR="${2:-}"
OUTPUT_DIR="${3:-}"
DEPTH_SCALE="${4:-0.001}"

if [[ -z "$IMAGE_DIR" ]] || [[ -z "$DEPTH_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
  echo "Usage: $0 <image_dir> <depth_dir> <output_dir> [depth_scale]"
  echo ""
  echo "Example:"
  echo "  $0 ./images ./depth ./heic 0.001"
  echo ""
  echo "Arguments:"
  echo "  image_dir    - Directory containing color PNG images"
  echo "  depth_dir    - Directory containing 16-bit depth PNG files"
  echo "  output_dir   - Output directory for HEIC files (will be created)"
  echo "  depth_scale  - Scale factor (default: 0.001 for mm→meters)"
  echo ""
  echo "File naming convention:"
  echo "  Color: view_000_img.png, view_001_img.png, ..."
  echo "  Depth: view_000_depth_mm.png, view_001_depth_mm.png, ..."
  echo "  Output: view_000.heic, view_001.heic, ..."
  exit 1
fi

# Find the HEICDepthTool executable
TOOL_PATH="$HOME/Library/Developer/Xcode/DerivedData/ObjectCaptureReconstruction-*/Build/Products/Debug/HEICDepthTool"
TOOL_PATH=$(echo $TOOL_PATH)

if [[ ! -x "$TOOL_PATH" ]]; then
  echo "[ERROR] HEICDepthTool not found at: $TOOL_PATH"
  echo "[INFO] Build the HEICDepthTool target in Xcode first (⌘B)"
  exit 1
fi

# Validate input directories
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "[ERROR] Image directory not found: $IMAGE_DIR"
  exit 1
fi

if [[ ! -d "$DEPTH_DIR" ]]; then
  echo "[ERROR] Depth directory not found: $DEPTH_DIR"
  exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Batch HEIC Export with Depth"
echo "=========================================="
echo "[INFO] Tool:        $TOOL_PATH"
echo "[INFO] Images:      $IMAGE_DIR"
echo "[INFO] Depth:       $DEPTH_DIR"
echo "[INFO] Output:      $OUTPUT_DIR"
echo "[INFO] Depth scale: $DEPTH_SCALE"
echo "=========================================="
echo ""

# Counter for progress
SUCCESS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0

# Find all color images (adjust pattern as needed)
# This script assumes naming like: view_000_img.png, view_001_img.png, etc.
shopt -s nullglob
for COLOR_IMG in "$IMAGE_DIR"/*_img.png "$IMAGE_DIR"/*.png; do
  if [[ ! -f "$COLOR_IMG" ]]; then
    continue
  fi
  
  TOTAL_COUNT=$((TOTAL_COUNT + 1))
  
  # Extract base name
  BASENAME=$(basename "$COLOR_IMG")
  
  # Try to match corresponding depth file
  # Pattern 1: view_000_img.png → view_000_depth_mm.png
  DEPTH_IMG=""
  if [[ "$BASENAME" =~ (.*)_img\.png$ ]]; then
    PREFIX="${BASH_REMATCH[1]}"
    DEPTH_IMG="$DEPTH_DIR/${PREFIX}_depth_mm.png"
    OUTPUT_NAME="${PREFIX}.heic"
  # Pattern 2: view_000.png → view_000_depth_mm.png
  elif [[ "$BASENAME" =~ (.*)\.png$ ]]; then
    PREFIX="${BASH_REMATCH[1]}"
    DEPTH_IMG="$DEPTH_DIR/${PREFIX}_depth_mm.png"
    OUTPUT_NAME="${PREFIX}.heic"
  else
    echo "[WARN] Could not parse filename: $BASENAME"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi
  
  # Check if depth file exists
  if [[ ! -f "$DEPTH_IMG" ]]; then
    echo "[WARN] Depth file not found for $BASENAME: $DEPTH_IMG"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi
  
  OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_NAME"
  
  echo "[INFO] Processing: $BASENAME"
  echo "       Color:  $COLOR_IMG"
  echo "       Depth:  $DEPTH_IMG"
  echo "       Output: $OUTPUT_FILE"
  
  # Run the tool
  if "$TOOL_PATH" \
      --color "$COLOR_IMG" \
      --depth-png16 "$DEPTH_IMG" \
      --depth-scale "$DEPTH_SCALE" \
      --out "$OUTPUT_FILE"; then
    echo "       ✓ Success"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "       ✗ Failed"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  echo ""
done

echo "=========================================="
echo "Batch Export Complete"
echo "=========================================="
echo "[INFO] Total:   $TOTAL_COUNT"
echo "[INFO] Success: $SUCCESS_COUNT"
echo "[INFO] Failed:  $FAIL_COUNT"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo ""

if [[ $SUCCESS_COUNT -gt 0 ]]; then
  echo "[INFO] Verify a file with:"
  FIRST_HEIC=$(find "$OUTPUT_DIR" -name "*.heic" -type f | head -n 1)
  if [[ -n "$FIRST_HEIC" ]]; then
    echo "  exiftool -a -G1 -s -Depth* '$FIRST_HEIC'"
  fi
fi
