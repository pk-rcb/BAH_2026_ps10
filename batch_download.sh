#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# batch_download.sh  —  Download Landsat data for all cities in cities.csv
#
# RESUME SUPPORT:
#   Set RESUME_FROM to the exact city name you want to restart from.
#   All cities before it will be skipped.
#   Set RESUME_FROM="" to run from the very beginning.
#
#   Example (resume from Ashoknagar):
#     RESUME_FROM="Ashoknagar" bash batch_download.sh
#
# SKIP-IF-EXISTS:
#   Cities whose input/<CityName>/ folder already exists are automatically
#   skipped, even if they appear after RESUME_FROM. Delete the folder to
#   force a re-download for a specific city.
# ─────────────────────────────────────────────────────────────────────────────

if [ -f .env ]; then
  source .env
fi

if [ -z "$EE_PROJECT_ID" ]; then
  echo "Error: EE_PROJECT_ID is not set in .env"
  exit 1
fi
PROJECT="$EE_PROJECT_ID"
BANDS="SR_B2,SR_B3,SR_B4,ST_B10"
START="2023-01-01"
END="2023-12-31"

# Set this to the city name you want to resume from.
# Leave empty ("") to download from the start.
RESUME_FROM="${RESUME_FROM:-Sirohi}"

# Internal flag: once we've hit the resume city, set to 1
RESUME_REACHED=0

# If RESUME_FROM is empty, start immediately
if [ -z "$RESUME_FROM" ]; then
  RESUME_REACHED=1
fi

TOTAL=0
SKIPPED_BEFORE=0
SKIPPED_EXISTS=0
DOWNLOADED=0
FAILED=0

echo "=================================================="
echo "  Batch Download — IR Colorization Dataset"
echo "=================================================="
echo "  Resuming from : ${RESUME_FROM:-<beginning>}"
echo "  Project ID    : $PROJECT"
echo "  Bands         : $BANDS"
echo "  Date range    : $START -> $END"
echo "=================================================="
echo ""

# Read CSV and iterate (skipping header)
tail -n +2 cities.csv | while IFS=, read -r CITY LAT LON
do
  TOTAL=$((TOTAL + 1))

  # Strip Windows-style carriage returns (if CSV was saved on Windows)
  CITY=$(echo "$CITY" | tr -d '\r')
  LAT=$(echo "$LAT"   | tr -d '\r')
  LON=$(echo "$LON"   | tr -d '\r')

  # ── Resume logic ──────────────────────────────────────────────────────────
  if [ "$RESUME_REACHED" -eq 0 ]; then
    if [ "$CITY" = "$RESUME_FROM" ]; then
      RESUME_REACHED=1
      echo "[RESUME] Starting from: $CITY"
    else
      SKIPPED_BEFORE=$((SKIPPED_BEFORE + 1))
      echo "[SKIP-BEFORE] $CITY"
      continue
    fi
  fi

  # ── Skip-if-exists logic ──────────────────────────────────────────────────
  OUTPUT_DIR="./input/$CITY"
  if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    SKIPPED_EXISTS=$((SKIPPED_EXISTS + 1))
    echo "[SKIP-EXISTS] $CITY  (folder already populated: $OUTPUT_DIR)"
    continue
  fi

  # ── Download ───────────────────────────────────────────────────────────────
  echo ""
  echo "[DOWNLOAD] $CITY  (lat=$LAT, lon=$LON)"

  python scripts/download.py "$CITY" $LAT $LON \
    --bands "$BANDS" \
    --start_date "$START" \
    --end_date "$END" \
    --output_path "$OUTPUT_DIR" \
    --ee_project_id "$PROJECT"

  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    DOWNLOADED=$((DOWNLOADED + 1))
    echo "[OK] $CITY downloaded successfully."
  else
    FAILED=$((FAILED + 1))
    echo "[FAIL] $CITY failed with exit code $EXIT_CODE. Continuing..."
  fi

done

echo ""
echo "=================================================="
echo "  Download Summary"
echo "=================================================="
echo "  Downloaded    : $DOWNLOADED"
echo "  Already exist : $SKIPPED_EXISTS"
echo "  Skipped before: $SKIPPED_BEFORE"
echo "  Failed        : $FAILED"
echo "=================================================="