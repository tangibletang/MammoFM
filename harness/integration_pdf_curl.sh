#!/usr/bin/env bash
# Integration test: upload 4 views → poll to done → download PDF → validate bytes.
# Usage:
#   bash harness/integration_pdf_curl.sh
#   SERVER_URL=http://localhost:8000 bash harness/integration_pdf_curl.sh
set -euo pipefail

SERVER="${SERVER_URL:-http://localhost:8000}"
SAMPLE_DIR="${MAMMOFM_SAMPLE_IMAGES:-/restricted/projectnb/batmanlab/shared/Data/RSNA_Breast_Imaging/Dataset/External/UPMC/DICOM/images_png_CC_MLO/Patient_36536418}"
OUT_DIR="${INTEGRATION_OUT_DIR:-/tmp/mammofm_integration_$$}"
MAX_WAIT="${INTEGRATION_MAX_WAIT:-600}"   # seconds before giving up
POLL_INTERVAL=10

mkdir -p "$OUT_DIR"

echo "=== MammoFM PDF integration test ==="
echo "Server:  $SERVER"
echo "Samples: $SAMPLE_DIR"
echo "Out dir: $OUT_DIR"

# Locate exactly one file per required view.
declare -A VIEW_FILES
for VIEW in LCC LMLO RCC RMLO; do
    FILE=$(find "$SAMPLE_DIR" -maxdepth 2 -iname "*${VIEW}*" \( -iname "*.png" -o -iname "*.jpg" \) | head -1 || true)
    if [[ -z "$FILE" ]]; then
        echo "ERROR: no ${VIEW} image found under $SAMPLE_DIR" >&2
        exit 1
    fi
    VIEW_FILES[$VIEW]="$FILE"
    echo "  $VIEW -> $FILE"
done

# Submit
echo ""
echo "--- Submitting job ---"
RESPONSE=$(curl -sf -X POST "$SERVER/api/run" \
    -F "patient_id=integration_test" \
    -F "exam_id=pdf_curl_test" \
    -F "files=@${VIEW_FILES[LCC]};filename=LCC.png" \
    -F "files=@${VIEW_FILES[LMLO]};filename=LMLO.png" \
    -F "files=@${VIEW_FILES[RCC]};filename=RCC.png" \
    -F "files=@${VIEW_FILES[RMLO]};filename=RMLO.png")
JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job id: $JOB_ID"

# Poll
echo ""
echo "--- Polling status (max ${MAX_WAIT}s) ---"
ELAPSED=0
STATUS=""
while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    DATA=$(curl -sf "$SERVER/api/status/$JOB_ID" || echo '{"status":"error"}')
    STATUS=$(echo "$DATA" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
    echo "  [${ELAPSED}s] $STATUS"
    if [[ "$STATUS" == "done" ]]; then break; fi
    if [[ "$STATUS" == "failed" || "$STATUS" == "error" ]]; then
        echo "ERROR: job failed — $DATA" >&2
        exit 1
    fi
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [[ "$STATUS" != "done" ]]; then
    echo "ERROR: job did not finish within ${MAX_WAIT}s (status: $STATUS)" >&2
    exit 1
fi

# Download PDF
echo ""
echo "--- Downloading PDF ---"
PDF_PATH="$OUT_DIR/mammo_report_${JOB_ID}.pdf"
HTTP_CODE=$(curl -sf -o "$PDF_PATH" -w "%{http_code}" "$SERVER/api/report.pdf/$JOB_ID")
echo "HTTP $HTTP_CODE  ->  $PDF_PATH"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "ERROR: PDF endpoint returned HTTP $HTTP_CODE" >&2
    exit 1
fi

# Validate
SIZE=$(stat -c%s "$PDF_PATH" 2>/dev/null || stat -f%z "$PDF_PATH")
MAGIC=$(head -c 4 "$PDF_PATH" 2>/dev/null || dd if="$PDF_PATH" bs=4 count=1 2>/dev/null)
echo "Size: $SIZE bytes"
echo "Magic: $MAGIC"

if [[ $SIZE -lt 5120 ]]; then
    echo "ERROR: PDF too small ($SIZE bytes < 5120)" >&2
    exit 1
fi
if [[ "$MAGIC" != "%PDF" ]]; then
    echo "ERROR: file does not start with %PDF" >&2
    exit 1
fi

echo ""
echo "=== PDF integration test PASSED ==="
echo "  job_id=$JOB_ID"
echo "  pdf_size=${SIZE}"
echo "  pdf_path=$PDF_PATH"
