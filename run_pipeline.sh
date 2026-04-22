#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Execute the full sleep-apnea detection pipeline
#
# Usage:
#   cd apnea-project
#   bash run_pipeline.sh [--html-only]
#
# Flags:
#   --html-only   Skip notebook execution; only regenerate HTML reports.
#
# Requirements:
#   - Jupyter + nbconvert installed in the active environment
#   - All packages from notebook 00 installed (run pip install first)
#   - Internet connection for PhysioNet download (notebook 01)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

HTML_ONLY=false
[[ "${1:-}" == "--html-only" ]] && HTML_ONLY=true

NOTEBOOKS_DIR="notebooks"
REPORTS_DIR="reports"
LOG_FILE="${REPORTS_DIR}/pipeline_run.log"

mkdir -p "${REPORTS_DIR}"
echo "Pipeline run started: $(date)" | tee "${LOG_FILE}"

# ── 1. Execute notebooks ──────────────────────────────────────────────────────
declare -A NB_STATUS

if [ "$HTML_ONLY" = false ]; then
  echo ""
  echo "Phase 1/2: Executing notebooks ..."
  echo "  (This will download ~200 MB of PhysioNet data on first run)"
  echo ""

  for nb in "${NOTEBOOKS_DIR}"/0*.ipynb; do
    nb_name=$(basename "$nb")
    printf "  %-40s" "$nb_name"
    out_nb="${NOTEBOOKS_DIR}/${nb_name%.ipynb}_executed.ipynb"

    if jupyter nbconvert \
         --to notebook \
         --execute \
         --ExecutePreprocessor.timeout=3600 \
         --output "$out_nb" \
         "$nb" >> "${LOG_FILE}" 2>&1; then
      printf "PASS\n"
      NB_STATUS["$nb_name"]="PASS"
    else
      printf "FAIL  (see ${LOG_FILE})\n"
      NB_STATUS["$nb_name"]="FAIL"
    fi
  done

  echo ""
  echo "Execution summary:"
  for nb_name in "${!NB_STATUS[@]}"; do
    printf "  %-40s %s\n" "$nb_name" "${NB_STATUS[$nb_name]}"
  done | sort
fi

# ── 2. Generate HTML reports ──────────────────────────────────────────────────
echo ""
echo "Phase 2/2: Generating HTML reports ..."

jupyter nbconvert \
  --to html \
  "${NOTEBOOKS_DIR}"/*.ipynb \
  --output-dir "${REPORTS_DIR}" 2>>"${LOG_FILE}"

echo ""
echo "HTML files generated:"
for f in "${REPORTS_DIR}"/*.html; do
  size=$(du -k "$f" | cut -f1)
  printf "  %-45s %s KB\n" "$(basename $f)" "$size"
done

HTML_COUNT=$(ls "${REPORTS_DIR}"/*.html 2>/dev/null | wc -l | tr -d ' ')
if [ "$HTML_COUNT" -eq 9 ]; then
  echo ""
  echo "  All 9 HTML reports confirmed."
else
  echo ""
  echo "  WARNING: Expected 9 HTML files, found ${HTML_COUNT}."
fi

echo ""
echo "Pipeline complete: $(date)" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
