#!/usr/bin/env bash
set -euo pipefail

uv run pytest -v ./cs336-data/tests --junitxml=test_results.xml || true
echo "Done running tests"

# Set the name of the output tar.gz file
output_file="cs336-spring2026-assignment-2-submission.zip"
rm "$output_file" || true

# Compress cs336-data/, data/, and the writeup PDF (excluding large/binary files to stay under 100 MB)
zip -r "$output_file" cs336-data/ data/ ECE405_assigment_2_writeup.pdf \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.log' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth' \
    -x 'data/CC-MAIN-*'

echo "All files have been compressed into $output_file"
