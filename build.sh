#!/usr/bin/env bash
# Build righter_thesis.tex into a PDF.
# Supports local TeX tools (latexmk/pdflatex) or a Docker fallback.

set -euo pipefail

THESIS="righter_thesis.tex"
TEX_DOCKER_IMAGE="${TEX_DOCKER_IMAGE:-blang/latex:ctanfull}"

if [ ! -f "$THESIS" ]; then
  echo "Error: $THESIS not found in the current directory." >&2
  exit 1
fi

if command -v latexmk &>/dev/null; then
  latexmk -pdf -interaction=nonstopmode "$THESIS"
  latexmk -c
elif command -v pdflatex &>/dev/null; then
  # Fallback: run pdflatex three times to resolve cross-references and TOC
  pdflatex -interaction=nonstopmode "$THESIS"
  pdflatex -interaction=nonstopmode "$THESIS"
  pdflatex -interaction=nonstopmode "$THESIS"
elif command -v docker &>/dev/null; then
  docker run --rm -v "$PWD":/data "$TEX_DOCKER_IMAGE" \
    latexmk -pdf -interaction=nonstopmode "$THESIS"
  docker run --rm -v "$PWD":/data "$TEX_DOCKER_IMAGE" latexmk -c
else
  echo "Error: no TeX builder found (latexmk, pdflatex, or docker)." >&2
  echo "Install a TeX distribution or Docker, then run build.sh again." >&2
  exit 1
fi

echo "Build complete: righter_thesis.pdf"
