#!/usr/bin/env bash
# Build righter_thesis.tex into a PDF.
# Requires a TeX distribution with latexmk and pdflatex (e.g. TeX Live or MiKTeX).

set -euo pipefail

THESIS="righter_thesis.tex"

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
else
  echo "Error: neither latexmk nor pdflatex found." >&2
  echo "Please install a TeX distribution such as TeX Live:" >&2
  echo "  Ubuntu/Debian: sudo apt-get install texlive-full" >&2
  echo "  macOS:         brew install --cask mactex" >&2
  echo "  Windows:       https://miktex.org/download" >&2
  exit 1
fi

echo "Build complete: righter_thesis.pdf"
