# Master's Thesis — Robert Righter

**Shawnee State University · MS in Mathematics**

This repository contains the LaTeX source for my Master's Thesis.
The primary source file is `righter_thesis.tex`.

---

## Prerequisites

Either of the following is sufficient:

1. A TeX distribution that provides **latexmk** and **pdflatex**
2. Docker (the build script will compile in a TeX container)

| Platform | Recommended distribution |
|---|---|
| Ubuntu / Debian | `sudo apt-get install texlive-full` |
| macOS | `brew install --cask mactex` |
| Windows | [MiKTeX](https://miktex.org/download) |

---

## Building the PDF

Run the provided build script from the root of the repository:

```bash
bash build.sh
```

The script will compile `righter_thesis.tex` using `latexmk` (preferred) or
fall back to three sequential `pdflatex` passes when `latexmk` is not
available. If no local TeX tools are installed but Docker is available, the
script uses `blang/latex:ctanfull` automatically. The finished document is
written to **`righter_thesis.pdf`** in the same directory.

To override the Docker image:

```bash
TEX_DOCKER_IMAGE=blang/latex:ctanfull bash build.sh
```

### Manual build (latexmk)

```bash
latexmk -pdf righter_thesis.tex
```

### Manual build (pdflatex only)

```bash
pdflatex righter_thesis.tex
pdflatex righter_thesis.tex   # second pass for cross-references
pdflatex righter_thesis.tex   # third pass for TOC / list of figures
```

---

## License

This work is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
