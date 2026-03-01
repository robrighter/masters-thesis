# Master's Thesis — Robert Righter

**Shawnee State University · MS in Mathematics**

**Thesis Advisor:** [Dr. John Whitaker](https://www.shawnee.edu/areas-study/college-arts-and-sciences/math-sciences/faculty)

## Recursive Computation of Compound Random Variables with a Finite-Mixture Count Distribution

### Abstract

Compound random variables provide a natural framework for modeling aggregate quantities
subject to two layers of randomness: the distribution of the summands and the distribution
of the count variable. While the concept is straightforward, calculating the exact
probability distribution of a compound random variable can be a significant computational
challenge. Direct methods involving multiple convolutions are often cumbersome and
inefficient. A more elegant and powerful approach is the use of recursive methods, which
allow for the efficient calculation of the probability P(S_N = k) based on the
probabilities of preceding values.

The primary contribution of this work is the extension and simplification of this recursive
method to the case where the compounding distribution is a finite mixture of Poisson,
Binomial, and Negative Binomial distributions. A direct application of the recursive formula
to this case leads to a computationally intensive, nested problem, where the mixing weights
of the distribution must be recalculated at each step. Additionally, a closed-form
expression is derived that computes these recursive weights directly, thereby eliminating
some of the nested recursion and simplifying the calculation.

📄 [Read the PDF of the thesis](righter_thesis.pdf) · 📊 [View the results of the computation implementation](implementation/results.txt)

---

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

## Implementation

The `implementation/` directory contains a Python 3 implementation of the recursive method
described in the thesis.

**Requirements:** Python 3 (no external dependencies required)

### Running the implementation

```bash
python3 implementation/implementation.py
```

This will run all built-in tests (verifying the recursive computation against expected
values from the thesis examples) followed by Monte Carlo simulation comparisons that
validate the recursive results numerically.

---

## License

This work is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
