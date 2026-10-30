# BOSS reproduction with BoTorch

Code for the seasonality and change-point examples in my oral comprehensive exam report. The BOSS sequential design is implemented in Python with BoTorch; conditional model fits use BayesGP in R.

## Setup

Tested on macOS with Python 3.10 and R 4.4.1. Requires Xcode command-line tools.

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
brew install poppler
Rscript --vanilla -e 'install.packages("remotes", repos="https://cloud.r-project.org"); remotes::install_version("BayesGP", version="0.1.3", repos="https://cloud.r-project.org"); install.packages(c("aghq", "npreg", "jsonlite"), repos="https://cloud.r-project.org")'
```

A clean environment installation has not been tested.

## Reproduce the figures

```sh
scripts/boss-python submission.py plot
```

This uses the saved results in `results/` and writes PNGs, PDFs, and spline CSVs to `output/`. Use `--output` to choose another empty directory.

To rerun the computations:

```sh
scripts/boss-python submission.py compute --figure 1 --output runs/seasonality
scripts/boss-python submission.py compute --figure 2 --output runs/change-point
```

Figure 1 requires 9,691 conditional model fits. Figure 2 requires 50. Add `--resume` to the Figure 1 command to continue an interrupted run. To plot fresh results, put the resulting `figure1/` and `figure2/` folders in one directory and pass it with `--results-dir`.

A smaller computation check is available with:

```sh
scripts/boss-python submission.py smoke --output runs/smoke
```

## Source and scope

These examples reproduce Figures 1 and 2 of [Li and Zhang (2026)](https://doi.org/10.1016/j.csda.2025.108253), using their [simulation code and change-point data](https://github.com/AgueroZZ/BOSS_website).

## R package versions

R 4.4.1, with BayesGP 0.1.3, TMB 1.9.19, aghq 0.4.3, Matrix 1.7.2, npreg 1.1.0, jsonlite 1.8.9.
