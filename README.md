# Report figure reproduction

This repository reproduces the two figures in Noah Ripstein's report using Python/BoTorch and persistent R/BayesGP model evaluations. It contains only the two workflows and their inputs. The original exploratory repository was preserved.

## Setup on macOS

Use Python 3.10 and R 4.4.1, with Xcode command-line tools installed. From this repository:

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
brew install poppler
Rscript --vanilla -e 'install.packages("remotes", repos="https://cloud.r-project.org"); remotes::install_version("BayesGP", version="0.1.3", repos="https://cloud.r-project.org"); install.packages(c("aghq","npreg","jsonlite"), repos="https://cloud.r-project.org")'
```

Recorded R dependency versions are in `provenance/original_r_environment.json`. Use `remotes::install_version` to match them where necessary; current dependency resolution may differ. Scientific recomputation can vary with R dependencies and optimizer versions. The pinned Python file includes the plotting dependencies. The launcher sets R's shared-library path before Python starts so R and PyTorch share an OpenMP runtime. `BOSS_PYTHON` can select an existing scientific Python environment.

## Regenerate both report figures

```sh
scripts/boss-python submission.py plot
```

This writes only the two figure PNG/PDF pairs plus Figure 2's numerical spline CSVs under `output/`. Choose another empty directory with `--output`. Results bundled under `results/` are never modified. Plotting requires R/npreg for the change-point splines but does not evaluate conditional models.

## Fresh computation

```sh
scripts/boss-python submission.py compute --figure 1 --output runs/seasonality
scripts/boss-python submission.py compute --figure 1 --output runs/seasonality --resume
scripts/boss-python submission.py compute --figure 2 --output runs/change-point
```

Figure 1 regenerates seed 123 and seeds 1–10, evaluates 801 grid points and 80 BOSS points per dataset (9,691 conditional fits), and independently refits reporting prefixes. This is expensive. Figure 2 computes 50 BOSS evaluations on the bundled original dataset. Both B20/B50 splines use prefixes of that trajectory. Budgets include five initial points. Fresh Figure 2 does not run MCMC or a reference grid.

`--resume` is explicit and supported for Figure 1. Checkpoints verify source, configuration, grid, runtime, and regenerated dataset. Committed evaluations are reused; an evaluation interrupted before checkpoint commit may repeat. Other nonempty outputs are rejected. To draw fresh results, assemble a results directory containing `figure1/` from the seasonality run and `figure2/` from the change-point run, then:

```sh
scripts/boss-python submission.py plot --results-dir runs/combined --output output/fresh
scripts/boss-python submission.py smoke --output runs/smoke
```

The smoke workflow uses seven seasonality grid points and six BOSS evaluations for each model. It checks acquisition, data handoff, completed-checkpoint resume, and incompatible-checkpoint rejection. Its results are not report figures.

## Figure methods and attribution

Figure 1 plots a fresh-computation grid oracle and GP posterior-mean densities at B30/B60 for seed 123, and arithmetic mean KL/KS with linear 10th–90th percentiles over exactly ten independent datasets. Inputs are normalized to [0,1]; journal GP fitting uses current-mean centering, fixed sample-variance scale and bounded lengthscale fitting. Saved metadata records the complete settings. Original journal comparison PNGs are retained in `references/`.

Figure 2 recomputes smoothing splines (`npreg::ss`) from our saved B20/B50 objective prefixes, then exponentiates and normalizes by trapezoidal integration. The blue MCMC histogram, dashed grid curve, axes and labels are published reference graphics; they are not independently recomputed. `references/change_point_layer.pdf` contains the cropped journal figure with the original red curves suppressed. Only our red BOSS curves are added. The original acquisition iteration cap is explicitly retained at 300; seasonality uses 100.

Reference graphics: Dayi Li and Ziang Zhang (2026), DOI 10.1016/j.csda.2025.108253, journal Figures 1 and 2 (PDF pages 9 and 10). Original simulation source: https://github.com/AgueroZZ/BOSS_website . The change-point observations were copied from the original simulation dataset through the exploratory repository. Reference graphics remain attributed to their original authors; no new license is asserted for them.

## Verification

`verification/report.json` records rendering and numerical comparisons. Figure 1 matched every RGB pixel. Figure 2 has tiny rasterization differences from composing the same geometry after cropping; its spline values and layout match. Both were visually inspected. Reduced fresh model runs passed; full scientific recomputation was not performed during extraction. Validation reused the installed Python/R scientific runtime, with PDF dependencies supplied from the available document runtime; a clean environment install was not tested.

The clean-clone check verifies all input assets are tracked and both figures redraw without reading code or data from the original checkout. Expected report PNGs are verification targets, never plotting inputs. The manifest records original source checksums, the source commit, and bundled input hashes. No GitHub remote is configured.
