# AstroCLIP Teaching Notebook — Setup & Run Guide

This guide gets `astroclip_teaching_standalone.ipynb` running on your own laptop,
from a fresh environment to a finished run. Everything below happens **inside
this repo**, in `Y2obs/notebooks/`.

The notebook trains a small image autoencoder and a small spectrum autoencoder
on a tiny subset of the AstroCLIP dataset, aligns them with a CLIP loss,
visualises the joint embedding space, and finishes with a toy redshift
regression. On the default tiny subset it is meant to run entirely on CPU (or
Apple Silicon) in well under an hour.

**Short on time in class?** `astroclip_teaching_standalone_quicklook.ipynb`
is a fast-forward version that *loads* the three pretrained checkpoints
(`astroclip_demo/*.ckpt`) instead of training them, then jumps straight to
reconstructions, the UMAP plot, and the redshift regression — a live run of
it takes well under a minute. It needs those checkpoints to already exist,
so run the full notebook at least once beforehand to produce them.

## 0. Prerequisites

- [Miniforge/conda](https://github.com/conda-forge/miniforge) installed
- ~2 GB free disk space (dataset subset + checkpoints + logs)
- Internet access, only for the one-time environment setup and dataset download

## 1. Create the environment

```bash
conda create -n astroclip-teaching python=3.10 -y
conda activate astroclip-teaching
export PYTHONNOUSERSITE=1   # make sure packages install into this env, not user-site
pip install --upgrade pip
```

### Install PyTorch

**macOS (Apple Silicon):**
```bash
pip install torch torchvision torchmetrics
```

**Linux/Windows with an NVIDIA GPU** — pick the wheel matching your CUDA version
from https://pytorch.org/get-started/locally/, e.g.:
```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchmetrics
```

**No GPU available:** the plain `pip install torch torchvision torchmetrics`
command also works — everything below falls back to CPU automatically.

### Install the remaining dependencies

```bash
python -m pip install astropy "datasets==4.2.0" huggingface_hub jaxtyping wandb \
    python-dotenv scikit-image scikit-learn plotly pyro-ppl "lightning[extra]==2.3.3"
python -m pip install "h5py>=3.0" emcee speclite zeus-mcmc
python -m pip install jupyterlab ipywidgets umap-learn seaborn

# Git dependencies used internally by AstroCLIP (installed without deps —
# everything they need is already covered above)
python -m pip install --no-deps git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069
python -m pip install --no-deps git+https://github.com/changhoonhahn/provabgs.git
```

A known-good, fully pinned package list (from a machine this notebook has been
verified to run on) is available at
[`teaching_scripts/requirements_pip.txt`](teaching_scripts/requirements_pip.txt)
if you'd rather install everything in one shot with
`pip install -r teaching_scripts/requirements_pip.txt` — note it pins an
Apple-Silicon build of PyTorch, so on Linux/CUDA prefer the manual steps above.

### Install the AstroCLIP package (editable)

From inside `Y2obs/notebooks/`:

```bash
cd Y2obs/notebooks
pip install -e ./astroclip --no-deps
```

This exposes the `astroclip` package used by the notebook (`astroclip.data.datamodule`,
etc.) and the console scripts described in `astroclip/`. Because it's an
**editable** install, pip records a pointer to *this exact folder*. If you
have set up AstroCLIP before for a different checkout of this repo (or a
different year's version), reinstalling with the command above updates that
pointer to the current repo — always do this from inside the checkout you
intend to actually run.

## 2. Prepare the tiny local dataset

The notebook expects a prepared Hugging Face dataset on disk — it does **not**
download anything itself. Build the tiny subset once with:

```bash
cd Y2obs/notebooks   # if not already there
python teaching_scripts/prepare_dataset.py \
    --dataset EiffL/AstroCLIP \
    --output-dir astroclip_demo/teaching_notebook_subset \
    --train-size 256 --test-size 64 \
    --streaming --overwrite
```

- `--streaming` pulls only the first `train-size + test-size` examples instead
  of downloading the full hosted dataset — this is what keeps the download
  small and fast (a few minutes) on a laptop connection.
- `--train-size` / `--test-size` control how much data you train on. 256/64 is
  a good laptop-friendly default; raise them (e.g. 1024/256, the script's own
  defaults) if you have more time and want less noisy results.
- The output path **must** be `astroclip_demo/teaching_notebook_subset`
  relative to `Y2obs/notebooks/` — that's where the notebook looks for it by
  default (see §3 below for why).

## 3. Launch Jupyter — from `Y2obs/notebooks/`

```bash
cd Y2obs/notebooks
jupyter lab
```

**This working directory matters.** The notebook resolves both its dataset
path and its `astroclip`/`teaching_scripts` imports relative to the directory
Jupyter was launched from (`ASTROCLIP_ROOT` defaults to `./astroclip_demo` if
you haven't set the environment variable). Launching from anywhere else is
the most common cause of `ModuleNotFoundError: No module named
'teaching_scripts'` or a dataset `FileNotFoundError` — see Troubleshooting
below.

Then open `astroclip_teaching_standalone.ipynb` and run the cells top to
bottom (`Run All` is fine).

## 4. What each section does

| Section | What happens |
|---|---|
| 0. Setup | Imports, resolves `ASTROCLIP_ROOT`, sets `DATASET_PATH` |
| Load/inspect subset | Loads the prepared dataset, plots a few image/spectrum pairs |
| 1. Image autoencoder | Trains a small conv autoencoder on 144×144 crops (8 epochs) |
| 2. Spectrum autoencoder | Trains a small fully-connected autoencoder on 1D spectra (12 epochs) |
| 3. CLIP alignment | Aligns the two frozen encoders with a CLIP loss (10 epochs) |
| 4. Embed + UMAP | Embeds the test split, visualises the joint space |
| 5. Redshift regression | Trains a tiny MLP on the embeddings to predict redshift |
| 6. Morphology (optional) | **Skip unless you have `galaxy_zoo_embeddings.h5`** — see below |

Each training stage logs to `astroclip_demo/logs/<name>/version_*/metrics.csv`
and saves a checkpoint under `astroclip_demo/`; none of this is meant to be
committed to git (see `.gitignore`).

The final **"6. Morphology Classification"** section is optional and expects
a pre-built `galaxy_zoo_embeddings.h5` file that most students won't have —
it will raise a clear `FileNotFoundError` if that file is missing. That's
expected; everything before it stands on its own.

## 5. Troubleshooting

**`ModuleNotFoundError: No module named 'teaching_scripts'` (or `astroclip`)**
Jupyter wasn't launched from `Y2obs/notebooks/`, so the local package folders
aren't on the import path and Python falls back to whatever (possibly stale,
possibly nonexistent) editable install is registered in your environment.
Fix: quit Jupyter, `cd Y2obs/notebooks`, relaunch `jupyter lab` from there.

**`FileNotFoundError: ... teaching_notebook_subset not found`**
You haven't run the dataset preparation step yet, or you're not in the right
working directory. Re-run §2, and confirm you're running Jupyter from
`Y2obs/notebooks/` (§3).

**Section 6 (Morphology) raises `FileNotFoundError` for `galaxy_zoo_embeddings.h5`**
Expected — this section needs a separately-prepared file most students won't
have. Safe to skip.

**Training feels slow**
Reduce `--train-size`/`--test-size` in §2 and re-run `prepare_dataset.py`
with `--overwrite`, or lower `max_epochs` in the training cells.

## 6. Cleaning up

Everything generated locally lives under `Y2obs/notebooks/astroclip_demo/`
(dataset, checkpoints, logs) — safe to delete and regenerate at any time, and
already excluded from git via `.gitignore`.
