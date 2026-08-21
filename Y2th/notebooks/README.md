# Y2th Notebooks — Environment Setup

Notebooks in this folder:

* `MDN_illustration.ipynb`
* `MNIST_GAN.ipynb`
* `MNIST_VAE.ipynb`
* `MNIST_ScoreDiffusion.ipynb`
* `Flow_RealNVP_twomoon_conditional.ipynb`
* `DSM_ALD_2D_Mixture.ipynb`
* `sbi_LtU_MHo.ipynb` / `sbi_LtU_MHo_exercise.ipynb`

For the full, repo-wide environment setup (conda, CUDA, Colab option) see the
[top-level README](../../readme.md). This file only lists what's specifically
needed to run **every notebook in this folder**, and the extra packages required
by the SBI notebook.

## Minimum requirements

| Package | Notes |
|---|---|
| `python` | 3.11 |
| `numpy`, `scipy` | |
| `pandas` | **pin to `2.3.1`** — see "Known pitfalls" below |
| `matplotlib`, `seaborn` | |
| `scikit-learn` | used for `make_moons` in `Flow_RealNVP_twomoon_conditional.ipynb` |
| `torch`, `torchvision`, `torchaudio` | `2.5.1` / `0.20.1` / `2.5.1` (CPU or MPS on macOS; see top-level README for CUDA wheels) |
| `ipympl`, `ipywidgets` | required for `%matplotlib widget` in `MNIST_VAE.ipynb` |
| `jupyterlab` or `notebook`, `ipykernel` | |
| `tqdm`, `h5py` | |

Extra requirements **only** for `sbi_LtU_MHo.ipynb` / `sbi_LtU_MHo_exercise.ipynb`:

| Package | Notes |
|---|---|
| `sbi` | simulation-based inference backend |
| `lampe` | pin to `0.9.0` |
| `zuko` | normalizing flows used by `sbi` |
| `einops` | |
| `ltu-ili` | not on PyPI — install from GitHub (see below) |
| `gdown` | used to fetch CAMELS data from Google Drive; auto-installed by the notebook if missing |

## Quick setup

If you already created the repo's shared environment from
[`env/environment_full.yml`](../../env/environment_full.yml), it already has
everything except `ipympl` and the SBI/ILI extras:

```bash
conda activate euclid-school-ml   # or whatever you named it
pip install ipympl einops "git+https://github.com/maho3/ltu-ili.git"
```

Starting from scratch instead:

```bash
conda create -n euclid-school-y2th -c conda-forge \
    python=3.11 numpy scipy "pandas=2.3.1" scikit-learn \
    matplotlib seaborn h5py tqdm ipykernel jupyterlab ipympl ipywidgets
conda activate euclid-school-y2th

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# only needed for the sbi_LtU_MHo notebooks:
pip install sbi "lampe==0.9.0" zuko einops
pip install "git+https://github.com/maho3/ltu-ili.git"
```

Then make sure the notebook's kernel (top-right of the notebook, or VS Code's
kernel picker) points at this environment, and **restart the kernel** after
installing anything.

## Known pitfalls

* **`pandas >= 3.0` breaks `ltu-ili`'s plotting code** (`PlotSinglePosterior`),
  raising `KeyError: 'future'` or `NameError: name 'ABCSeries' is not defined`.
  `ltu-ili` doesn't pin a pandas version, so a plain `pip install ltu-ili`'s
  dependencies can silently pull in pandas 3.x. Pin `pandas==2.3.1` and, if you
  had to downgrade it after the fact, **restart the kernel** — an already-running
  kernel keeps the old pandas modules loaded in memory even after the package on
  disk has been downgraded.
* **`%matplotlib widget` fails with `RuntimeError: 'widget' is not a recognised
  GUI loop or backend`** if `ipympl` isn't installed in the environment the
  selected kernel is actually running (a stale/mismatched kernel selection in
  VS Code is a common cause). Confirm the kernel picker points at the right
  interpreter path, install `ipympl` there, and restart the kernel.
