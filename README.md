# Lab 5: Galaxy Image Generation with Diffusion Models

Build a denoising diffusion probabilistic model (DDPM) to generate galaxy images from scratch, using 32x32 grayscale images from the Galaxy Zoo survey.

## Learning Objectives

- Implement the forward diffusion process and understand the closed-form noising kernel
- Build a time-conditioned noise predictor network $\epsilon_\theta(z_t, t)$
- Understand why the denoising loss reduces to simple MSE on noise
- Think carefully about spatial inductive biases for the network architecture

## Getting Started

### 1. Accept the assignment

Click the GitHub Classroom link shared by the instructor. This creates your own private copy of this repository under your GitHub account.

### 2. Clone your repository

```bash
git clone https://github.com/bu-ds595/lab05-starter-YOUR_USERNAME.git
cd lab05-starter-YOUR_USERNAME
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

**Option A: Google Colab (recommended, this lab benefits a lot from GPU acceleration!)**

1. Go to [colab.research.google.com](https://colab.research.google.com/) and upload your notebook: **File → Upload notebook**, then select `lab-05-diffusion.ipynb`.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU** (or L4 if available), then click Save.
3. Upload the data and module files using the **Files pane** on the left sidebar (folder icon): drag and drop `galaxy_data.npz` and `diffusion.py` there, or use the upload button at the top of the pane.
4. After training, download `model_params.pkl` from the Files pane (right-click → Download) and save it to your local repo before committing. Also save your notebook (File → Save) and the `diffusion.py` file to your local repo which you will commit and push to complete the assignment.

**Option B: VS Code / JupyterLab (not recommended, training will be very slow)**

```bash
jupyter lab
```

## Exercises

Complete the `TODO` sections in `diffusion.py`:

1. **`linear_noise_schedule`** — Compute betas, alphas, and cumulative alpha_bars from a linear schedule.
2. **`q_sample`** — Implement the closed-form forward process (diffusion kernel): $z_t = \sqrt{\bar\alpha_t}\,x + \sqrt{1-\bar\alpha_t}\,\epsilon$.
3. **`EpsNet`** — Build a time-conditioned network that predicts the noise $\epsilon$ added at each timestep. Think carefully about the spatial structure of the problem.
4. **`compute_loss`** — Implement the denoising objective: sample $t$ and $\epsilon$, compute $z_t$, predict noise, return MSE.

The notebook walks through each step. The self-check cell at the end replicates the autograder.

## Running Tests Locally

```bash
pytest test_diffusion.py -v
```

> **Note:** The first two tests check your schedule and training loop and will pass without a trained model. The last two (`test_sample_quality`, `test_sample_centered`) load `model_params.pkl`, so you need to complete the full notebook and save your trained model before the entire test suite will pass.

## Submitting Your Work

After training, save your notebook, diffusion.py file, and model, then commit and push:

```bash
git add lab-05-diffusion.ipynb diffusion.py model_params.pkl
git commit -m "Complete lab 5"
git push
```

You can push multiple times — only the final version at the deadline will be graded.

## Grading (4 points)

| Test | Points | Requirement |
|------|--------|-------------|
| `test_schedule_and_forward` | 1 pt | Noise schedule shapes/properties + forward process limiting behavior |
| `test_loss_decreases` | 1 pt | Training reduces loss over 20 steps |
| `test_sample_quality` | 1 pt | Saved model generates finite, valid samples |
| `test_sample_centered` | 1 pt | Generated galaxies are brighter at the center than the edges |

## Data

[GalaxyMNIST](https://github.com/mwalmsley/galaxy_mnist) (Walmsley et al.), from Galaxy Zoo DECaLS Campaign A. Images are 32x32 grayscale.
