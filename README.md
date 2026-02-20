# Signapse — Computer Vision Research Engineer: Technical Test

## Welcome

This technical test asks you to build a simplified version of a core component from our research pipeline: a **single-step diffusion-based image enhancer**, inspired by the ELITE paper ([arXiv:2601.10200](https://arxiv.org/abs/2601.10200)). This component takes a degraded rendering (simulating output from a 3D Gaussian avatar) and produces a clean, detail-restored image in a single forward pass.

**Time budget**: 4-6 hours. We respect your time — focus on demonstrating methodology and understanding rather than perfect results. A short training run on limited data is perfectly acceptable.

**Compute**: The test is designed to run on a **free Google Colab GPU** (T4, 16GB). You do not need access to your own GPU hardware. See [Compute Setup](#compute-setup-google-colab) below.

**AI tools**: You are strongly encouraged to use AI coding assistants (Claude Code, Codex, Copilot, etc.) as Signapse does in normal work. We care about your ability to direct and validate the output, not whether you typed every line.

---

## Background Reading

Please review these resources before starting:

1. **ELITE paper** — Section 3.3 + Supplementary B.2: [arXiv:2601.10200](https://arxiv.org/abs/2601.10200)
2. **pix2pix-turbo** — The reference implementation for single-step SD-Turbo image translation: [github.com/GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo)
   - Particularly the [training docs](https://github.com/GaParmar/img2img-turbo/blob/main/docs/training_pix2pix_turbo.md) and `src/pix2pix_turbo.py`

### Context

ELITE uses a **rendering-guided single-step diffusion enhancer** that takes degraded 3D Gaussian avatar renderings and enhances them into photorealistic images. Unlike multi-step diffusion approaches that generate from pure noise (slow, identity-hallucinating), this approach grounds generation on the existing rendering, achieving 60x faster inference with better identity preservation.

The enhancer follows the **pix2pix-turbo / DIFIX** approach to adapt SD-Turbo for paired image-to-image translation:
- The **input image** is encoded via the VAE encoder and passed through the UNet (at a fixed high noise timestep) — replacing random noise with structured input
- **LoRA** adapters are applied to the UNet and VAE decoder; skip connections bridge encoder→decoder
- The VAE encoder is **frozen**; only LoRA weights, skip convolutions, and UNet `conv_in` are trainable
- Training uses **L1 + LPIPS + Gram matrix** losses on paired data: `(degraded_rendering, clean_ground_truth)`

You will build a simplified version of this enhancer using a public face dataset as a proxy for our proprietary sign language data.

---

## The Task

### Part 1: Data Pipeline

**Goal**: Build a training data pipeline that produces paired data: `(degraded, ground_truth)`.

**Dataset**: Use **CelebA-HQ** — 30K face images at 1024x1024, freely available on HuggingFace with no login required:

```python
from datasets import load_dataset

# Downloads ~2.76GB. You only need a small subset for this test.
dataset = load_dataset("mattymchen/celeba-hq")
```

You only need a small subset — **200-500 images** is sufficient. Resize to **512x512** (SD-Turbo's native resolution).

**What to build**:

1. **Synthetic degradation pipeline**: Create degraded versions of clean images that simulate typical artefacts from 3D Gaussian avatar renderings. Consider combinations of:
   - Gaussian blur (simulating rendering softness)
   - Downscale + upscale (resolution loss)
   - Noise injection (rendering noise)
   - Colour jitter / slight colour shift
   - Localised artefacts (e.g., block artefacts, edge ghosting)

2. **Pair construction**: For each training sample, produce:
   - `degraded`: Synthetically corrupted version of a clean image
   - `ground_truth`: The original clean image (target)

3. **Train/test split**: Hold out a test set for evaluation.

**Deliverable**: A clean, configurable data preparation script with a README explaining your degradation choices.

---

### Part 2: Model Architecture & Training

**Goal**: Implement a single-step image enhancer by adapting SD-Turbo for image-to-image enhancement, following the pix2pix-turbo architecture.

**Reference code**: Study the [pix2pix-turbo implementation](https://github.com/GaParmar/img2img-turbo), particularly `src/pix2pix_turbo.py`. You may use this as a starting point, adapt it, or reimplement the key ideas — your choice.

**What to build**:

1. **Base architecture** — adapt SD-Turbo (`stabilityai/sd-turbo`) for paired image-to-image enhancement following the pix2pix-turbo approach:
   - **VAE encoder** encodes the degraded input image into latent space (frozen)
   - **UNet** processes the latent at a fixed high noise timestep (e.g., t=999) — this is what makes it single-step, not iterative denoising from random noise
   - **VAE decoder** reconstructs the enhanced output, with **skip connections** from encoder to decoder preserving spatial detail
   - **LoRA adapters** on the UNet and VAE decoder for parameter-efficient fine-tuning
   - Only train: LoRA weights, skip convolution layers, UNet `conv_in`

2. **Training**:
   - Combined loss: **L1 + LPIPS + Gram matrix** (style loss)
   - Use appropriate loss weights (experiment or reference ELITE/DIFIX)
   - A short run is fine — we're assessing methodology, not convergence. On a Colab T4, even 50-200 training steps with a small dataset will demonstrate your approach

3. **Colab compatibility**:
   - Use `fp16` / mixed precision to fit in 16GB VRAM
   - Keep batch size small (1-2)
   - The model should load, train, and run inference without exceeding T4 memory

**Deliverable**: Model architecture code and training script with configurable hyperparameters.

---

### Part 3: Inference & Evaluation

**Goal**: Evaluate your enhancer quantitatively and qualitatively.

**What to build**:

1. **Inference script** that runs the trained enhancer on the held-out test set

2. **Quantitative metric** — compute and report **LPIPS** (learned perceptual similarity) across the test set

3. **Qualitative results** — produce a visual comparison grid showing:
   - `Degraded input | Enhanced output | Ground truth`
   - Include at least 5-10 test examples

**Deliverable**: Inference script, LPIPS score, and comparison grid images.

---

## Compute Setup: Google Colab

You can run this entire test for free on Google Colab with a T4 GPU (16GB VRAM):

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook
2. Connect to a GPU runtime: **Runtime → Change runtime type → T4 GPU**
3. Free tier provides GPU sessions (usage limits apply — typically several hours per day). If you hit a limit, you can resume later; your code should support this via checkpoint saving/loading.
4. Install dependencies in a cell:
   ```python
   !pip install diffusers transformers accelerate peft lpips datasets torchvision
   ```
5. SD-Turbo model weights (~3.3GB) will download on first use. Combined with the dataset (~2.76GB, though you only need a subset), this fits comfortably in Colab's disk allocation.

**Tips for staying within T4 memory (16GB VRAM)**:
- Use `torch.float16` everywhere (model loading, training, inference)
- Batch size 1-2 during training
- Use `torch.cuda.amp.autocast` or `accelerate` mixed precision
- Call `torch.cuda.empty_cache()` between data loading and training if needed
- Gradient checkpointing can help if you hit OOM during training

If you prefer to work locally or have access to other compute, that's fine too — just note the hardware used in your README.

---

## Deliverables

Please submit a **GitHub repository** containing:

```
your-repo/
  data/
    prepare_data.py       # Data download + degradation pipeline
  model/
    architecture.py       # SD-Turbo adaptation + LoRA setup
    losses.py             # L1 + LPIPS + Gram matrix loss
  train.py                # Training script (configurable hyperparams)
  inference.py            # Inference + metric computation
  requirements.txt        # Dependencies
  README.md               # Setup, how to run, design decisions
  results/
    metrics.json           # LPIPS score
    comparisons/           # Visual comparison images
```

A **Google Colab notebook** (`.ipynb`) that runs the full pipeline end-to-end is also welcome as an alternative or supplement to the scripts above.

---

## What We're Looking For

| Area | What matters |
|------|-------------|
| **Diffusion model understanding** | Correct adaptation of SD-Turbo for single-step image enhancement; understanding of why single-step (not multi-step); proper LoRA application |
| **Code understanding** | An overall understanding of what the code does, design decisions you took and why different parts may have been challenging |
| **Data pipeline** | Thoughtful degradation design; justified choices; efficient loading |
| **Evaluation** | Correct LPIPS implementation; meaningful visual comparisons |

We **do not** expect:
- A fully converged model — short training runs are fine
- Perfect results — methodology matters more than numbers
- Access to expensive hardware — Colab T4 is the target

## What next

After sharing the results, we can assess and then do a follow-up call to discuss your approach, results and potential improvements.
