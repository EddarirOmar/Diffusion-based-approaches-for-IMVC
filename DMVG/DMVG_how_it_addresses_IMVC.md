# DMVG approach: How it addresses IMVC

Note: folder name requested is `IMVG`; this note explains the implemented DMVG pipeline in [DMVG](../DMVG).

## 1) IMVC objective in this implementation
In IMVC, some views are unavailable per sample. DMVG addresses this by generating missing views first, then enabling complete-view downstream processing.

Core idea in this codebase:
1. learn per-view latent representation with AE,
2. train conditional diffusion to generate the missing-view latent from available-view latents,
3. decode generated latent back to view space.

## 2) Framework type: decoupled multi-stage
This implementation is decoupled because it is trained and executed as separate stages/scripts, not one end-to-end joint optimizer.

Stages are split across scripts:
- AE training: [DMVG/AE_train.py](../DMVG/AE_train.py)
- latent extraction from trained AEs: [DMVG/AE_test.py](../DMVG/AE_test.py)
- conditional DDPM training per target view: [DMVG/carl_train.py](../DMVG/carl_train.py)
- DDPM sampling for missing-view latent generation: [DMVG/carl_test.py](../DMVG/carl_test.py)
- AE decoding of generated latents: [DMVG/AE_generate.py](../DMVG/AE_generate.py)

## 3) Architecture pieces and role in IMVC

### 3.1 Per-view AE (2D conv autoencoder)
- Module: `AE2d` in [DMVG/model.py](../DMVG/model.py).
- Role: compress each view into 512-d latent and reconstruct.
- IMVC value: creates a compact latent space where missing-view generation is easier.

### 3.2 Conditional latent DDPM
- Modules: `UNet`, `DDPM`, schedules in [DMVG/model.py](../DMVG/model.py).
- Inputs:
  - target-view latent as diffusion variable `x`,
  - other-view latent concat as condition `c`.
- IMVC value: generates plausible target-view latent for missing entries conditioned on observed views.

### 3.3 Classifier-free guidance style conditioning
- Context masking in DDPM training/sampling in [DMVG/model.py](../DMVG/model.py).
- IMVC value: improves controllability and robustness of conditional generation.

## 4) Training procedure (how to train)

### Stage A: train AE per view
- Script: [DMVG/AE_train.py](../DMVG/AE_train.py)
- Uses available target-view training split from fold mask.
- Loss: reconstruction MSE on cropped output.
- Output: `models/AE_*_ep100.pth`.

### Stage B: extract AE latents
- Script: [DMVG/AE_test.py](../DMVG/AE_test.py)
- Encodes samples and saves `data/AE_carl_view*_del*_fold*.npy`.

### Stage C: train DDPM per target view
- Data assembly: [DMVG/datasets.py](../DMVG/datasets.py)
  - `x_train`: target view latent for observed target-view samples.
  - `c_train`: concatenated other-view latents, masked by view availability.
- Script: [DMVG/carl_train.py](../DMVG/carl_train.py)
- DDPM objective: noise prediction MSE between true noise and U-Net prediction.
- Output: `models/ddpm_*_ep1000.pth`.

## 5) Inference procedure

### Stage D: generate missing-view latents
- Script: [DMVG/carl_test.py](../DMVG/carl_test.py)
- For target-missing test samples, run reverse diffusion with condition `c`.
- Output: `data/ddpm_*_pairedrate*_fold*.npy`.

### Stage E: decode generated latents to view images/features
- Script: [DMVG/AE_generate.py](../DMVG/AE_generate.py)
- Loads generated DDPM latents and decodes via AE decoder (`forward_x_rec`).
- Output: `data/generate_*_del*_fold*.npy`.

## 6) Why this addresses IMVC
1. Missing views are explicitly synthesized rather than ignored.
2. Conditioning on available views preserves cross-view consistency.
3. Working in latent space reduces generation complexity and stabilizes training.

## 7) Practical summary
- Framework type: decoupled (multi-stage pipeline).
- Training: sequential stage-wise training (AE then DDPM).
- Inference: conditional latent sampling then AE decoding.
