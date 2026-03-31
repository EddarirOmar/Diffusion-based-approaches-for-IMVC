# Diffusion-Based Incomplete Multi-View Clustering (DCG): Project Description

## 1. Project Overview

This project implements an **incomplete multi-view clustering (IMVC)** method in PyTorch, where samples may miss one or more views during training and inference. The implementation focuses on:

- learning view-specific latent representations from partially observed multi-view data,
- reconstructing missing-view latent structure through **diffusion modeling in latent space**, and
- performing unsupervised clustering from a fused representation.

In the current codebase, the training and evaluation pipeline is implemented for **two-view IMVC** in [ICDM.py](ICDM.py), with per-view autoencoders, per-view latent denoisers, an attention fusion module, and a clustering head.

## 2. Implemented Model Architecture

### 2.1 Per-View Autoencoders

Each view has an independent MLP autoencoder:

- View-1 autoencoder: `autoencoder1`
- View-2 autoencoder: `autoencoder2`

Defined in [baseModels.py](baseModels.py) as `Autoencoder`.

#### Architecture details

- Encoder and decoder are both fully connected stacks (`nn.Linear`).
- Hidden layers optionally use `nn.LayerNorm` (`use_norm=True` in `Autoencoder`).
- Nonlinearity is configurable (`sigmoid`, `leakyrelu`, `tanh`, `relu`, `silu`), with default `'silu'`.
- The latent output is explicitly L2-normalized:

$$
z = \frac{f_{enc}(x)}{\|f_{enc}(x)\|_2 + 10^{-8}}
$$

This normalized latent code is used by diffusion, fusion, and clustering modules.

#### Notes on normalization and activation

The currently active model definitions in [baseModels.py](baseModels.py) use `LayerNorm` across major blocks (autoencoder hidden layers when enabled, diffusion network, clustering projector, and attention MLP). This is a deliberate shift from the older BatchNorm-heavy version kept in [baseModels (old).py](baseModels%20(old).py).

### 2.2 Latent Diffusion Model

Diffusion is applied to **latent vectors** (not raw inputs), with one denoiser per view:

- `df1` for view 1 latent space
- `df2` for view 2 latent space

Implemented as `Unet` in [baseModels.py](baseModels.py).

#### Denoiser architecture (MLP-UNet style)

- Input: noisy latent $z_t \in \mathbb{R}^{d}$ plus timestep embedding.
- Timestep embedding: sinusoidal embedding followed by an MLP (`TimeEmbed`):

$$
e_t = \text{MLP}(\text{SinusoidalEmbedding}(t))
$$

- Conditioning: concatenate latent with time embedding:

$$
h_t = [z_t \| e_t]
$$

- Predict noise with a deep MLP stack using `Linear + LayerNorm + SiLU` blocks.

The network predicts $\epsilon_\theta(z_t, t)$ for DDPM-style training.

### 2.3 Noise Scheduler

`NoiseScheduler` in [baseModels.py](baseModels.py) provides forward diffusion, posterior coefficients, and reverse sampling.

#### Supported schedules

- `linear`: linearly spaced $\beta_t$
- `cosine`: cosine cumulative schedule transformed into betas

#### Core quantities

Given $\alpha_t = 1 - \beta_t$:

$$
\bar\alpha_t = \prod_{s=1}^{t} \alpha_s
$$

The forward noising process (implemented in `add_noise`) is:

$$
z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

For reverse steps, the scheduler implements:

- reconstruction of $z_0$ from $(z_t, \epsilon_\theta)$,
- posterior mean coefficients,
- variance term for stochastic reverse transition.

### 2.4 Attention-Based Multi-View Fusion

`AttentionLayer` in [baseModels.py](baseModels.py) fuses view latents with learnable soft attention.

For two views $z^{(1)}, z^{(2)}$:

1. Concatenate features: $h = [z^{(1)} \| z^{(2)}]$
2. Feed through an MLP (`LayerNorm + SiLU`)
3. Produce view logits, apply `sigmoid`, divide by temperature $\tau$, then `softmax`
4. Weighted sum:

$$
z_f = w_1 z^{(1)} + w_2 z^{(2)}, \quad (w_1,w_2)=\text{softmax}(\sigma(g(h))/\tau)
$$

This yields a unified latent representation for clustering.

### 2.5 Clustering Module

`ClusterProject` in [baseModels.py](baseModels.py):

- projection head: `Linear -> LayerNorm -> SiLU`
- clustering head: `Linear -> Softmax`

Outputs:

- cluster probability $y \in \Delta^{K-1}$
- projected feature $p$ (intermediate latent for clustering consistency)

## 3. Training Strategy (Implemented Pipeline)

The training loop is in [ICDM.py](ICDM.py), method `icdm.train`.

For each batch:

1. **Encode available views**
   - Use masks to select observed samples per view.
   - Encode via per-view autoencoders.

2. **Build complete-pair subset**
   - Identify samples where both views exist (`index_both`).

3. **Autoencoder reconstruction on complete pairs**
   - Decode each complete latent and compute MSE reconstruction loss.

4. **Diffusion noising and denoising training**
   - Sample random timesteps.
   - Add scheduler noise to each view latent.
   - Predict injected noise with `df1`/`df2`.
   - Optimize MSE between predicted and true noise.

5. **Reverse diffusion recovery (latent-level)**
   - Run iterative reverse steps over all timesteps in descending order.
   - Obtain recovered latent estimates (`v1_recov`, `v2_recov`).

6. **Cross-view fusion and clustering**
   - Fuse complete-pair latents by attention.
   - Produce per-view and fused cluster distributions.

7. **Joint objective optimization**
   - Sum reconstruction, diffusion, cross-view, MI, and clustering terms.
   - Backprop through all trainable modules jointly.

## 4. Loss Functions and Objective

All main loss terms are combined in [ICDM.py](ICDM.py) using functions/classes from [loss.py](loss.py).

### 4.1 Reconstruction Loss (MSE)

On complete pairs:

$$
\mathcal{L}_{rec} = \|\hat{x}^{(1)} - x^{(1)}\|_2^2 + \|\hat{x}^{(2)} - x^{(2)}\|_2^2
$$

### 4.2 Diffusion Loss (Noise Prediction MSE)

For each view $v$:

$$
\mathcal{L}_{df}^{(v)} = \mathbb{E}_{z_0,\epsilon,t}\left[\|\epsilon_\theta^{(v)}(z_t,t)-\epsilon\|_2^2\right]
$$

Total: $\mathcal{L}_{df}=\mathcal{L}_{df}^{(1)}+\mathcal{L}_{df}^{(2)}$.

### 4.3 Cross-View Consistency / Instance-Level Recovery Loss

Implemented with `InstanceLoss` (contrastive form) between recovered latent of one branch and opposite-view latent on complete pairs:

$$
\mathcal{L}_{ce}=\mathcal{L}_{inst}(z_{rec}^{(1)}, z^{(2)}) + \mathcal{L}_{inst}(z_{rec}^{(2)}, z^{(1)})
$$

This enforces alignment between reverse-diffused recovery and counterpart view representation.

### 4.4 Mutual Information Loss (MMI)

From [loss.py](loss.py), MMI is computed between fused latent and each view latent using a joint probability matrix:

$$
\mathcal{L}_{mmi}=\text{MMI}(z_f, z^{(1)}) + \text{MMI}(z_f, z^{(2)})
$$

This regularizes fused representation to preserve shared information across views.

### 4.5 Clustering Consistency Loss

`ClusterLoss` enforces agreement between view-specific cluster assignments, combining contrastive cluster alignment with entropy regularization.

$$
\mathcal{L}_{cluster}=\text{ClusterLoss}(y^{(1)}, y^{(2)})
$$

### 4.6 High-Confidence (HC) KL Loss

A pseudo-target is formed by elementwise maxima of view/fused assignments and L2-style target sharpening (`target_l2` in [util.py](util.py)).

$$
y_{max}=\text{target\_l2}(\max(y^{(1)},y^{(2)},y^{(f)}))
$$

Then KL divergence aligns fused prediction to high-confidence target:

$$
\mathcal{L}_{hc}=D_{KL}(y^{(f)}\;\|\;y_{max})
$$

### 4.7 Final Objective in Current Code

The implemented aggregate loss is:

$$
\mathcal{L}=\mathcal{L}_{rec} + (\mathcal{L}_{df}+\mathcal{L}_{ce}) + \mathcal{L}_{mmi} + (\mathcal{L}_{cluster}+\mathcal{L}_{hc})
$$

All coefficients are currently fixed to 1 in [ICDM.py](ICDM.py).

## 5. Inference and Evaluation Flow

Implemented in `icdm.evaluation` in [ICDM.py](ICDM.py).

1. Encode all observed views.
2. For missing-view subsets, initialize from available opposite-view latent and run reverse diffusion loop to recover missing latent codes.
3. Assemble full latent tensors for both views.
4. Fuse with attention.
5. Obtain cluster probabilities via `ClusterProject`, then hard assignments by `argmax`.
6. Evaluate against labels.

Metrics are computed in [evaluation.py](evaluation.py):

- ACC (Hungarian-aligned clustering accuracy)
- NMI
- ARI

AMI and F-score are also available in the utility evaluation output.

## 6. Supported Datasets in This Repository

Data loaders and defaults are defined in [datasets.py](datasets.py) and [configure.py](configure.py).

The run script supports:

- `CUB` (image-text style two-view data)
- `HandWritten` (handwritten multi-view benchmark; two selected views)
- `Multi-Fashion`
- `Synthetic3d`
- `LandUse_21` (loaded multi-view source, training currently uses two selected views)

Each experiment creates artificial incompleteness via an indicator mask (`get_mask`) so some samples are missing one of the two active views.

## 7. Design Choices and Motivation

### 7.1 Why latent diffusion instead of input-space diffusion?

- Lower-dimensional latent vectors reduce denoising complexity and memory usage.
- Autoencoders separate modality-specific feature extraction from stochastic completion.
- Reverse sampling operates in a representation space already shaped for clustering.

### 7.2 Why LayerNorm + SiLU in the modernized blocks?

- `LayerNorm` is batch-size agnostic and stable under irregular observed/missing sample partitions.
- `SiLU` provides smooth nonlinearity beneficial for score/noise regression in diffusion-style objectives.
- The currently active architecture applies this pattern in diffusion, fusion, and clustering heads, and optionally in autoencoder hidden layers.

### 7.3 Why attention-based fusion?

- View quality can vary under missingness and reconstruction uncertainty.
- Learnable per-sample weighting lets the model down-weight less reliable view latents.

### 7.4 Why multi-loss optimization?

No single objective is sufficient:

- reconstruction preserves per-view fidelity,
- diffusion loss trains generative completion dynamics,
- cross-view/instance loss aligns recovered and counterpart latents,
- MI and cluster-consistency terms enforce shared semantics,
- HC KL sharpens assignments using confident pseudo-targets.

## 8. Current Limitations

- Reverse diffusion during training and evaluation is computationally expensive (iterative per timestep).
- Performance is sensitive to scheduler horizon, latent dimensionality, and loss interactions.
- Loss weights are currently hard-coded in the final objective; careful balancing is still required for robust behavior.
- The codebase contains an API mismatch risk between diffusion constructor calls and the active `Unet` signature; this should be reconciled for strict reproducibility.

## 9. Future Improvements

- Cross-view conditional diffusion (explicit conditioning on available view embeddings).
- Shared denoiser across views with view-token conditioning to reduce parameter count.
- Stronger clustering objectives (e.g., prototype consistency, contrastive clustering with memory bank).
- Generalization from fixed two-view training to scalable $N$-view dynamic fusion and completion.

## 10. Code Map

- Core model blocks: [baseModels.py](baseModels.py)
- Training/evaluation orchestration: [ICDM.py](ICDM.py)
- Loss definitions: [loss.py](loss.py)
- Dataset loading: [datasets.py](datasets.py)
- Dataset-specific hyperparameters: [configure.py](configure.py)
- Entry point: [run.py](run.py)
- Metric computation: [evaluation.py](evaluation.py)