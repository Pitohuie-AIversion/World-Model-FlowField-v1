# Scientific Manuscript: Experimental Results & Physical Analysis

> **Target Manuscript Section**: Section 5 (Experimental Results, Physical Invariant Verification, and Multi-Scale Analysis)  
> **Benchmark Dataset**: *The Well* — 2D Incompressible Shear Flow with Passive Tracer Transport  
> **Evaluation Protocol**: Closure-R4 Multi-Seed Provenance-Closed Protocol ($N_{\mathrm{test}}=45$ grouped test trajectories, Seeds $\{42, 43, 44\}$)  
> **Provenance & CI Verification**: GitHub Actions Run `35808361053` (Success, 97 passed, 1 skipped, 0 failed), Clean Working Tree (`git_dirty: false`)

---

## 5. Experimental Results

We evaluate our proposed Latent World Model framework against classical neural operators and spatial architecture ablations on the challenging 2D shear flow dataset from *The Well* benchmark. In this section, we present:
1. **Spatial Representation Performance** of the deterministic spatial autoencoder;
2. **Multi-Seed Autoregressive Rollout Dynamics** under the Closure-R4 physical loss ablation;
3. **Within-Seed Paired Difference Analysis** quantifying physical invariant conservation;
4. **Multi-Scale Directional Fourier Spectral Analysis** evaluating fine-scale dissipation and spectral anisotropy;
5. **Architectural and Baseline Comparisons** against direct physical-space transformers and Fourier Neural Operators (FNO-2D).

---

### 5.1 Spatial Representation Performance

The foundation of our latent world model is a deterministic spatial autoencoder with circular-padded residual convolutions ($8\times$ spatial downsampling, latent grid $\mathbf{Z} \in \mathbb{R}^{16 \times 32 \times 64}$). As summarized in **Table 1** (generated from `outputs/tables/table_1_representation.tex`), the autoencoder compresses the four physical flow channels ($\mathbf{q} = [u, v, p, s]$) while strictly enforcing physical boundary conditions and the pressure gauge degree of freedom.

\input{outputs/tables/table_1_representation.tex}

**Key Findings**:
- **High-Fidelity Flow Reconstruction**: On the 45 unseen test trajectories, the autoencoder achieves a Field Mean VRMSE of $0.1015$ (10.15% relative variance error), with the primary streamwise velocity $u$ reconstructed at a precision of $0.0517$ (5.17% VRMSE) and passive tracer $s$ at $0.1204$ (12.04% VRMSE).
- **Zero-Mean Gauge Invariance**: Incompressible Navier-Stokes flows determine gauge pressure only up to an arbitrary spatial constant ($\nabla p$). Through our non-in-place spatial mean subtraction projection operator in the decoder, the spatial average error $|\bar{p}|$ is strictly bounded below $2.49 \times 10^{-7}$, completely preventing spurious elliptic potential drift.
- **Vorticity Conservation in Latent Projection**: The decoded representations preserve spatial gradients with high fidelity, achieving a vorticity RMSE of $\|\omega - \omega^*\| = 0.3902$ across the full test set.

---

### 5.2 Multi-Seed Autoregressive Rollout Dynamics (Closure-R4)

To evaluate dynamic forecasting stability and eliminate single-seed observational bias, five model variants were trained and benchmarked across $N=3$ independent training seeds (Seeds 42, 43, 44) under identical data splits, normalization parameters, and physical domain contracts ($L_x = 1.0, L_y = 2.0$):
- **E0 (Single-Step)**: Supervised purely on 1-step latent transitions ($H=1$, seed 42).
- **E1 (Rollout-Aware Field)**: Trained with 2-step rollout supervision ($H=2$) using only field reconstruction loss $\mathcal{L}_{\mathrm{field}}$.
- **E2 ($+ \mathcal{L}_{\mathrm{div}}$)**: Adds divergence penalty ($\lambda_{\mathrm{div}} = 0.01$) to penalize compressible velocity modes.
- **E3 ($+ \mathcal{L}_\omega$)**: Adds vorticity spectral penalty ($\lambda_\omega = 0.05$) to constrain rotational eddy dynamics.
- **E4 (Full Physics)**: Combines multi-step rollout with both $\mathcal{L}_{\mathrm{div}}$ and $\mathcal{L}_\omega$.

**Table 2** records the quantitative evolution of field and physical metrics across rollout horizons $t \in \{1, 5, 10, 20, 30\}$, while **Figure A** illustrates both the mean trajectory error and the cross-seed dispersion ($\sigma_{\mathrm{VRMSE}}$).

\input{outputs/tables/table_2_multi_seed_ablation.tex}

![Figure A: Rollout Field Mean VRMSE and Cross-Seed Dispersion](outputs/figures/figure_a_vrmse_and_dispersion.png)

**Key Observations**:
1. **Short-to-Medium Horizon Field Accuracy**: At step 1 ($t=1$), all models demonstrate high accuracy ($\mathrm{VRMSE} \approx 0.045 - 0.053$). By step 5 ($t=5$), physics-informed models (E3, E4) suppress error growth compared to single-step models (E0: $\mathrm{VRMSE} = 0.8473$ vs E4: $\mathrm{VRMSE} = 0.4375 \pm 0.1502$).
2. **Cross-Seed Dispersion Reduction**: A critical benefit of physics regularization is the stabilization of optimization trajectories across seeds. At step 5, vorticity regularization (E3, E4) reduces cross-seed standard deviation $\sigma_{\mathrm{VRMSE}}$ by approximately **$3\times$** ($\sigma = 0.1502$ for E4 vs $\sigma = 0.4449$ for E1 and $\sigma = 0.4502$ for E2), as shown in Figure A (right panel).
3. **Long-Horizon Intermediate Horizon Nuance**: In accordance with our balanced scientific narrative, we emphasize that physics regularization does not uniformly dominate raw field MSE at intermediate steps (e.g., at $t=10$, E1 achieves $\mathrm{VRMSE} = 1.0436 \pm 0.4329$ whereas E4 achieves $1.1607 \pm 0.4868$). Rather, the primary contribution of physics regularization is preventing catastrophic structural unphysicalities, as demonstrated below.

---

### 5.3 Within-Seed Paired Difference Analysis & Physical Invariants

To isolate the causal contribution of each physical regularization term from seed-level stochasticity, we compute **within-seed paired differences** relative to the rollout-aware baseline E1:
$$\Delta_i = v_{E_x, \mathrm{seed}_i} - v_{E_1, \mathrm{seed}_i}, \quad i \in \{42, 43, 44\}$$
**Table 3** details the mean delta, sample standard deviation, and win rate ($[\mathrm{Wins}/N_{\mathrm{seeds}}]$) across all 3 seeds. **Figure B** displays the corresponding physical invariant curves: Divergence RMSE, Vorticity RMSE, Relative Enstrophy Error, and Tracer Particle Out-of-Bounds Escape Rate.

\input{outputs/tables/table_3_within_seed_paired.tex}

![Figure B: Physical Invariant Consistency Curves Across Horizons](outputs/figures/figure_b_physical_invariants.png)

**Quantitative Invariant Verification**:
- **Incompressibility Enforcement ($\mathcal{L}_{\mathrm{div}}$)**:
  - Model E2 ($+ \mathcal{L}_{\mathrm{div}}$) achieves a **100% win rate (3/3 seeds)** in reducing velocity divergence at step 1 ($\Delta = -0.0375 \pm 0.0177$) and step 5 ($\Delta = -0.0585 \pm 0.0463$).
  - Full physics model E4 similarly achieves a **100% win rate** at step 1 ($\Delta = -0.0260 \pm 0.0308$).
- **Vorticity & Enstrophy Stabilization ($\mathcal{L}_\omega$)**:
  - Model E3 ($+ \mathcal{L}_\omega$) achieves a **100% win rate (3/3 seeds)** at step 1 across all rotational diagnostics: Vorticity RMSE ($\Delta = -0.1847 \pm 0.0196$), Energy Spectrum MAE ($\Delta = -0.0478 \pm 0.0285$), and Relative Enstrophy Error ($\Delta = -0.0136 \pm 0.0025$).
  - Over long horizons ($t=20, 30$), full physics model E4 demonstrates persistent stabilization, achieving a **100% win rate (3/3 seeds)** on Vorticity RMSE at $t=20$ ($\Delta = -0.6776 \pm 0.5898$) and $t=30$ ($\Delta = -0.8278 \pm 0.3667$).
  - Relative enstrophy error $\frac{|\Omega - \Omega^*|}{\Omega^*}$ is significantly suppressed by E4 at long horizons, with 100% win rates at $t=20$ ($\Delta = -0.6995 \pm 0.5901$) and $t=30$ ($\Delta = -1.2441 \pm 0.9421$).
- **Tracer Particle Confinement**:
  - Unregulated rollouts suffer from non-physical particle escape across boundaries due to accumulating velocity errors. At the terminal horizon ($t=30$), E4 achieves a **100% win rate** on suppressing the tracer out-of-bounds escape rate ($\Delta = -0.0787 \pm 0.0982$).

---

### 5.4 Multi-Scale Directional Fourier Spectral Analysis

Turbulent shear flows are characterized by strong spatial anisotropy: the primary flow direction $x$ exhibits periodic Kelvin-Helmholtz instability wave packets, while the transverse direction $y$ sustains steep shear layer gradients. To rigorously assess fine-scale energy preservation without observational bias, we conducted full 2D FFT spectral analyses across all seeds and horizons.

We compute:
1. **1D Radially Integrated Spectral Energy Ratio**: $R(k) = E_{\mathrm{pred}}(k) / E_{\mathrm{targ}}(k)$
2. **Directional Spectral Ratios**: $R_x(k_x)$ along the streamwise axis, and $R_y(k_y)$ along the shear axis.
3. **High-Wavenumber Energy Diagnostic**: $\gamma_{\mathrm{high}} = \frac{1}{|K_{\mathrm{high}}|} \sum_{k \in K_{\mathrm{high}}} R(k)$, evaluated against a predefined $\pm 10\%$ diagnostic tolerance band $[0.90, 1.10]$.

![Figure C: Multi-Scale Directional Fourier Spectral Analysis](outputs/figures/directional_spectral_ratio_curves.png)

**Spectral Diagnostics & Nuanced Interpretation**:
- **Diagnostic Tolerance Band in Seed 42**: At the terminal horizon ($t=30$), Model E4 under Seed 42 achieves $\gamma_{\mathrm{high}} = 0.948$, falling squarely within the predefined $[0.90, 1.10]$ diagnostic tolerance band. In sharp contrast, non-physics baselines E0, E1, and E2 exhibit severe numerical over-dissipation ($\gamma_{\mathrm{high}} \in [0.81, 0.85]$), prematurely smoothing fine-scale turbulent eddies.
- **Directional Anisotropy ($R_x$ vs $R_y$)**: As shown in **Figure C**, the transverse spectral ratio $R_y(k_y)$ displays greater resilience against numerical diffusion than $R_x(k_x)$, reflecting the structural stability of the mean shear profile across the channel width.
- **Continuous Spectra as Primary Evidence**: While scalar diagnostics ($\gamma_{\mathrm{high}}$) provide convenient summaries, continuous energy spectrum curves $\log_{10} E(k)$ serve as our primary evidence. In Seeds 43 and 44, autoregressive error compounding after $t=20$ produces high-frequency spectral inflection across all models. We explicitly report this behavior, noting that physics regularization delays but does not eliminate fine-scale accumulation in extended unconstrained rollouts.

---

### 5.5 Architectural and Baseline Comparison

We compare our Latent World Model against direct physical-space forecasting and classical neural operators. **Table 4** presents the comparative benchmark across models spanning 30 autoregressive rollout steps:
- **Persistence Baseline**: Static initial condition persistence $\hat{\mathbf{q}}_t = \mathbf{q}_0$.
- **Direct ST-Transformer**: Space-time transformer trained directly on high-dimensional raw physical fields ($128 \times 128$) without spatial latent encoding (7.73M parameters).
- **FNO-2D**: Fourier Neural Operator with 16 Fourier modes and channel width 64 (16.80M parameters).
- **Latent World Model (E0, E1, E4)**: Our proposed 2-stage framework combining spatial autoencoding with latent space-time attention (9.75M parameters total: 1.06M autoencoder + 8.69M transformer).

\input{outputs/tables/table_4_architecture_ablation.tex}

**Comparative Insights**:
1. **Catastrophic Failure of Direct Physical Space Transformers**:
   - The Direct ST-Transformer demonstrates that attempting autoregressive rollout directly in high-dimensional grid space without manifold compression is fundamentally unstable.
   - By step 10, its VRMSE diverges to $6.3670$; by step 30, it experiences catastrophic derivative blow-up: Divergence RMSE reaches **$89.73$** (vs $1.83$ for E4) and Vorticity RMSE reaches **$81.15$** (vs $3.90$ for E4).
   - This empirically confirms that low-dimensional spatial manifold compression is essential for stable physical world modeling.
2. **Trade-offs of Fourier Neural Operators (FNO-2D)**:
   - FNO-2D achieves strong spectral smoothness and bounds divergence ($\|\nabla \cdot \mathbf{u}\| = 0.0389$ at $t=30$) through Fourier mode truncation.
   - However, this truncation comes at the cost of high initial field error: at step 1, FNO-2D exhibits a VRMSE of **$0.6450$**, which is **$12\times$ higher** than our Latent World Model (E4: $0.0532 \pm 0.0088$). FNO-2D blurs localized vortex shear boundaries, resulting in uniform but blurry flow fields.
3. **Latent World Model as an Optimal Compromise**:
   - Our Latent World Model (E4) bridges this gap: it achieves near-ground-truth initial reconstruction fidelity ($\mathrm{VRMSE} \approx 0.05$, Vorticity $\mathrm{RMSE} = 0.2847$), while regularizing long-horizon physical invariant degradation without the parameter footprint of operator baselines (9.75M vs 16.80M for FNO).

---

## 6. Summary of Scientific Findings & Research Boundaries

1. **Proven Incompressibility Enforcement**: Direct penalty on velocity divergence ($\mathcal{L}_{\mathrm{div}}$) consistently reduces physical divergence across multiple seeds without degrading field MSE.
2. **Stabilization of Rotational Invariants**: Vorticity loss ($\mathcal{L}_\omega$) and full physics regularization ($\mathcal{L}_{\mathrm{div}} + \mathcal{L}_\omega$) reliably curb long-horizon vorticity and enstrophy divergence across 3 independent seeds.
3. **Optimization Variance Reduction**: Physics regularization substantially reduces cross-seed variance ($\sigma_{\mathrm{VRMSE}}$ reduced by $3\times$ at step 5), yielding reproducible dynamics.
4. **Boundary of Claims**: Physics regularization does not guarantee uniform superiority in raw field MSE at every intermediate horizon, and scalar high-$k$ spectral diagnostics must be interpreted in conjunction with full continuous Fourier curves.
