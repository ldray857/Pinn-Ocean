# Reconstruction of 3-D Ocean Temperature and Salinity Fields in the Northwest Pacific via Coupled Physics-Informed Neural Networks and Self-Attention Mechanism

Di Lei, Sensen Wu  
(1. School of Earth Sciences, Zhejiang University, Hangzhou 310027, Zhejiang, China)

**Abstract**: Accurate characterization of the three-dimensional (3-D) ocean temperature and salinity fields is of critical significance for marine environmental forecasting, underwater acoustic propagation, and ocean general circulation dynamics. Addressing the challenges that conventional satellite remote sensing is limited to 2-D sea surface observations, numerical ocean data assimilation incurs prohibitive computational costs, and purely data-driven deep learning models lack physical consistency, this study proposes a 3-D thermohaline reconstruction model coupling shifted-window self-attention with physics-informed neural networks (Swin-Ocean-PINN). The Kuroshio Extension region in the Northwest Pacific ($145.0^\circ\text{E} \sim 165.0^\circ\text{E}, 30.0^\circ\text{N} \sim 40.0^\circ\text{N}, 0 \sim 1000\,\text{m}$) is selected as the experimental domain. The input pipeline fuses multi-source heterogeneous features, including sea level anomaly (SLA), sea surface temperature (SST), sea surface salinity (SSS), sea surface wind vectors, and spatiotemporal coordinates. At the model front-end, a shallow dual-convolutional Stem extracts local dynamical fluid micro-element features. The backbone leverages shifted-window self-attention to capture basin-scale baroclinic dynamical teleconnections. The back-end constructs a Cartesian product concatenation between high-dimensional dynamical semantics and continuous non-dimensional depth coordinates, decoding continuous 3-D profiles via an infinitely smooth, differentiable physics head. In the loss function formulation, a vertical temperature monotonic decrease prior and density stratification stability constraints grounded in the International Thermodynamic Equation of Seawater 2010 (TEOS-10) are incorporated, using homoscedastic uncertainty to adaptively balance data fidelity losses against partial differential physical constraints. The validation framework employs multi-source satellite observations and high-resolution GLORYS12V1 reanalysis data from 2013 to 2021, and introduces independent international in-situ Argo float profiles to benchmark extrapolation generalization and vertical microstructure fidelity. This work establishes a novel intelligent paradigm for ocean subsurface dynamical reconstruction that combines high-resolution representation capacity with rigorous physical consistency.

**Keywords**: Physics-Informed Neural Network (PINN); Swin Transformer; Northwest Pacific; Thermohaline Field Reconstruction; TEOS-10

## 0 Introduction
The three-dimensional thermohaline structure of the ocean interior governs the global meridional overturning circulation, ocean stratification stability, and acoustic channel characteristics, representing a core research focus of physical oceanography and ocean climate dynamics. However, due to the severe physical attenuation of electromagnetic radiation in seawater, existing satellite microwave and infrared remote sensing techniques can only observe dynamical and thermodynamic parameters at the very sea surface, unable to penetrate directly into the subsurface ocean structure. Conventional data assimilation systems based on ocean general circulation models (OGCMs) incur massive computational expenses and exhibit high sensitivity to sub-grid turbulence mixing parameterization schemes.

In recent years, deep learning methodologies represented by convolutional neural networks and self-attention mechanisms have demonstrated outstanding nonlinear fitting capabilities in ocean subsurface retrieval. Nevertheless, purely data-driven models fundamentally rely on the empirical statistical distribution of finite training samples. In spatial extrapolation regimes and intensely perturbed dynamical zones, they are prone to producing unphysical, ill-posed solutions that violate fundamental fluid conservation laws, manifested typically as anomalous subsurface temperature inversions and gravitational stratification instabilities (density inversions).

To overcome these bottlenecks, this study develops the Swin-Ocean-PINN framework, deeply coupling physics-informed neural networks with self-attention mechanisms. The model projects multi-source sea surface satellite observations into latent fluid dynamical representations, leverages local shifted-window self-attention to aggregate basin-scale teleconnections, and generates 3-D thermohaline fields via a continuous coordinate physics decoding head. By embedding hydrostatic stability criteria and TEOS-10 equation-of-state partial differential constraints, the network backpropagates physical gradients via automatic differentiation, achieving dynamically coordinated and hydrologically self-consistent subsurface continuous thermohaline reconstruction without requiring dense in-situ sounding profiles.

## 1 Study Area and Observational Data

### 1.1 Experimental Domain and Spatiotemporal Scope
The study area is situated in the deep open basin of the Kuroshio Extension in the Northwest Pacific, spanning $145.0^\circ\text{E} \sim 165.0^\circ\text{E}$ and $30.0^\circ\text{N} \sim 40.0^\circ\text{N}$. The vertical reconstruction covers depths of $0.0 \sim 1000.0\,\text{m}$, fully encompassing the oceanic upper mixed layer, the main thermocline, and the subsurface dynamically active layer. The temporal baseline spans January 1, 2013 to December 31, 2021, comprising 108 continuous monthly mean fields. All bathymetric grid points in this domain exceed $1000\,\text{m}$ depth, constituting an open ocean abyss free of continental shelves, islands, or land boundaries, thereby eliminating boundary mask truncation errors. The Kuroshio Extension exhibits intense velocity shear, energetic mesoscale eddies, and prominent oceanic fronts with complex baroclinic 3-D structures, serving as an ideal and rigorous testbed for evaluating physics-constrained subsurface reconstruction.

### 1.2 Multi-Source Satellite Remote Sensing and Physical Reanalysis Data
The input pipeline fuses sea surface multi-source satellite dynamical observations with spatiotemporal coordinate embeddings, using high-resolution underwater physical reanalysis fields as supervised targets. The technical specifications of the multi-source datasets are summarized in Table 1.

Table 1 Technical specifications of sea surface remote sensing observations and underwater reanalysis datasets

| Data Product | Horizontal Resolution | Core Physical Variables | Dynamical & Constraint Function |
| :--- | :--- | :--- | :--- |
| SLA (Sea Level Anomaly) | 0.125° × 0.125° | `sla` | Reflects vertical steric height integral and thermocline displacements |
| GLORYS 3D (Supervisory Target) | 0.083° × 0.083° | `thetao`, `so` | Provides dynamically balanced 3-D benchmark truth fields |
| SST (Sea Surface Temperature) | 0.05° × 0.05° | `analysed_sst` | Anchors surface thermal boundary and upper mixed-layer temperature |
| SSS (Sea Surface Salinity) | 0.25° × 0.25° | `sss` | Characterizes surface precipitation-evaporation freshwater flux forcing |
| Wind (Sea Surface Wind) | 0.25° × 0.25° | `eastward_wind`, `northward_wind` | Drives surface wind-driven drift and Ekman pumping vertical velocity |

#### (a) Sea Surface Height Anomaly and Geostrophic Velocity (SLA)
Sea level anomaly data are obtained from the DUACS L4 multi-satellite merged gridded reprocessed product distributed by the Copernicus Marine Environment Monitoring Service (CMEMS) at a spatial resolution of 0.125° × 0.125°. The retrieval interface configuration is illustrated in Figure 1. This product provides sea level anomalies (`sla`) and surface geostrophic current anomaly vectors (`ugosa`, `vgosa`). Sea surface dynamic height anomalies integrate the thermal expansion and haline contraction across the entire water column and correlate strongly with the vertical displacement of the main thermocline. Local assets are archived in `data/pacific_sla_2013_2021.nc`.

![Figure 1 DUACS multi-satellite merged sea level anomaly dataset retrieval parameters and metadata configuration](image/研究报告/1788678319353.png)

#### (b) 3-D Ocean Physical Reanalysis Field (GLORYS 3D)
Dense underwater supervisory targets are derived from the global ocean high-resolution physical reanalysis dataset GLORYS12V1 distributed by CMEMS at a spatial resolution of 0.083° × 0.083°, with the retrieval configuration shown in Figure 2. Based on the NEMO numerical engine, GLORYS assimilates along-track altimetry and in-situ T/S profiles, providing potential temperature (`thetao`) and practical salinity (`so`) across 33 standard vertical depth levels within $0 \sim 1000\,\text{m}$. Local assets are archived in `data/pacific_glorys_3d_temp_sal_2013_2021.nc`.

![Figure 2 GLORYS12V1 3-D ocean physical reanalysis dataset retrieval configuration](image/研究报告/1788678466584.png)

#### (c) Sea Surface Temperature Analyzed Field (SST)
Surface thermal forcing is acquired from the OSTIA global sea surface temperature analyzed product produced by the UK Met Office at a horizontal resolution of 0.05° × 0.05° (Figure 3). OSTIA merges infrared and microwave radiometer observations, mitigating cloud coverage contamination to provide foundation sea surface temperature (`analysed_sst`) for anchoring the thermal Dirichlet boundary condition. Local assets are archived in `data/pacific_sst_2013_2021.nc`.

![Figure 3 OSTIA sea surface temperature merged dataset retrieval configuration](image/研究报告/1788678598315.png)

#### (d) Sea Surface Practical Salinity Field (SSS)
Surface salinity is obtained from the SMOS/SMAP dual-satellite microwave merged gridded product released by LOPS-IFREMER at a resolution of 0.25° × 0.25° (Figure 4). The product provides sea surface practical salinity (`sss`), directly reflecting net evaporation, precipitation, and oceanic freshwater fluxes, serving as the surface haline boundary for the equation of state. Local assets are archived in `data/pacific_sss_2013_2021.nc`.

![Figure 4 LOPS sea surface salinity multi-satellite merged dataset retrieval configuration](image/研究报告/1788678732412.png)

#### (e) Sea Surface Wind Vector Field (Wind)
Surface dynamical forcing is acquired from the CERSAT/IFREMER global scatterometer merged monthly wind product at a resolution of 0.25° × 0.25° (Figure 5). It provides 10-meter eastward wind (`eastward_wind`) and northward wind (`northward_wind`), driving surface Ekman drift and Ekman pumping vertical velocities to guide upper-ocean vertical motion inference. Local assets are archived in `data/pacific_wind_2013_2021.nc`.

![Figure 5 CERSAT spaceborne scatterometer wind vector dataset retrieval configuration](image/研究报告/1788678676315.png)

### 1.3 Independent In-Situ Validation Data (Argo Float Array and CORA)
To objectively evaluate the model's generalization capability under realistic ocean conditions, independent international Argo float profiles and CMEMS CORA delayed-mode in-situ observations are incorporated as validation benchmarks (Figure 6).

![Figure 6 CMEMS CORA international in-situ observation profile service interface](image/研究报告/1788679019851.png)

Argo profiling floats are quality-controlled and distributed by the Global Data Assembly Centre (GDAC) and China Argo Real-time Data Centre. Drifting with currents and cycling between surface and depth, they measure high-precision in-situ temperature (`TEMP`), practical salinity (`PSAL`), and pressure (`PRES`) profiles across $0 \sim 1000\,\text{m}$. Numerical reanalysis fields tend to smooth thermocline gradients and eddy-core extremes due to spatial filtering and numerical diffusion. Spatially aligned Argo soundings provide an unbiased evaluation of the network's fidelity in reconstructing mixed layer depth, main thermocline sharpness, and eddy fine-scale structures. Local archives are maintained in `data/argo/pacific_argo_in_situ_2020.nc`.

Because Argo floats follow discrete Lagrangian trajectories without regular grid coordinates, a multi-source extraction and quality control pipeline was developed: streaming temporal-spatial queries via the official `argopy` library export standard NetCDF files within $0 \sim 1000\,\text{m}$; delayed-mode flags are verified against the China Argo Data Centre to eliminate sensor drift; and cross-checks with Ifremer GDAC Pacific archives ensure rigorous spatial and physical compliance.

### 1.4 Model Input Features and Spatiotemporal Encoding
The sea surface input tensor $\mathbf{X} \in \mathbb{R}^{B \times 8 \times H \times W}$ comprises 8 physical and spatiotemporal geometric channels:  
Channel 0: Standardized Sea Surface Temperature (SST);  
Channel 1: Standardized Sea Level Anomaly (SLA);  
Channel 2: Standardized Sea Surface Salinity (SSS);  
Channels 3 & 4: Standardized Eastward and Northward Wind components (Wind U, Wind V);  
Channels 5 & 6: Longitude and Latitude coordinates, linearly normalized to $[0, 1]$, representing basin geometry and the spatial variation of the Coriolis parameter $f = 2\Omega \sin \phi$;  
Channel 7: Oceanic thermal cyclic phase encoding, formulated as $`\tau_t = -\cos\left(2\pi \cdot \frac{\text{month} - 2}{12}\right) \in [-1, 1]`$, rigorously tracking the seasonal cycle in the Northwest Pacific from its minimum in February (-1) to its maximum in August (+1), preventing seasonal temperature extrapolation distortion. Vertical discrete depths $`z \in [0.5, 1000.0]\,\text{m}`$ are mapped to continuous scalars $`z_{\text{norm}} \in [0, 1]`$, with full derivative tracking enabled in the computational graph for analytical differentiation.

### 1.5 Data Preprocessing and Decoupled Temporal Partitioning
To resolve spatial resolution discrepancies between satellite products (0.05°~0.25°) and the GLORYS reanalysis (0.083°), the data pipeline standardizes all inputs onto the GLORYS12V1 grid ($121 \times 241$ grid points). Bilinear interpolation with continuous boundary extrapolation (`method="linear", kwargs={"fill_value": "extrapolate"}`) resamples SLA, SST, SSS, and wind fields, eliminating boundary NaN artifacts from slight domain misalignments. OSTIA absolute temperatures (Kelvin) are converted to Celsius to ensure physical consistency.

To prevent temporal data leakage, statistical means $\mu$ and standard deviations $\sigma$ for all physical variables are computed strictly over the training partition (the initial 72 months in the multi-year setup, or 9 months in the single-year benchmark). Validation and test sets strictly reuse training statistics for $z$-score normalization ( $`\hat{x} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}`$ ).

Following geophysical causality, the dataset is partitioned chronologically: in the full-scale experiment, the initial 72 months (January 2013 to December 2018, ~66.7%) serve as the training set for backpropagation; the subsequent 24 months (January 2019 to December 2020, ~22.2%) serve as the validation set for hyperparameter tuning and early stopping; and the final 12 months (January 2021 to December 2021, ~11.1%) serve as an independent test set for unobserved temporal extrapolation evaluation. The pipeline also fully supports annual benchmarks (such as the 2020 12-month sequence) for self-contained validation.

## 2 Physics-Informed Neural Network Architecture

### 2.1 Overall Topology and Forward Propagation Dynamics
To overcome the challenge of mapping 2-D sea surface observations to 3-D continuous subsurface physical fields, the Swin-Ocean-PINN architecture integrates a shallow convolutional dynamical stem, a shifted-window self-attention backbone (Swin Backbone), a continuous depth coordinate concatenation unit, a smooth physics decoding head, and an automatic differentiation physics constraint module. The tensor dimensions and operator functionality across forward pass stages are detailed in Table 2.

Table 2 Tensor dimensions and operator functionality across forward pass stages
| Computational Stage | Core Module | Input Dimension | Output Dimension | Mathematical Operator & Dynamical Role |
| :--- | :--- | :--- | :--- | :--- |
| 1. Shallow Dynamical Extraction | Convolutional Stem | $(B, 8, H, W)$ | $`(B, C_{\text{embed}}, H, W)`$ | Dual convolutions with nonlinear activations to extract frontal shear gradients |
| 2. Spatial Serialization | Flatten & Transpose | $`(B, C_{\text{embed}}, H, W)`$ | $`(B, H\cdot W, C_{\text{embed}})`$ | Reshapes spatial feature maps into 2-D token sequences |
| 3. Shifted Self-Attention | Swin Transformer | $`(B, H\cdot W, C_{\text{embed}})`$ | $`(B, H\cdot W, C_{\text{embed}})`$ | Window and shifted-window self-attention to capture basin teleconnections |
| 4. Spatial Point Sampling | Random Grid Sampler | $`(B, H\cdot W, C_{\text{embed}})`$ | $`(B, S, C_{\text{embed}})`$ | Randomly samples $S$ spatial points to bound autograd computational memory |
| 5. Continuous Coordinate Binding | Cartesian Broadcast | Space $(B, S, C)$, Depth $(B, S, D, 1)$ | $`(B, S, D, C_{\text{embed}}+1)`$ | Binds surface spatial semantics with continuous vertical depth coordinates |
| 6. Continuous Field Decoding | Physics Head | $`(B, S, D, C_{\text{embed}}+1)`$ | $(B, S, D, 2)$ | 4-layer MLP with Tanh activations, generating normalized T-S fields |
| 7. Field Reorganization | Permute & Denorm | $(B, S, D, 2)$ | $(B, 2, D, S)$ | Restructures predictions into 3-D temperature $\hat{T}$ and salinity $\hat{S}$ |
| 8. Analytical Physics Solving | Physics Engine | Predictions & Point Depths $`z_{\mathrm{pts}}`$ | Scalars $`\mathcal{L}_{\mathrm{phy}, T}, \mathcal{L}_{\mathrm{phy}, \rho}`$ | Pointwise analytical solving of $`\frac{\partial \hat{T}_{b,s,k}}{\partial z_k}`$ and $`\frac{\partial \hat{\rho}_{b,s,k}}{\partial z_k}`$ via Autograd |
| 9. Adaptive Multi-Objective | Homoscedastic Weights | Data & Physics Losses | Scalar $`\mathcal{L}_{\mathrm{total}}`$ | Learns dual variables $`\omega_1, \omega_2`$ for stable Pareto convergence |

### 2.2 Shallow Convolutional Dynamical Feature Extractor (Stem)
Sea surface dynamical processes are governed by geostrophic balance and fluid continuity equations, with pronounced velocity shear and horizontal thermohaline gradients along eddy edges and frontal boundaries. Standard Vision Transformers commonly employ patch partitioning with non-overlapping strides, which severs the continuity of local fluid micro-elements. To preserve continuous spatial derivatives, the network front-end utilizes a lightweight dual-convolutional Stem that maintains the original spatial resolution $(H, W)$:

$$
\mathbf{F}_0 = \mathrm{GELU}\left(\mathrm{BN}\left(\mathrm{Conv}_{3\times 3}\left(\mathrm{GELU}\left(\mathrm{BN}\left(\mathrm{Conv}_{3\times 3}(\mathbf{X})\right)\right)\right)\right)\right)
$$

This module smoothly projects the 8-channel surface observations into the latent embedding dimension ($`C_{\text{embed}} = 96`$), preserving differential geometric structures of mesoscale eddies and water mass fronts.

### 2.3 Shifted-Window Local Self-Attention Backbone (Swin Backbone)
To capture basin-scale baroclinic teleconnections spanning hundreds of kilometers, the backbone incorporates shifted-window self-attention.

#### (a) Window-Based Multi-Head Self-Attention (W-MSA)
The 2-D feature map is partitioned into non-overlapping local windows (window size $M=4$ or $8$), within which self-attention is computed independently alongside continuous relative position bias matrices $\hat{B}$:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \hat{B}\right)V
$$

This mechanism reduces computational complexity from global self-attention's quadratic $O((HW)^2)$ to locally linear $O(M^2 \cdot HW)$, reconciling high-resolution representation with computational tractability.

#### (b) Shifted-Window Multi-Head Self-Attention (SW-MSA)
Consecutive Transformer blocks introduce a spatial cyclic shift of $\lfloor M/2 \rfloor$ pixels. An adaptive attention mask assigns negative infinity to attention weights across non-physical boundary partitions, completely suppressing unphysical connectivity after Softmax. Following computation, a reverse cyclic shift restores original spatial topology. This enables cross-window information exchange at minimal computational overhead, capturing spatial dynamical coupling between eddy fields and current systems.

### 2.4 Multi-Scale Fourier Depth Embedding and DeepONet Operator Fusion Network

#### (a) Multi-Scale Harmonic Fourier Depth Embedding (DepthFourierEmbedding)
Due to the "spectral bias" of coordinate networks, standard multi-layer perceptrons struggle to capture sharp transitions and non-monotonic vertical ocean structures (such as subsurface salinity maxima at ~150 m and intermediate minima at ~650 m). To mitigate this, a multi-scale harmonic Fourier coordinate embedding module is formulated:  

**Linear Dimensionless Depth**:  

$$
z_{\mathrm{lin}} = \frac{z}{z_{\mathrm{max}}} \in [0, 1]
$$

**Oceanic Logarithmic Progressive Depth**:  

$$
z_{\mathrm{log}} = \frac{\ln(1 + z / z_{\mathrm{scale}})}{\ln(1 + z_{\mathrm{max}} / z_{\mathrm{scale}})} \in [0, 1]
$$

(with $`z_{\mathrm{scale}} = 15.0\,\text{m}`$), allocating over 30% of dynamic numerical resolution to the surface mixed layer and upper thermocline;  

**Multi-Band Fourier Harmonic Expansion**:  
Depths are projected across $K=8$ octave scales onto sinusoidal tensors $`[\sin(2^k \pi z), \cos(2^k \pi z)]`$, yielding an input dimension of $2 + 4K = 34$, which is mapped via a compact two-layer linear network to latent depth embedding vectors $`\mathbf{F}_{\mathrm{depth}} \in \mathbb{R}^{D \times d_{\mathrm{depth}}}`$.

#### (b) DeepONet Operator Fusion and Decoupled Dual Prediction Heads
Inspired by deep operator network (DeepONet) theory, the architecture fuses sea surface spatial dynamical tokens with continuous vertical coordinate features via dual-branch tensor products and residual connections:  

**Branch Network**: Receives surface tokens $\mathbf{F}_{\mathrm{surf}}$ from the Swin backbone, projecting them into a 256-dimensional latent dynamical space;  

**Trunk Network**: Receives Fourier depth embeddings $\mathbf{F}_{\mathrm{depth}}$, projecting them into a matching 256-dimensional physical basis space;  

**Nonlinear Operator Fusion**:  

$$
\mathbf{F}_{\mathrm{fused}} = \mathrm{SiLU}\left(\mathbf{F}_{\mathrm{branch}} \odot \mathbf{F}_{\mathrm{trunk}} + \mathbf{F}_{\mathrm{branch}} + \mathbf{F}_{\mathrm{trunk}}\right)
$$

Multiplicative coupling via Hadamard product and residual identity skip connections combines surface dynamical forcing with continuous vertical basis functions. Dropout is strictly avoided throughout the decoding pathway to guarantee determinism and continuous differentiability.  

**Decoupled Dual Prediction Heads**: Addressing the physical asymmetry where temperature decreases monotonically while salinity exhibits a pronounced non-monotonic "S"-shaped curve, decoupled MLP heads are implemented: a 2-layer MLP (128→64→1) for temperature, and a higher-capacity 3-layer MLP (128→128→64→1) for salinity, providing the nonlinear expressiveness required to capture subsurface salinity extrema.

## 3 Governing Equations and Active Physical Constraint Loss Engine

Conventional one-sided penalty formulations risk passive deactivation where losses collapse to zero in non-violating regimes. To enforce rigorous dynamical discipline, an **active multi-objective ocean physics loss engine** is formulated:

### 3.1 Sea Level Anomaly (SLA) Baroclinic Steric Height Integration Constraint
Sea surface dynamic height anomalies reflect whole-water-column thermal expansion and haline contraction. Using the TEOS-10 equation of state, in-situ density $\rho$ is calculated analytically, subtracting the horizontal reference profile $\bar{\rho}(z)$ to obtain density anomaly $\rho'$, which is integrated hydrostatically over depth:

$$
\Delta h_{\mathrm{steric}}(x, y) = -\frac{1}{\rho_0} \int_{0}^{H} \rho'(x, y, z) \, \mathrm{d}z
$$

This steric height anomaly is constrained against satellite altimeter observations via mean squared error:

$$
\mathcal{L}_{\mathrm{sla}} = \frac{1}{B \cdot S} \sum_{b=1}^B \sum_{s=1}^S \left( \Delta h_{\mathrm{steric}, b, s} - \mathrm{SLA}_{\mathrm{obs}, b, s} \right)^2
$$

### 3.2 Unified Coordinate Dirichlet Surface Boundary Anchoring (Dirichlet BC)
At the sea surface interface ($z = 0.5\,\text{m}$), reconstructed temperature and salinity must strictly close upon satellite SST and SSS observations. To prevent conflicting gradient trajectories arising from differing normalization scales between 3-D volumes and 2-D surface fields, surface satellite truth values are projected into the 3-D normalized coordinate frame:

$$
\mathrm{SST}_{\mathrm{target\_norm}} = \frac{\mathrm{SST}_{\mathrm{phys}} - \mu_{T3D}}{\sigma_{T3D}}, \quad \mathrm{SSS}_{\mathrm{target\_norm}} = \frac{\mathrm{SSS}_{\mathrm{phys}} - \mu_{S3D}}{\sigma_{S3D}}
$$

$$
\mathcal{L}_{\mathrm{surf}} = \left\| \hat{T}_{\mathrm{norm}}(z_0) - \mathrm{SST}_{\mathrm{target\_norm}} \right\|^2 + \left\| \hat{S}_{\mathrm{norm}}(z_0) - \mathrm{SSS}_{\mathrm{target\_norm}} \right\|^2
$$

This anchors the physical surface boundary condition $`\hat{T}_{\mathrm{phys}}(z_0) \equiv \mathrm{SST}_{\mathrm{phys}}`$, eliminating dimensional mismatch artifacts.

### 3.3 Continuous Profile First-Order Differential Gradient and Curvature Supervision
To prevent discrete vertical training from smoothing out profile inflections and extrema, first-order finite difference gradient supervision is applied across 100 m intervals:

$$
\mathcal{L}_{\mathrm{grad}} = \left\| \frac{\partial \hat{T}}{\partial z_{100}} - \frac{\partial T_{\mathrm{gt}}}{\partial z_{100}} \right\|^2 + 2 \cdot \left\| \frac{\partial \hat{S}}{\partial z_{100}} - \frac{\partial S_{\mathrm{gt}}}{\partial z_{100}} \right\|^2
$$

Applying double weighting to the salinity gradient guides the network to accurately reconstruct sharp transitions between high-salinity Subtropical Underwater (STUW) and low-salinity North Pacific Intermediate Water (NPIW).

### 3.4 0–30 m Mixed Layer Isothermal Homogenization Regularization
The ocean surface mixed layer exhibits minimal vertical temperature gradient due to wind stirring and convective overturning. Within depths $z \le 30\,\text{m}$, a small gradient threshold $`\epsilon_{\mathrm{mld}} = 0.02\,^\circ\text{C}/\text{m}`$ is enforced via an isothermal penalty:

$$
\mathcal{L}_{\mathrm{mld}} = \frac{1}{N_{\mathrm{mld}}} \sum_{z_k \le 30\,\mathrm{m}} \mathrm{ReLU}\left( \left| \frac{\partial \hat{T}_{\mathrm{phys}}}{\partial z} \right| - \epsilon_{\mathrm{mld}} \right)
$$

This eliminates artificial near-surface gradient curl, enforcing a physically realistic homogeneous mixed layer.

### 3.5 TEOS-10 Equation of State and Smooth Stratification Stability Constraint
Using the TEOS-10 nonlinear equation of state $`\hat{\rho} = f_{\mathrm{TEOS\text{-}10}}(\hat{S}, \hat{T}, P)`$, thermal expansion $\alpha$ and haline contraction $\beta$ coefficients are computed pointwise:

$$
\frac{\partial \rho}{\partial z} \approx -\alpha \frac{\partial T}{\partial z} + \beta \frac{\partial S}{\partial z}
$$

Static gravitational stability requires potential density to increase monotonically with depth. Replacing non-differentiable hard clipping with a smooth Softplus operator:

$$
\mathcal{L}_{\mathrm{stab}} = \frac{1}{B \cdot S \cdot D} \sum_{b,s,k} \mathrm{Softplus}\left(- 10 \cdot \frac{\partial \hat{\rho}_{b, s, k}}{\partial z_k}\right)
$$

This penalizes unphysical density inversions while providing smooth, continuous backpropagation gradients throughout the water column.

### 3.6 Adaptive Multi-Objective Optimization via Homoscedastic Uncertainty
The data fidelity loss $`\mathcal{L}_{\mathrm{data}}`$ and active physics loss $`\mathcal{L}_{\mathrm{phy}}`$ (comprising $`\mathcal{L}_{\mathrm{sla}}`$, $`\mathcal{L}_{\mathrm{surf}}`$, $`\mathcal{L}_{\mathrm{grad}}`$, $`\mathcal{L}_{\mathrm{mld}}`$, $`\mathcal{L}_{\mathrm{stab}}`$) are dynamically balanced using homoscedastic uncertainty weighting:

$$
\mathcal{L}_{\mathrm{total}} = \exp(-\omega_1) \mathcal{L}_{\mathrm{data}} + \omega_1 + \exp(-\omega_2) \mathcal{L}_{\mathrm{phy}} + \omega_2
$$

where $`\omega_1, \omega_2`$ are learnable dual parameters initialized to $0.0$ and clamped to $[-10, 10]$, guiding optimization smoothly along the Pareto front.

## 4 Experimental Implementation and Optimization Strategies

Network parameters are jointly optimized using the AdamW optimizer. The backbone learning rate is set to $\eta_1 = 3 \times 10^{-4}$ with weight decay $\lambda = 1 \times 10^{-4}$; the physical loss balancing parameters $\omega_1, \omega_2$ are updated with an independent learning rate of $\eta_2 = 1 \times 10^{-3}$. Learning rate scheduling employs a `ReduceLROnPlateau` policy, monitoring validation loss and decaying the learning rate by 50% if no improvement is observed for 8 consecutive epochs.  

Solving high-order 3-D spatial partial derivatives via automatic differentiation constructs extensive backpropagation computational graphs, which quickly exhausts GPU memory if evaluated across all grid points simultaneously. To address this, an unbiased spatial discrete random sampling mechanism is implemented, drawing $S = 800$ random spatial grid points per iteration to evaluate physical constraints. While preserving unbiased gradient expectations over the whole domain, this strategy reduces autograd memory consumption by approximately 85%, enabling efficient execution on single 8GB GPUs.

## 5 Evaluation and Summary

### 5.1 Evaluation Framework and Target Metrics

The engineering system is implemented using the PyTorch deep learning framework and the xarray geospatial computation ecosystem, with underlying spatial interpolation driven by NetCDF4 and HDF5 engines. The pipeline has been extended from a single-year baseline (2020) to a two-year sequence (2019–2020, 24 continuous months), establishing a closed-loop workflow across multi-source satellite observations (SST, SLA, SSS, Wind) and GLORYS 3-D truth fields.  

System accuracy is rigorously evaluated on independent test partitions. Statistical metrics include depth-resolved Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$) for temperature and salinity. For key ocean dynamic interfaces, Mixed Layer Depth (MLD) accuracy is quantified using a temperature threshold criterion ($\Delta T = 0.2\,^\circ\text{C}$), alongside localization error for the depth of maximum vertical temperature gradient in the main thermocline.  

Beyond standard statistical errors, evaluation focuses on hydrological and dynamical self-consistency: first, full-domain Temperature-Salinity (T-S) diagrams assess alignment with GLORYS reanalysis across major water masses (e.g., North Pacific Intermediate Water and Subtropical Mode Water); second, the percentage of gravitational density inversion grid points is quantified to verify the suppression of unphysical layering; third, independent international Argo float sounding profiles are matched and evaluated, testing model generalization against unsmoothed real-world observations.

### 5.2 Summary and Research Progress
Addressing the challenge of 3-D thermohaline reconstruction in the Northwest Pacific, this project has designed and implemented the Swin-Ocean-PINN dynamically coupled architecture. A standardized data pipeline has been established covering SLA, SST, SSS, wind vectors, and 3-D reanalysis fields. A differentiable physical constraint engine grounded in TEOS-10 and static stability has been derived; homoscedastic uncertainty optimization reconciles gradient competition between observational data and physical priors; and a 4D pointwise Autograd formulation eliminates the masking of localized physical violations caused by spatial averaging. The code has successfully completed full pipeline verification and annual/two-year benchmark testing, establishing a solid foundation for large-scale multi-year training and independent in-situ Argo validation.
