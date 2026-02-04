# Training Pipeline Walkthrough: simple_trainer.py

## What Happens When You Run `python simple_trainer.py`

This document explains the complete training pipeline step-by-step for 3D Gaussian Splatting.

---

## High-Level Overview

```
1. Setup & Initialization (once)
   ├── Load dataset (images + camera poses from COLMAP)
   ├── Initialize Gaussians (from sparse point cloud)
   ├── Create optimizers for all parameters
   └── Setup training strategy (densification/pruning)

2. Training Loop (30,000 iterations by default)
   ├── Load one training image
   ├── Rasterize Gaussians → rendered image
   ├── Compute loss (compare to ground truth)
   ├── Backpropagate gradients
   ├── Update parameters with optimizer
   ├── Densify & prune Gaussians (adaptive)
   └── Evaluate & save checkpoints (periodically)

3. Final Evaluation & Rendering
   ├── Evaluate on test set
   └── Render trajectory video
```

---

## Phase 1: Setup & Initialization (Runner.__init__)

### Step 1.1: Parse Arguments and Setup Directories
**File**: Lines 323-348

```python
def __init__(self, local_rank, world_rank, world_size, cfg):
    # Set random seed for reproducibility
    set_random_seed(42 + local_rank)
    
    # Setup output directories
    result_dir/
    ├── ckpts/      # Model checkpoints (.pt files)
    ├── stats/      # Training statistics (JSON)
    ├── renders/    # Rendered images/videos
    ├── ply/        # Exported PLY files
    └── tb/         # Tensorboard logs
```

**What happens:**
- Creates output directories for saving results
- Initializes tensorboard writer for logging
- Sets device (CPU/GPU)

---

### Step 1.2: Load Dataset
**File**: Lines 351-366

```python
# Load COLMAP data
self.parser = Parser(
    data_dir=cfg.data_dir,  # e.g., "data/360_v2/garden"
    factor=cfg.data_factor,  # Downsample factor (1, 2, 4, 8)
    normalize=cfg.normalize_world_space,  # Center & scale scene
    test_every=cfg.test_every,  # Every 8th image for validation
)
```

**What the Parser loads:**

```
data/360_v2/garden/
├── images/                    # Input photos
│   ├── DSC07959.JPG
│   ├── DSC07960.JPG
│   └── ...
└── sparse/0/                  # COLMAP reconstruction
    ├── cameras.bin            # Camera intrinsics (fx, fy, cx, cy)
    ├── images.bin             # Camera poses (rotation + translation)
    └── points3D.bin           # Sparse 3D point cloud
```

**Parser extracts:**
- `parser.points`: Nx3 array of 3D positions from sparse reconstruction
- `parser.points_rgb`: Nx3 array of RGB colors for each point
- `parser.camtoworlds`: Camera-to-world transformation matrices
- `parser.Ks`: Camera intrinsic matrices
- `parser.scene_scale`: Bounding box size of the scene

**Dataset splits:**
- Training set: 7 out of every 8 images (e.g., images 0,1,2,3,4,5,6, skip 7, then 8,9,...)
- Validation set: Every 8th image (e.g., 7, 15, 23, ...)

---

### Step 1.3: Initialize Gaussians
**File**: Lines 389-411, Function at 227-319

```python
self.splats, self.optimizers = create_splats_with_optimizers(
    self.parser,
    init_type=cfg.init_type,  # "sfm" or "random"
    ...
)
```

**What `create_splats_with_optimizers` does:**

#### A. Load Initial Points (lines 250-257)
```python
if init_type == "sfm":
    # Use COLMAP sparse points as initialization
    points = torch.from_numpy(parser.points).float()  # [N, 3]
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()  # [N, 3]
elif init_type == "random":
    # Random initialization in scene bounds
    points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
    rgbs = torch.rand((init_num_pts, 3))
```

**Example numbers:**
- Garden scene: ~30,000-100,000 initial points from COLMAP
- Scene scale: ~3.5 (meters, after normalization)

#### B. Compute Initial Scales (lines 259-262)
```python
# Each Gaussian's size = average distance to 3 nearest neighbors
dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
dist_avg = torch.sqrt(dist2_avg)
scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
```

**Why log space?**
- Scales must be positive during rendering
- Log parameterization: `actual_scale = exp(parameter)`
- Makes optimization easier (unbounded parameter space)

#### C. Initialize Other Parameters (lines 269-271)
```python
quats = torch.rand((N, 4))  # Random rotations (will be normalized)
opacities = torch.logit(torch.full((N,), init_opacity))  # Default 0.1
```

**Why logit space for opacity?**
- Opacities must be in [0, 1] during rendering
- Logit parameterization: `actual_opacity = sigmoid(parameter)`
- Prevents optimization from going outside valid range

#### D. Initialize Colors as Spherical Harmonics (lines 281-289)
```python
colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
colors[:, 0, :] = rgb_to_sh(rgbs)  # 0th degree SH = base color
# Higher degrees start at zero, will be learned
```

**Spherical Harmonics degrees:**
- Degree 0: 1 coefficient (constant color, view-independent)
- Degree 1: 4 coefficients (linear variation)
- Degree 2: 9 coefficients (quadratic variation)
- Degree 3: 16 coefficients (default, rich view-dependent effects)

**Why SH?**
- Compact representation of view-dependent color
- Same Gaussian can look different from different angles
- Important for specular effects and lighting

#### E. Create Learnable Parameters (lines 273-279)
```python
params = [
    ("means", torch.nn.Parameter(points), means_lr * scene_scale),
    ("scales", torch.nn.Parameter(scales), scales_lr),
    ("quats", torch.nn.Parameter(quats), quats_lr),
    ("opacities", torch.nn.Parameter(opacities), opacities_lr),
]
```

**Learning rates (scaled by scene size):**
- means_lr: 1.6e-4 * scene_scale
- scales_lr: 5e-3
- quats_lr: 1e-3
- opacities_lr: 5e-2
- sh0_lr: 2.5e-3 (base color)
- shN_lr: 2.5e-3 / 20 (higher-order SH, lower LR)

#### F. Create Optimizers (lines 308-319)
```python
optimizers = {
    name: torch.optim.Adam(
        [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
        eps=1e-15 / math.sqrt(batch_size),
        betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
    )
    for name, _, lr in params
}
```

**Final parameter counts (example for Garden scene):**
- N = 50,000 Gaussians (initial)
- means: 50,000 × 3 = 150,000 parameters
- scales: 50,000 × 3 = 150,000 parameters
- quats: 50,000 × 4 = 200,000 parameters
- opacities: 50,000 × 1 = 50,000 parameters
- sh0: 50,000 × 1 × 3 = 150,000 parameters
- shN: 50,000 × 15 × 3 = 2,250,000 parameters (for degree 3)
- **Total: ~3 million parameters initially**

---

### Step 1.4: Initialize Training Strategy
**File**: Lines 415-424

```python
self.cfg.strategy.check_sanity(self.splats, self.optimizers)

if isinstance(self.cfg.strategy, DefaultStrategy):
    self.strategy_state = self.cfg.strategy.initialize_state(
        scene_scale=self.scene_scale
    )
```

**Two strategies available:**

#### DefaultStrategy (original 3DGS paper)
- Tracks gradient statistics per Gaussian
- Splits large Gaussians with high gradients
- Clones small Gaussians with high gradients
- Prunes low-opacity Gaussians
- Parameters:
  - `refine_start_iter`: 500 (when to start densification)
  - `refine_stop_iter`: 15,000 (when to stop)
  - `refine_every`: 100 (densification frequency)
  - `reset_alpha_every`: 3000 (reset opacities to prevent floaters)

#### MCMCStrategy (newer approach)
- Probabilistic relocation of Gaussians
- Based on rendering gradients
- More stable for complex scenes

---

## Phase 2: Training Loop (Runner.train())

### Main Loop Structure
**File**: Lines 725-1044

```python
trainloader = torch.utils.data.DataLoader(
    self.trainset,
    batch_size=cfg.batch_size,  # Usually 1
    shuffle=True,
    num_workers=4,
)

for step in range(max_steps):  # 30,000 iterations
    # Load data
    # Render
    # Compute loss
    # Backprop
    # Update parameters
    # Densify/prune
    # Evaluate periodically
```

---

### Step 2.1: Load Training Data
**File**: Lines 744-765

```python
data = next(trainloader_iter)

camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
Ks = data["K"].to(device)  # [1, 3, 3]
pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
image_ids = data["image_id"].to(device)  # [1]
```

**What you get:**
- One random training image per iteration
- Camera pose (4×4 transformation matrix)
- Camera intrinsics (3×3 matrix with focal length, principal point)
- Ground truth RGB image (normalized to [0, 1])
- Image ID (for tracking which image)

**Example dimensions:**
- Image: [1, 546, 980, 3] (height=546, width=980 after downsampling by factor 4)
- Pixels: ~535,000 pixels per image

---

### Step 2.2: Rasterize Gaussians (Render Image)
**File**: Lines 777-795, Function at 556-661

```python
renders, alphas, info = self.rasterize_splats(
    camtoworlds=camtoworlds,
    Ks=Ks,
    width=width,
    height=height,
    sh_degree=sh_degree_to_use,  # Gradually increase during training
    near_plane=cfg.near_plane,  # 0.01
    far_plane=cfg.far_plane,  # 1e10
)
colors = renders[..., 0:3]  # [1, H, W, 3]
```

**Inside `rasterize_splats`:**

#### A. Prepare Gaussian Parameters (lines 570-588)
```python
means = self.splats["means"]  # [N, 3]
quats = self.splats["quats"]  # [N, 4]
scales = torch.exp(self.splats["scales"])  # [N, 3] - convert from log
opacities = torch.sigmoid(self.splats["opacities"])  # [N] - convert from logit
colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
```

#### B. Call Core Rasterization (lines 594-617)
```python
render_colors, render_alphas, info = rasterization(
    means=means,
    quats=quats,
    scales=scales,
    opacities=opacities,
    colors=colors,
    viewmats=torch.linalg.inv(camtoworlds),  # World-to-camera
    Ks=Ks,
    width=width,
    height=height,
    packed=self.cfg.packed,  # Memory-efficient mode
)
```

**What `rasterization()` does (in gsplat/rendering.py):**

1. **Project 3D Gaussians to 2D** (CUDA kernel)
   - Transform means from world space to camera space
   - Apply perspective projection
   - Transform 3D covariance to 2D covariance on image plane
   - Compute 2D bounding box for each Gaussian

2. **Tile-based Culling** (CUDA kernel)
   - Divide image into 16×16 pixel tiles
   - Find which Gaussians overlap which tiles
   - Build per-tile lists of Gaussians

3. **Sort by Depth** (CUDA kernel)
   - Sort Gaussians within each tile from front to back
   - Uses radix sort for efficiency

4. **Rasterization** (CUDA kernel)
   - For each pixel in parallel:
     - Iterate through Gaussians in depth order
     - Evaluate 2D Gaussian: `G(x) = exp(-0.5 * (x-μ)ᵀ Σ⁻¹ (x-μ))`
     - Alpha blend colors:
       ```
       T = 1.0  (transmittance)
       C = 0.0  (accumulated color)
       for each Gaussian i:
           α_i = opacity_i * G_i(pixel)
           C += T * α_i * color_i
           T *= (1 - α_i)
           if T < threshold: break  (early stopping)
       ```
   - Output: RGB color per pixel

**info dictionary contains:**
- `gaussian_ids`: Which Gaussians were rendered
- `radii`: Projected 2D radius of each Gaussian (for gradient tracking)
- `num_tiles_hit`: How many tiles each Gaussian overlaps

---

### Step 2.3: Compute Loss
**File**: Lines 809-843

```python
# L1 loss
l1loss = F.l1_loss(colors, pixels)  # Mean absolute error

# SSIM loss (structural similarity)
ssimloss = 1.0 - fused_ssim(
    colors.permute(0, 3, 1, 2),
    pixels.permute(0, 3, 1, 2),
    padding="valid"
)

# Combined loss (default: 80% L1 + 20% SSIM)
loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
```

**Why both losses?**
- **L1**: Measures per-pixel color accuracy
- **SSIM**: Measures perceptual similarity (structure, contrast, luminance)
- SSIM helps with high-frequency details and prevents blurring

**Optional losses:**
- Depth loss (if depth data available): Lines 815-833
- Opacity regularization: Encourages sparsity
- Scale regularization: Prevents Gaussians from growing too large

**Typical loss values:**
- Initial: ~0.15-0.20
- After 1000 steps: ~0.05-0.08
- Converged (30k steps): ~0.02-0.03

---

### Step 2.4: Pre-Backward Strategy Hook
**File**: Lines 801-807

```python
self.cfg.strategy.step_pre_backward(
    params=self.splats,
    optimizers=self.optimizers,
    state=self.strategy_state,
    step=step,
    info=info,
)
```

**What this does (in DefaultStrategy):**
- Accumulates gradient statistics per Gaussian
- Tracks how often each Gaussian is rendered
- Tracks magnitude of gradients (for splitting/cloning decisions)

---

### Step 2.5: Backpropagation
**File**: Line 851

```python
loss.backward()
```

**What happens:**
- PyTorch autograd computes gradients for all parameters
- Calls custom CUDA backward kernels for rasterization
- Gradients flow through:
  ```
  loss → colors → rasterization → [means, scales, quats, opacities, SH colors]
  ```

**Gradient computation (in CUDA):**
- For each pixel with gradient dL/dC:
  - Backtrack through alpha blending to get dL/dG for each Gaussian
  - Chain rule through 2D Gaussian evaluation
  - Chain rule through projection to get dL/d(3D params)

**Memory note:**
- Gradients are sparse (only for rendered Gaussians)
- If `sparse_grad=True`, converts to sparse tensor (lines 958-970)

---

### Step 2.6: Optimize Parameters
**File**: Lines 983-999

```python
# Update all Gaussian parameters
for optimizer in self.optimizers.values():
    if cfg.visible_adam:
        optimizer.step(visibility_mask)  # Only update visible Gaussians
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)

# Update learning rates
for scheduler in schedulers:
    scheduler.step()
```

**What gets updated:**
- means: Move Gaussians to reduce error
- scales: Grow/shrink Gaussians
- quats: Rotate Gaussians
- opacities: Make more/less transparent
- sh0, shN: Adjust colors

**Learning rate schedule:**
- Exponential decay for means: `lr = initial_lr * 0.01^(step/max_steps)`
- Helps with stability as training progresses

---

### Step 2.7: Post-Backward Strategy (Densification & Pruning)
**File**: Lines 1002-1021

```python
self.cfg.strategy.step_post_backward(
    params=self.splats,
    optimizers=self.optimizers,
    state=self.strategy_state,
    step=step,
    info=info,
)
```

**What happens (every 100 steps, after step 500):**

#### A. Identify Gaussians to Modify
```python
# High-gradient Gaussians need refinement
is_grad_high = grad_2d_accum > grad_2d_threshold  # e.g., > 0.0002
n_splits = is_grad_high.sum()

# Large Gaussians should be split
is_large = scales.max(dim=-1)[0] > percent_dense * scene_scale  # e.g., > 0.01

# Small Gaussians should be cloned
is_small = scales.max(dim=-1)[0] <= percent_dense * scene_scale
```

#### B. Split Large Gaussians
```python
# For each large, high-gradient Gaussian:
#   1. Create 2 new Gaussians
#   2. Position them slightly offset along principal axis
#   3. Make them smaller (scale /= 1.6)
#   4. Reduce opacity (opacity /= 2)
#   5. Remove original
```

**Why split?**
- Large Gaussians can't capture fine details
- Splitting creates more resolution where needed
- Driven by gradient: high error → needs refinement

#### C. Clone Small Gaussians
```python
# For each small, high-gradient Gaussian:
#   1. Create exact copy
#   2. Keep same position/size/color
```

**Why clone?**
- Need more Gaussians in high-detail areas
- Cloning is simpler than splitting for small Gaussians
- Subsequent optimization will move them apart

#### D. Prune Low-Quality Gaussians
```python
# Remove Gaussians that are:
#   1. Nearly transparent (opacity < 0.005)
#   2. Very large (in camera space)
#   3. In front of camera (for some variants)
```

**Why prune?**
- Reduces memory and computation
- Removes "floaters" (artifacts in empty space)
- Keeps model compact

**Typical numbers:**
- Start: 50,000 Gaussians
- Peak (around step 10,000): 200,000-500,000 Gaussians
- End (after pruning): 100,000-300,000 Gaussians

#### E. Reset Optimizers
```python
# After modifying Gaussians (split/clone/prune):
#   1. Add new parameters to optimizer state
#   2. Remove deleted parameters
#   3. Inherit momentum from parent Gaussians
```

---

### Step 2.8: Periodic Evaluation
**File**: Lines 1024-1026

```python
if step in [i - 1 for i in cfg.eval_steps]:  # Default: [6999, 29999]
    self.eval(step)
    self.render_traj(step)
```

---

## Phase 3: Evaluation (Runner.eval())

### Step 3.1: Evaluate on Validation Set
**File**: Lines 1046-1137

```python
@torch.no_grad()
def eval(self, step: int, stage: str = "val"):
    # No gradients needed for evaluation
    
    for data in valset:
        # Render each validation image
        renders, alphas, info = self.rasterize_splats(...)
        
        # Compute metrics
        psnr = compute_psnr(renders, ground_truth)
        ssim = compute_ssim(renders, ground_truth)
        lpips = compute_lpips(renders, ground_truth)
        
        # Save rendered images
```

**Metrics:**
- **PSNR** (Peak Signal-to-Noise Ratio): 
  - Higher is better (30+ is good, 35+ is excellent)
  - Measures pixel-wise accuracy
  
- **SSIM** (Structural Similarity):
  - Range [0, 1], higher is better (>0.95 is excellent)
  - Measures perceptual quality
  
- **LPIPS** (Learned Perceptual Image Patch Similarity):
  - Lower is better (<0.1 is good)
  - Uses neural network to measure perceptual distance

**Example results (Garden scene after 30k steps):**
- PSNR: 27-28 dB
- SSIM: 0.86-0.88
- LPIPS: 0.10-0.15

---

### Step 3.2: Render Trajectory Video
**File**: Lines 1147-1271

```python
def render_traj(self, step: int):
    # Generate camera path (spiral, interpolated, etc.)
    camtoworlds_all = generate_interpolated_path(
        trainset.camtoworlds,
        n_interp=240  # 240 frames = 8 seconds at 30 fps
    )
    
    # Render each frame
    for camtoworld in camtoworlds_all:
        frame = self.rasterize_splats(camtoworld, ...)
        frames.append(frame)
    
    # Save as MP4 video
    imageio.mimwrite(f"renders/traj_{step}.mp4", frames, fps=30)
```

**Camera paths:**
- `interp`: Interpolate between training views
- `spiral`: Spiral around scene center
- `ellipse_z`: Elliptical path with varying height

---

## Phase 4: Save Checkpoints

### During Training
**File**: Lines 893-921

```python
if step in cfg.save_steps or step == max_steps - 1:
    data = {
        "step": step,
        "splats": self.splats.state_dict(),  # All Gaussian parameters
    }
    torch.save(data, f"ckpts/ckpt_{step}_rank{world_rank}.pt")
```

**Checkpoint contains:**
- Current iteration number
- All Gaussian parameters (means, scales, quats, opacities, colors)
- Optimizer state (optional)
- Camera adjustments (if pose optimization enabled)

**File sizes:**
- 100k Gaussians: ~100-200 MB per checkpoint
- 500k Gaussians: ~500 MB - 1 GB per checkpoint

---

### Export PLY Format
**File**: Lines 922-955

```python
if step in cfg.ply_steps and cfg.save_ply:
    export_splats(
        means=self.splats["means"],
        scales=torch.exp(self.splats["scales"]),  # Convert from log
        quats=self.splats["quats"],
        opacities=torch.sigmoid(self.splats["opacities"]),  # Convert from logit
        sh0=self.splats["sh0"],
        shN=self.splats["shN"],
        format="ply",
        save_to=f"ply/point_cloud_{step}.ply",
    )
```

**PLY format:**
- Standard 3D point cloud format
- Can be viewed in external tools (CloudCompare, MeshLab, etc.)
- Contains positions, colors, and Gaussian parameters

---

## Summary: One Training Iteration in Detail

```
Step N (e.g., step 1000):

1. Load Data (1-5ms)
   - Random training image
   - Camera pose & intrinsics
   
2. Rasterize (10-30ms)
   - Project 50,000 Gaussians to 2D
   - Tile-based culling
   - Alpha-blend to create image
   - Output: 546×980×3 rendered image
   
3. Compute Loss (1-2ms)
   - L1 + SSIM loss
   - Typical value: 0.05
   
4. Backward (15-40ms)
   - Compute gradients via CUDA kernels
   - Sparse gradients for ~5000 visible Gaussians
   
5. Optimizer Step (1-2ms)
   - Update all parameters
   - Apply learning rate schedule
   
6. Densification (every 100 steps, 5-20ms)
   - Split large Gaussians: +500 new
   - Clone small Gaussians: +200 new
   - Prune transparent: -300 removed
   - New total: 50,400 Gaussians
   
Total time per iteration: 30-100ms
Training time for 30k steps: 15-50 minutes (1 GPU)
```

---

## Key Hyperparameters and Their Effects

### Initialization
- `init_type`: "sfm" (use COLMAP points) vs "random"
- `init_num_pts`: Number of initial Gaussians (if random)
- `init_opa`: Initial opacity (0.1 = mostly transparent)
- `init_scale`: Scale multiplier for initial sizes

### Training
- `max_steps`: Total iterations (30,000 default)
- `batch_size`: Images per iteration (usually 1)
- `data_factor`: Image downsampling (4 = 1/4 resolution)

### Loss Weights
- `ssim_lambda`: SSIM vs L1 balance (0.2 = 20% SSIM, 80% L1)
- `opacity_reg`: Regularization for sparsity
- `scale_reg`: Regularization for small Gaussians

### Densification (DefaultStrategy)
- `refine_start_iter`: When to start (500)
- `refine_stop_iter`: When to stop (15,000)
- `refine_every`: Frequency (100)
- `densify_grad_threshold`: Gradient threshold for refinement (0.0002)
- `densify_size_threshold`: Size threshold for splitting (0.01)
- `prune_opa`: Opacity threshold for pruning (0.005)

### Spherical Harmonics
- `sh_degree`: Max SH degree (3 = 16 coefficients)
- `sh_degree_interval`: Gradually increase every N steps (1000)

---

## Memory Usage Breakdown

**For Garden scene (546×980 image, 200k Gaussians):**

- Gaussian parameters: ~500 MB
  - means: 200k × 3 × 4 bytes = 2.4 MB
  - scales: 200k × 3 × 4 bytes = 2.4 MB
  - quats: 200k × 4 × 4 bytes = 3.2 MB
  - opacities: 200k × 4 bytes = 0.8 MB
  - SH colors: 200k × 16 × 3 × 4 bytes = 38.4 MB
  
- Optimizer state (Adam): ~1 GB
  - Momentum and variance for all parameters
  
- Forward pass: ~500 MB
  - Projected Gaussians, tile lists, depth sorting
  
- Backward pass: ~800 MB
  - Intermediate gradients, alpha blending data
  
- Image tensors: ~10 MB
  - Input image, rendered image, ground truth

**Total: ~3-4 GB GPU memory**

---

## Typical Training Timeline

```
Step 0: 50,000 Gaussians, loss=0.18, PSNR=15dB
  - Random colors, rough shapes

Step 1000: 80,000 Gaussians, loss=0.08, PSNR=22dB
  - Basic scene structure visible
  - Colors mostly correct
  - Blurry details

Step 5000: 250,000 Gaussians, loss=0.04, PSNR=26dB
  - Most details captured
  - Some artifacts in complex areas
  - View-dependent effects starting

Step 15,000: 300,000 Gaussians, loss=0.03, PSNR=27.5dB
  - High quality rendering
  - Densification stopped
  - Only pruning continues

Step 30,000: 200,000 Gaussians, loss=0.025, PSNR=28dB
  - Converged
  - Pruned unnecessary Gaussians
  - Ready for export
```

---

## Output Files

After training completes, you'll have:

```
results/garden/
├── ckpts/
│   ├── ckpt_6999_rank0.pt      # Checkpoint at 7k steps
│   └── ckpt_29999_rank0.pt     # Final checkpoint
├── renders/
│   ├── val_step6999/           # Validation renders
│   │   ├── 000.png
│   │   ├── 001.png
│   │   └── ...
│   ├── traj_interp_step29999.mp4  # Trajectory video
│   └── val_metrics_step29999.json  # Metrics (PSNR, SSIM, LPIPS)
├── stats/
│   ├── train_step6999_rank0.json
│   └── train_step29999_rank0.json
├── ply/
│   └── point_cloud_29999.ply   # Exportable 3D model
└── tb/                          # Tensorboard logs
    └── events.out.tfevents.*
```

**View results:**
```bash
# View tensorboard
tensorboard --logdir results/garden/tb

# View PLY
cloudcompare results/garden/ply/point_cloud_29999.ply

# Load checkpoint
python simple_trainer.py --ckpt results/garden/ckpts/ckpt_29999_rank0.pt
```

---

## Next Steps

Now that you understand the full pipeline, you can:

1. **Modify initialization**: Use your custom Gaussians (see IMPLEMENTATION_GUIDE.md)
2. **Tune hyperparameters**: Adjust learning rates, densification thresholds
3. **Add custom losses**: Depth supervision, semantic guidance
4. **Modify rendering**: Add depth output, normal maps
5. **Change strategy**: Try MCMC instead of default densification

The key insight: training is an iterative refinement process where Gaussians adapt to minimize rendering error while growing/splitting/pruning to handle details at multiple scales.
