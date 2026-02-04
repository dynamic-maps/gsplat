# Implementation Guide: Custom Initialization for gsplat

## Goal 1: Train from Custom 3D Gaussian Parameters

Instead of initializing from a sparse point cloud (SfM), you want to provide your own pre-defined 3D Gaussians with specific positions, scales, rotations, colors, and opacities.

## Goal 2: Train 2D Gaussians from Custom Dataset

Train 2D Gaussian Splatting variant on your own images and camera data.

---

## Implementation 1: Custom 3D Gaussian Initialization

### What You Need to Understand

**Key Files:**
1. `examples/simple_trainer.py` (lines 225-280) - Contains `create_splats_with_optimizers()` function
2. `examples/datasets/colmap.py` - How datasets and parsers work
3. `gsplat/rendering.py` - How Gaussians are rendered

**Key Concepts:**
- Each Gaussian has: **means** (position), **scales** (size), **quats** (rotation), **opacities**, **colors** (SH coefficients)
- Parameters are stored as `torch.nn.Parameter` in a `ParameterDict`
- Scales and opacities are in log/logit space for optimization

---

### Step-by-Step Implementation

#### Step 1: Prepare Your Gaussian Data

Create a file that stores your custom Gaussians. Format options:

**Option A: NumPy .npz file**
```python
import numpy as np

# Your Gaussian parameters
means = np.random.randn(10000, 3)  # [N, 3] - positions
scales = np.ones((10000, 3)) * 0.01  # [N, 3] - log scales (x, y, z)
quats = np.array([[1, 0, 0, 0]] * 10000)  # [N, 4] - quaternions (w, x, y, z)
opacities = np.ones(10000) * 0.5  # [N] - opacity values [0, 1]
colors_rgb = np.random.rand(10000, 3)  # [N, 3] - RGB colors [0, 1]

# Save to file
np.savez(
    'my_gaussians.npz',
    means=means,
    scales=scales,
    quats=quats,
    opacities=opacities,
    colors_rgb=colors_rgb
)
```

**Option B: PyTorch .pt file**
```python
import torch

gaussians = {
    'means': torch.randn(10000, 3),
    'scales': torch.ones(10000, 3) * 0.01,
    'quats': torch.tensor([[1, 0, 0, 0]] * 10000, dtype=torch.float),
    'opacities': torch.ones(10000) * 0.5,
    'colors_rgb': torch.rand(10000, 3)
}

torch.save(gaussians, 'my_gaussians.pt')
```

**Option C: PLY file** (standard format)
You can export from other tools like Blender/MeshLab in PLY format with point colors.

---

#### Step 2: Create Custom Initialization Function

Create a new file: `examples/custom_init.py`

```python
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from utils import rgb_to_sh


def load_gaussians_from_file(
    filepath: str,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Load Gaussian parameters from a file.
    
    Args:
        filepath: Path to .npz, .pt, or .ply file
        device: Device to load tensors to
        
    Returns:
        Dictionary with keys: means, scales, quats, opacities, colors_rgb
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        gaussians = {
            'means': torch.from_numpy(data['means']).float(),
            'scales': torch.from_numpy(data['scales']).float(),
            'quats': torch.from_numpy(data['quats']).float(),
            'opacities': torch.from_numpy(data['opacities']).float(),
            'colors_rgb': torch.from_numpy(data['colors_rgb']).float(),
        }
    elif filepath.endswith('.pt'):
        gaussians = torch.load(filepath, map_location='cpu')
    elif filepath.endswith('.ply'):
        # You'll need to implement PLY parsing
        # See plyfile library: from plyfile import PlyData
        gaussians = load_from_ply(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    # Move to device
    gaussians = {k: v.to(device) for k, v in gaussians.items()}
    
    return gaussians


def load_from_ply(filepath: str) -> Dict[str, torch.Tensor]:
    """Load point cloud from PLY file."""
    from plyfile import PlyData
    
    plydata = PlyData.read(filepath)
    vertices = plydata['vertex']
    
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Try to load colors if available
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.ones_like(positions)  # Default white
    
    N = positions.shape[0]
    
    return {
        'means': torch.from_numpy(positions).float(),
        'colors_rgb': torch.from_numpy(colors).float(),
        # Defaults for other parameters
        'scales': torch.ones(N, 3) * 0.01,
        'quats': torch.tensor([[1, 0, 0, 0]] * N, dtype=torch.float),
        'opacities': torch.ones(N) * 0.5,
    }


def create_splats_from_custom_gaussians(
    gaussian_file: str,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Create splats and optimizers from custom Gaussian file.
    
    This replaces the create_splats_with_optimizers function from simple_trainer.py
    but initializes from your custom Gaussians instead of SfM points.
    """
    import math
    
    # Load your custom Gaussians
    gaussians = load_gaussians_from_file(gaussian_file, device=device)
    
    means = gaussians['means']  # [N, 3]
    colors_rgb = gaussians['colors_rgb']  # [N, 3]
    N = means.shape[0]
    
    # Process scales: should be in log space for optimization
    if 'scales' in gaussians:
        scales = gaussians['scales']
        # If scales are in linear space, convert to log space
        if scales.min() > 0:  # Assume linear if all positive
            scales = torch.log(scales)
    else:
        # Default: compute from nearest neighbors
        from utils import knn
        dist2_avg = (knn(means, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)
    
    # Process quaternions: should be normalized
    if 'quats' in gaussians:
        quats = gaussians['quats']
        quats = F.normalize(quats, dim=-1)  # Ensure normalized
    else:
        quats = torch.rand((N, 4))  # Random rotation
        quats = F.normalize(quats, dim=-1)
    
    # Process opacities: should be in logit space for optimization
    if 'opacities' in gaussians:
        opacities = gaussians['opacities']
        # Convert from [0, 1] to logit space
        opacities = torch.clamp(opacities, 0.001, 0.999)  # Avoid inf
        opacities = torch.logit(opacities)
    else:
        opacities = torch.logit(torch.full((N,), 0.1))
    
    # Setup parameters
    params = [
        ("means", torch.nn.Parameter(means), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]
    
    # Setup colors as spherical harmonics
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(colors_rgb)  # Convert RGB to SH
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    
    # Create parameter dict and optimizers
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    
    return splats, optimizers
```

---

#### Step 3: Modify the Trainer to Use Custom Init

Create a modified trainer: `examples/train_with_custom_init.py`

```python
#!/usr/bin/env python3
"""
Train Gaussian Splatting with custom initialization.

Usage:
    python train_with_custom_init.py \
        --gaussian_file my_gaussians.npz \
        --data_dir data/360_v2/garden \
        --result_dir results/custom_init
"""

# Copy most of simple_trainer.py and modify the Config class and Runner.__init__

from simple_trainer import *  # Import everything from simple_trainer
from custom_init import create_splats_from_custom_gaussians


@dataclass
class CustomConfig(Config):
    """Extended config with custom Gaussian initialization."""
    
    # Path to custom Gaussian file (.npz, .pt, or .ply)
    gaussian_file: Optional[str] = None
    
    # Override init_type to add 'custom' option
    init_type: str = "sfm"  # Options: "sfm", "random", "custom"


class CustomRunner(Runner):
    """Modified Runner that supports custom Gaussian initialization."""
    
    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: CustomConfig):
        # Copy the parent init but modify the splat creation part
        set_random_seed(42 + local_rank)
        
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        # Setup directories (same as parent)
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")
        
        # Load dataset (same as parent)
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
        )
        self.trainset = Dataset(self.parser, split="train", patch_size=cfg.patch_size)
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)
        
        # CREATE SPLATS WITH CUSTOM INITIALIZATION
        if cfg.init_type == "custom":
            if cfg.gaussian_file is None:
                raise ValueError("Must provide --gaussian_file when using init_type=custom")
            
            print(f"Loading custom Gaussians from: {cfg.gaussian_file}")
            self.splats, self.optimizers = create_splats_from_custom_gaussians(
                gaussian_file=cfg.gaussian_file,
                means_lr=cfg.means_lr,
                scales_lr=cfg.scales_lr,
                opacities_lr=cfg.opacities_lr,
                quats_lr=cfg.quats_lr,
                sh0_lr=cfg.sh0_lr,
                shN_lr=cfg.shN_lr,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                sparse_grad=cfg.sparse_grad,
                batch_size=cfg.batch_size,
                device=self.device,
            )
        else:
            # Use standard initialization (sfm or random)
            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                means_lr=cfg.means_lr,
                scales_lr=cfg.scales_lr,
                opacities_lr=cfg.opacities_lr,
                quats_lr=cfg.quats_lr,
                sh0_lr=cfg.sh0_lr,
                shN_lr=cfg.shN_lr,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                sparse_grad=cfg.sparse_grad,
                visible_adam=cfg.visible_adam,
                batch_size=cfg.batch_size,
                feature_dim=None,
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
            )
        
        print(f"Model initialized. Number of Gaussians: {len(self.splats['means'])}")
        
        # Continue with rest of initialization (strategy, etc.)
        # ... copy remaining code from parent __init__ ...
        
        # Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()


def main():
    """Entry point."""
    cfg = tyro.cli(CustomConfig)
    
    # Run training
    cli(main_fn=lambda local_rank, world_rank, world_size: 
        CustomRunner(local_rank, world_rank, world_size, cfg).train(),
        backend="nccl")


if __name__ == "__main__":
    main()
```

---

#### Step 4: Run Training

```bash
# First, create your Gaussian file (see Step 1)
python -c "
import numpy as np
means = np.random.randn(10000, 3)
scales = np.ones((10000, 3)) * 0.01
quats = np.tile([1, 0, 0, 0], (10000, 1))
opacities = np.ones(10000) * 0.5
colors_rgb = np.random.rand(10000, 3)
np.savez('my_gaussians.npz', means=means, scales=scales, quats=quats, 
         opacities=opacities, colors_rgb=colors_rgb)
"

# Train with your custom Gaussians
cd examples
python train_with_custom_init.py \
    --init_type custom \
    --gaussian_file ../my_gaussians.npz \
    --data_dir data/360_v2/garden \
    --result_dir results/custom_init
```

---

## Implementation 2: Train 2D Gaussians from Custom Dataset

### What You Need to Understand

**Key Files:**
1. `examples/simple_trainer_2dgs.py` - 2D Gaussian trainer
2. `examples/datasets/colmap.py` - Dataset loader
3. `gsplat/rendering.py` - `rasterization_2dgs` function

**Key Concepts:**
- 2DGS uses flat Gaussians aligned with surfaces
- Requires RGB images + camera parameters (intrinsics + extrinsics)
- Camera data typically comes from COLMAP structure-from-motion

---

### Step-by-Step Implementation

#### Step 1: Prepare Your Dataset

Your dataset should have this structure:
```
my_dataset/
├── images/              # RGB images
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── sparse/              # COLMAP sparse reconstruction
│   └── 0/
│       ├── cameras.bin  # Camera intrinsics
│       ├── images.bin   # Camera extrinsics
│       └── points3D.bin # Sparse point cloud
```

**Option A: Use COLMAP to generate camera data**

```bash
# Install COLMAP first: https://colmap.github.io/install.html

# Run COLMAP automatic reconstruction
colmap automatic_reconstructor \
    --workspace_path my_dataset \
    --image_path my_dataset/images

# This creates my_dataset/sparse/0/ with camera data
```

**Option B: Create camera data programmatically**

If you already know camera parameters (e.g., from synthetic data):

```python
"""
Create COLMAP-format camera data from known parameters.
"""
import numpy as np
from pathlib import Path

def create_colmap_data(
    output_dir: Path,
    image_names: list,
    intrinsics: dict,  # {'fx': float, 'fy': float, 'cx': float, 'cy': float}
    extrinsics: list,  # List of 4x4 transformation matrices
    image_width: int,
    image_height: int,
):
    """
    Create COLMAP binary files from known camera parameters.
    
    You'll need to use the COLMAP Python bindings or write binary files directly.
    See: https://github.com/colmap/colmap/blob/master/scripts/python/read_write_model.py
    """
    from read_write_model import (
        write_cameras_binary,
        write_images_binary,
        write_points3D_binary,
        Camera,
        Image,
        Point3D
    )
    
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Create camera
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=image_width,
            height=image_height,
            params=np.array([
                intrinsics['fx'],
                intrinsics['fy'],
                intrinsics['cx'],
                intrinsics['cy']
            ])
        )
    }
    
    # Create images (camera poses)
    images = {}
    for i, (name, ext) in enumerate(zip(image_names, extrinsics)):
        # Convert 4x4 matrix to quaternion + translation
        R = ext[:3, :3]
        t = ext[:3, 3]
        quat = rotation_matrix_to_quaternion(R)
        
        images[i+1] = Image(
            id=i+1,
            qvec=quat,
            tvec=t,
            camera_id=1,
            name=name,
            xys=np.zeros((0, 2)),  # No 2D points
            point3D_ids=np.full(0, -1)
        )
    
    # Create empty point cloud (will be initialized randomly)
    points3D = {}
    
    # Write binary files
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary(points3D, sparse_dir / "points3D.bin")


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qw, qx, qy, qz]."""
    # Implementation: use scipy.spatial.transform.Rotation
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()  # Returns [qx,qy,qz,qw]
    # Note: COLMAP uses [qw,qx,qy,qz], so reorder!
```

#### Step 2: Use Existing 2DGS Trainer

The good news: `simple_trainer_2dgs.py` already exists! You can use it directly:

```bash
cd examples

# Train 2DGS on your custom dataset
python simple_trainer_2dgs.py \
    --data_dir /path/to/my_dataset \
    --data_factor 1 \
    --result_dir results/my_2dgs \
    --init_type sfm \
    --max_steps 30000
```

**Important parameters:**
- `--data_dir`: Your dataset directory (must have images/ and sparse/)
- `--data_factor`: Downsample factor (1=full res, 2=half, 4=quarter)
- `--init_type`: "sfm" (from COLMAP points) or "random"
- `--max_steps`: Number of training iterations

#### Step 3: Monitor Training

```bash
# View tensorboard logs
tensorboard --logdir results/my_2dgs/tb

# View in real-time with interactive viewer
# (The viewer automatically starts on port 8080)
# Open browser to http://localhost:8080
```

---

## Quick Start Examples

### Example 1: Load and visualize your Gaussians

```python
"""Test loading custom Gaussians before training."""
import torch
from custom_init import load_gaussians_from_file

# Load
gaussians = load_gaussians_from_file('my_gaussians.npz')

print(f"Loaded {gaussians['means'].shape[0]} Gaussians")
print(f"Position range: {gaussians['means'].min()} to {gaussians['means'].max()}")
print(f"Color range: {gaussians['colors_rgb'].min()} to {gaussians['colors_rgb'].max()}")

# Quick visualization with matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
positions = gaussians['means'].cpu().numpy()
colors = gaussians['colors_rgb'].cpu().numpy()
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
           c=colors, s=1)
plt.show()
```

### Example 2: Convert existing point cloud to Gaussian format

```python
"""Convert .ply point cloud to Gaussian .npz format."""
import numpy as np
from plyfile import PlyData

# Load PLY
plydata = PlyData.read('input_pointcloud.ply')
vertices = plydata['vertex']

positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

N = len(positions)

# Create Gaussian parameters with sensible defaults
gaussians = {
    'means': positions,
    'scales': np.ones((N, 3)) * 0.01,  # Small uniform scales
    'quats': np.tile([1, 0, 0, 0], (N, 1)),  # No rotation
    'opacities': np.ones(N) * 0.8,  # Mostly opaque
    'colors_rgb': colors
}

# Save
np.savez('converted_gaussians.npz', **gaussians)
print(f"Converted {N} points to Gaussian format")
```

---

## Key Files Summary

### Must Understand (for Goal 1: Custom 3D Gaussian Init)
1. **`examples/simple_trainer.py`**
   - Lines 225-280: `create_splats_with_optimizers()` - **THIS IS THE KEY FUNCTION**
   - Lines 389-411: How splats are created in `Runner.__init__`
   - Shows the data structure and parameter setup

2. **`examples/utils.py`**
   - `rgb_to_sh()` - Convert RGB to spherical harmonics
   - `knn()` - K-nearest neighbors (for scale estimation)

3. **`gsplat/rendering.py`**
   - `rasterization()` function - Main rendering entry point
   - Understand input format (what the splats dict should contain)

### Must Understand (for Goal 2: 2DGS Custom Dataset)
1. **`examples/simple_trainer_2dgs.py`**
   - Complete 2DGS training pipeline
   - Lines 193-250: Initialization function

2. **`examples/datasets/colmap.py`**
   - `Parser` class - Loads COLMAP data
   - `Dataset` class - Wraps images and cameras
   - **CRITICAL**: Shows expected data format

3. **`gsplat/rendering.py`**
   - `rasterization_2dgs()` function - 2DGS rendering

### Nice to Understand (Supporting)
- `gsplat/strategy/default.py` - Densification strategy
- `gsplat/optimizers/selective_adam.py` - Custom optimizer
- `examples/datasets/traj.py` - Camera trajectory generation

---

## Common Issues and Solutions

### Issue 1: "ValueError: Please specify a correct init_type"
**Solution**: Make sure to add "custom" as a valid option or use the modified trainer.

### Issue 2: Gaussians are invisible/wrong colors
**Solution**: 
- Check scales aren't too small (< 1e-6) or too large (> 1.0)
- Ensure colors are in [0, 1] range
- Verify opacities are not near 0

### Issue 3: NaN/Inf during training
**Solution**:
- Clamp opacities before logit: `torch.clamp(opacities, 0.001, 0.999)`
- Ensure scales are positive before log
- Check for degenerate quaternions (all zeros)

### Issue 4: "Expected COLMAP sparse reconstruction"
**Solution**: 
- Run COLMAP on your images first
- Or create synthetic camera data (see Option B in Step 1)

### Issue 5: Out of memory
**Solution**:
- Reduce number of Gaussians
- Use `--data_factor 4` to downsample images
- Reduce batch size (usually it's already 1)

---

## Testing Your Implementation

```bash
# 1. Test loading
python -c "from custom_init import load_gaussians_from_file; \
           g = load_gaussians_from_file('my_gaussians.npz'); \
           print(f'Loaded {len(g[\"means\"])} Gaussians')"

# 2. Test initialization
python -c "from custom_init import create_splats_from_custom_gaussians; \
           s, o = create_splats_from_custom_gaussians('my_gaussians.npz'); \
           print(f'Created splats with keys: {list(s.keys())}')"

# 3. Test training (1 iteration)
python train_with_custom_init.py \
    --init_type custom \
    --gaussian_file my_gaussians.npz \
    --data_dir data/360_v2/garden \
    --result_dir results/test \
    --max_steps 1 \
    --disable_viewer
```

---

## Next Steps

1. **Start simple**: Use the example NPZ creation code to make a small test file (100-1000 Gaussians)
2. **Visualize first**: Load and plot your Gaussians before training
3. **Train for a few steps**: Set `--max_steps 100` to verify everything works
4. **Compare**: Try both `--init_type sfm` and `--init_type custom` to compare results
5. **Iterate**: Adjust your initial Gaussian parameters based on results

Good luck! The implementation is straightforward once you understand the data structure.
