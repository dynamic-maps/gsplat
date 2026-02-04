# Complete Learning Path for gsplat Mastery

## Overview
To fully understand and contribute to gsplat, you need knowledge across multiple domains. The codebase has ~9,000 lines of Python and ~9,000 lines of CUDA C++, implementing cutting-edge computer graphics and machine learning techniques.

---

## Required Knowledge Areas

### 1. **Foundational Mathematics** 📐

#### Linear Algebra (Critical)
- **3D transformations**: rotation matrices, translation, scaling
- **Coordinate systems**: world space, camera space, screen space
- **Matrix operations**: matrix multiplication, inverse, transpose
- **Eigenvalues/eigenvectors**: used in covariance decomposition
- **Quaternions**: compact rotation representation
- **Homogeneous coordinates**: 4D representation for 3D transforms

#### Probability & Statistics
- **Gaussian distributions**: multivariate normals, mean, covariance
- **Probability density functions**: how Gaussians represent 3D "splats"
- **Covariance matrices**: shape and orientation of Gaussians
- **Mahalanobis distance**: distance metric in Gaussian space

#### Calculus
- **Gradients**: backpropagation through rendering
- **Chain rule**: differentiating through complex operations
- **Partial derivatives**: sensitivity of output to each parameter
- **Jacobian matrices**: used in projection operations

**Key concept**: Every Gaussian is defined by mean μ (position) and covariance Σ (shape/orientation). Projection from 3D to 2D transforms these parameters.

---

### 2. **Computer Graphics** 🎨

#### Rendering Fundamentals
- **Rasterization**: converting 3D geometry to 2D pixels
- **Z-buffering**: depth ordering (though Gaussians use alpha-blending instead)
- **Alpha blending**: compositing semi-transparent layers
  - Formula: `C = Σ(αᵢ * cᵢ * Πⱼ₍ⱼ₎(1-αⱼ))`
- **Perspective projection**: camera model and projection matrix
- **Viewport transformation**: mapping to screen coordinates

#### Camera Models
- **Pinhole camera**: basic projection model
- **Intrinsic parameters**: focal length, principal point, aspect ratio
- **Extrinsic parameters**: camera position and orientation
- **Projection matrix**: combining intrinsic and extrinsic

#### Tile-based Rendering
- **Screen space tiling**: dividing image into blocks (e.g., 16×16)
- **Per-tile culling**: only processing Gaussians affecting each tile
- **Reduces overdraw**: efficiency optimization

#### Color Representations
- **RGB color space**: standard representation
- **Spherical harmonics (SH)**: view-dependent color
  - Compact representation of directional functions
  - Allows Gaussians to have different colors from different angles
  - Degree 0: constant, Degree 1: linear, Degree 2: quadratic, etc.

---

### 3. **Machine Learning & Optimization** 🤖

#### Deep Learning Basics
- **Automatic differentiation**: PyTorch autograd
- **Backpropagation**: computing gradients
- **Custom autograd functions**: implementing new differentiable ops
- **Loss functions**: L1, L2, SSIM, LPIPS for image comparison

#### Optimization
- **Adam optimizer**: adaptive learning rates
- **Learning rate scheduling**: exponential decay, warmup
- **Gradient clipping**: preventing instability
- **Momentum**: smoothing parameter updates

#### Neural Radiance Fields (NeRF) Background
- **Volume rendering**: accumulating color along rays
- **Positional encoding**: high-frequency detail
- **View synthesis**: generating novel views
- **Understanding why Gaussians are better**: faster, explicit, editable

#### Training Strategies (Specific to Gaussian Splatting)
- **Adaptive densification**: 
  - Splitting large Gaussians in high-gradient regions
  - Cloning small Gaussians where detail is needed
- **Pruning**: removing low-opacity or very large Gaussians
- **MCMC strategies**: probabilistic sampling of Gaussian positions

---

### 4. **CUDA Programming** 🚀

#### CUDA Fundamentals
- **Thread hierarchy**: grids, blocks, threads
  - Grid: entire kernel launch
  - Block: group of threads (e.g., 256 threads)
  - Thread: individual execution unit
- **Memory hierarchy**: 
  - Global memory (slow, large, accessible by all)
  - Shared memory (fast, limited, per-block)
  - Registers (fastest, per-thread)
  - Constant memory (cached, read-only)
- **Synchronization**: `__syncthreads()`, atomics
- **Warp execution**: 32 threads execute in lockstep
- **Occupancy**: balancing threads vs. memory usage

#### CUDA Optimization Techniques
- **Memory coalescing**: aligned, contiguous memory access
- **Bank conflicts**: avoiding shared memory conflicts
- **Warp divergence**: minimizing conditional branches
- **Kernel fusion**: combining operations to reduce memory traffic
- **Cooperative groups**: flexible thread synchronization

#### Key CUDA Patterns in gsplat
- **Parallel reduction**: summing/max across threads
- **Prefix sum (scan)**: cumulative operations
- **Sorting**: radix sort for Gaussian ordering
- **Atomic operations**: concurrent updates (e.g., histograms)

#### Custom Autograd with CUDA
- **Forward kernel**: compute output from input
- **Backward kernel**: compute input gradients from output gradients
- **PyTorch C++ extensions**: binding CUDA to Python
- **Memory management**: allocating tensors, handling device pointers

---

### 5. **3D Gaussian Splatting Theory** 💡

#### Core Papers (MUST READ)
1. **"3D Gaussian Splatting for Real-Time Radiance Fields"** (Kerbl et al., 2023)
   - The foundational paper
   - URL: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
   
2. **gsplat white paper** (Ye et al., 2024)
   - Mathematical conventions and derivations
   - URL: https://arxiv.org/abs/2409.06765
   - Benchmarking and implementation details

3. **2D Gaussian Splatting** (2DGS variant)
   - Surface-aligned Gaussians
   - Better for geometric detail

#### Key Concepts

**Gaussian Representation**
```
Each Gaussian is parameterized by:
- μ (mean): 3D position [x, y, z]
- Σ (covariance): 3×3 matrix defining shape
  - Often parameterized as: Σ = R S S^T R^T
  - R: rotation (quaternion → matrix)
  - S: scale (3D vector [sx, sy, sz])
- c (color): RGB or SH coefficients
- α (opacity): transparency value [0, 1]
```

**Projection to 2D**
```
3D Gaussian → Camera space → 2D Gaussian on image plane

Key operation: covariance transformation
Σ2D = J * W * Σ3D * W^T * J^T

Where:
- W: world-to-camera transform
- J: Jacobian of perspective projection
- Σ3D: 3D covariance matrix
- Σ2D: resulting 2D covariance on image
```

**Splatting (Rendering)**
```
For each pixel p:
  1. Find all Gaussians overlapping this pixel
  2. Sort by depth (front to back)
  3. Accumulate color with alpha blending:
     C(p) = Σᵢ cᵢ * αᵢ * G(p|μᵢ, Σᵢ) * Tᵢ
     
     Where:
     - G(p|μᵢ, Σᵢ): 2D Gaussian value at pixel p
     - Tᵢ = Πⱼ₍ⱼ₎(1 - αⱼ * G(p|μⱼ, Σⱼ)): transmittance
```

**Tile-based Rendering**
```
1. Divide screen into tiles (e.g., 16×16 pixels)
2. For each Gaussian:
   - Compute 2D bounding box
   - Find which tiles it overlaps
3. Sort Gaussians per-tile by depth
4. Rasterize each tile independently in parallel
```

---

### 6. **Software Engineering** 💻

#### Python
- **PyTorch**: tensors, autograd, custom Functions
- **Type hints**: understanding jaxtyping annotations
- **Decorators**: @torch.no_grad(), @dataclass, etc.
- **Context managers**: for profiling, memory tracking

#### C++/CUDA
- **Templates**: generic programming for types
- **Memory management**: raw pointers, RAII
- **STL**: vectors, maps, algorithms
- **Namespaces**: code organization

#### Build Systems
- **setuptools**: Python package building
- **CMake**: C++/CUDA compilation (used implicitly)
- **nvcc**: NVIDIA CUDA compiler
- **JIT compilation**: PyTorch's torch.utils.cpp_extension

#### Version Control
- **Git**: branching, merging, rebasing
- **GitHub workflows**: CI/CD, automated testing

---

### 7. **Domain-Specific Techniques** 🎯

#### COLMAP (Structure from Motion)
- **Camera calibration**: extracting intrinsics/extrinsics
- **Point cloud initialization**: starting Gaussians from SfM points
- **Sparse reconstruction**: camera poses and 3D points

#### Image Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (simple)
- **SSIM**: Structural Similarity Index (perceptual)
- **LPIPS**: Learned Perceptual Image Patch Similarity (neural network based)

#### Compression Techniques
- **Quantization**: reducing parameter precision
- **Entropy coding**: lossless compression
- **Pruning**: removing redundant Gaussians
- **Sorting**: arranging for better compression

#### Distributed Training
- **Data parallelism**: replicating model across GPUs
- **Model parallelism**: splitting model across GPUs
- **Gradient synchronization**: AllReduce operations
- **PyTorch DDP**: DistributedDataParallel

---

## Learning Roadmap

### Phase 1: Foundations (2-4 weeks)
**Goal**: Build necessary background knowledge

1. **Mathematics Review**
   - Linear algebra: 3Blue1Brown's "Essence of Linear Algebra" series
   - Multivariate Gaussians: Understanding covariance matrices
   - 3D transformations: Rotation matrices and quaternions

2. **Computer Graphics Basics**
   - Read "Real-Time Rendering" chapters on rasterization
   - Understand perspective projection
   - Learn about alpha blending and compositing

3. **PyTorch Fundamentals**
   - Official PyTorch tutorials
   - Custom autograd functions
   - CUDA extensions basics

### Phase 2: Theory Deep Dive (2-3 weeks)
**Goal**: Understand Gaussian Splatting algorithm

1. **Read the Papers**
   - Original 3DGS paper (carefully, multiple times)
   - gsplat white paper for mathematical details
   - Take notes on every equation

2. **Mathematical Derivations**
   - Work through covariance projection math
   - Understand gradient computation
   - Derive backward passes on paper

3. **Algorithm Understanding**
   - Trace through rendering pipeline step-by-step
   - Understand densification/pruning logic
   - Learn tile-based rasterization strategy

### Phase 3: CUDA Programming (3-4 weeks)
**Goal**: Master parallel programming

1. **CUDA Basics**
   - NVIDIA's "CUDA C Programming Guide"
   - Write simple kernels (vector add, reduction, etc.)
   - Understand memory hierarchy

2. **Advanced CUDA**
   - Shared memory usage
   - Warp-level primitives
   - Cooperative groups
   - Profiling with nsys/ncu

3. **PyTorch + CUDA**
   - Write custom CUDA extensions for PyTorch
   - Implement forward/backward passes
   - Debug with cuda-gdb

### Phase 4: Code Exploration (2-3 weeks)
**Goal**: Navigate and understand the codebase

1. **Start with Python API** (`gsplat/rendering.py`)
   - Trace a simple rendering call
   - Understand data flow
   - Map high-level functions to CUDA kernels

2. **CUDA Kernels** (`gsplat/cuda/csrc/`)
   - Read projection kernels
   - Understand rasterization kernels
   - Study backward pass implementations

3. **Training Logic** (`gsplat/strategy/`)
   - Understand densification strategy
   - Learn pruning criteria
   - Study MCMC approach

4. **Run Examples**
   - Start with `simple_trainer.py`
   - Add print statements to trace execution
   - Visualize with `simple_viewer.py`

### Phase 5: Experimentation (Ongoing)
**Goal**: Make modifications and improvements

1. **Small Changes**
   - Modify learning rates
   - Change densification thresholds
   - Try different pruning strategies

2. **Debugging Practice**
   - Intentionally break things
   - Use CUDA debugging tools
   - Profile performance bottlenecks

3. **New Features**
   - Implement papers building on 3DGS
   - Add new rendering modes
   - Optimize specific kernels

---

## Key Files to Study (In Order)

### Python Side
1. `gsplat/__init__.py` - See what's exported
2. `gsplat/rendering.py` - Main rendering API
3. `gsplat/cuda/_wrapper.py` - Python-CUDA bridge
4. `gsplat/strategy/default.py` - Training strategy
5. `examples/simple_trainer.py` - End-to-end training

### CUDA Side
1. `gsplat/cuda/csrc/Projection.h` - Projection declarations
2. `gsplat/cuda/csrc/ProjectionEWA3DGSFused.cu` - 3D→2D projection
3. `gsplat/cuda/csrc/RasterizeToPixels3DGSFwd.cu` - Forward rendering
4. `gsplat/cuda/csrc/RasterizeToPixels3DGSBwd.cu` - Backward pass
5. `gsplat/cuda/csrc/IntersectTile.cu` - Tile intersection

---

## Essential Resources

### Papers
- Original 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- gsplat white paper: https://arxiv.org/abs/2409.06765
- Related work: Search "Gaussian Splatting" on arXiv

### Documentation
- gsplat docs: https://docs.gsplat.studio/
- PyTorch CUDA extensions: https://pytorch.org/tutorials/advanced/cpp_extension.html
- NVIDIA CUDA docs: https://docs.nvidia.com/cuda/

### Courses
- Stanford CS231n (Computer Vision)
- NVIDIA CUDA Teaching Kit
- Scratchapixel (Computer Graphics fundamentals)

### Tools
- **CUDA**: nsys (system profiling), ncu (kernel profiling), cuda-gdb (debugging)
- **Python**: pdb (debugger), cProfile (profiling), torch.profiler
- **Visualization**: Rerun, Polyscope, Open3D

---

## Practical Tips

### Debugging Strategy
1. **Start small**: Use tiny scenes (10-100 Gaussians)
2. **Visualize everything**: Print tensor shapes, values, gradients
3. **Compare with reference**: Use `_torch_impl.py` as ground truth
4. **Profile first**: Don't optimize without measuring
5. **Unit tests**: Write tests for new features

### Common Pitfalls
- **Memory alignment**: CUDA performance depends on coalesced access
- **Gradient correctness**: Always run torch.autograd.gradcheck
- **Numerical stability**: Watch for NaN/Inf in backward passes
- **Race conditions**: Atomic operations for concurrent writes
- **Synchronization**: CUDA kernels are async, need explicit sync for timing

### Development Workflow
```bash
# 1. Make changes to CUDA code
vim gsplat/cuda/csrc/MyKernel.cu

# 2. Reinstall package (triggers recompilation)
pip install -e .

# 3. Run tests
python tests/test_rasterization.py

# 4. Profile if needed
nsys profile python examples/simple_trainer.py

# 5. Iterate
```

---

## Reality Check

**Estimated time to mastery**: 3-6 months of focused effort

**You don't need to know everything perfectly before starting**. Many concepts will click into place as you work with the code. Start with the learning path above, but don't be afraid to jump into the code early.

**The best way to learn is by doing**: Run the examples, make small modifications, break things and fix them. Reading alone won't get you there.

**Focus areas based on your goals**:
- Want to add features? → Focus on Python API and algorithm theory
- Want to optimize? → Deep dive into CUDA and profiling
- Want to research? → Study papers and mathematical foundations
- Want to integrate into apps? → Understand API and examples

You've got this! The codebase is well-structured and the community is active. Don't hesitate to ask questions in GitHub issues or discussions.
