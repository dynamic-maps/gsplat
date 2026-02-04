# gsplat Repository Structure Guide

## Overview
gsplat is a CUDA-accelerated library for 3D Gaussian Splatting with Python bindings. The repository implements fast, memory-efficient rasterization of 3D Gaussians for neural rendering.

---

## Directory Structure

### Root Level Files
- **setup.py** - Python package installation configuration
- **README.md** - Project documentation and getting started guide
- **LICENSE** - Project license
- **CITATION.bib** - Citation information for academic use
- **formatter.sh** - Code formatting script
- **MANIFEST.in** - Specifies additional files to include in package distribution

### Main Folders

#### **gsplat/** *CORE LIBRARY*
The main Python package containing all core functionality. This is the most important folder.

**Top-level modules:**
- `__init__.py` - Package initialization and exports
- `rendering.py` - Main rendering functions and pipelines
- `distributed.py` - Distributed training support (multi-GPU)
- `utils.py` - General utility functions
- `relocation.py` - Gaussian relocation operations
- `exporter.py` - Export functionality for trained models
- `color_correct.py` - Color correction utilities
- `profile.py` - Performance profiling tools
- `_helper.py` - Internal helper functions
- `version.py` - Version information
- `csrc.so` - Compiled CUDA extension (binary)

**Subdirectories:**

**gsplat/cuda/** - CUDA implementations
- `_backend.py` - Backend interface for CUDA operations
- `_wrapper.py` - Python wrappers for CUDA functions
- `_torch_impl.py` - Pure PyTorch implementations (fallback/reference)
- `_torch_impl_2dgs.py` - PyTorch implementation for 2D Gaussian Splatting
- `ext.cpp` - C++ extension entry point
- `csrc/` - CUDA source code (.cu, .cuh, .h files)
  - Projection operations (3DGS and 2DGS variants)
  - Rasterization kernels (forward and backward passes)
  - Spherical harmonics calculations
  - Tile intersection tests
  - Relocation operations
  - Third-party dependencies (GLM library)
- `include/` - Header files for CUDA code

**gsplat/strategy/** - Training strategies and optimization
- `base.py` - Base strategy class/interface
- `default.py` - Default Gaussian densification/pruning strategy
- `mcmc.py` - MCMC-based optimization strategy
- `ops.py` - Strategy-related operations

**gsplat/optimizers/** - Custom optimizers
- `selective_adam.py` - Selective Adam optimizer (optimizes subset of parameters)

**gsplat/compression/** - Model compression
- `png_compression.py` - PNG-based compression for Gaussians
- `sort.py` - Sorting operations for compression

#### **examples/**
Demonstration scripts and training code showing how to use gsplat.

**Key files:**
- `simple_trainer.py` - Basic training script for 3D Gaussian Splatting
- `simple_trainer_2dgs.py` - Training script for 2D Gaussian Splatting variant
- `simple_viewer.py` - Interactive viewer for trained models
- `simple_viewer_2dgs.py` - Viewer for 2DGS models
- `simple_viewer_3dgut.py` - Viewer for 3DGUT models
- `gsplat_viewer.py` - Advanced viewer with more features
- `gsplat_viewer_2dgs.py` - Advanced 2DGS viewer
- `image_fitting.py` - Example: fitting a 2D image with 3D Gaussians
- `lib_bilagrid.py` - Bilateral grid library
- `exif.py` - EXIF data handling utilities
- `utils.py` - Example utilities

**Subdirectories:**
- `benchmarks/` - Benchmark scripts and configurations
- `datasets/` - Dataset download and preprocessing scripts
- `data/` - Local data storage (likely empty or .gitignored)

#### **tests/**
Unit and integration tests for the library.

**Test files:**
- `test_rasterization.py` - Tests for rasterization operations
- `test_basic.py` - Basic functionality tests
- `test_2dgs.py` - Tests for 2D Gaussian Splatting
- `test_compression.py` - Compression functionality tests
- `test_strategy.py` - Strategy/optimization tests
- `test_ftheta.py` - Specific function tests
- `_test_distributed.py` - Distributed training tests

#### **docs/**
Documentation source files and build configuration.

**Files:**
- `DEV.md` - Development guidelines for contributors
- `INSTALL_WIN.md` - Windows installation instructions
- `3dgut.md` - Documentation for 3DGUT integration
- `batch.md` - Batching documentation
- `Makefile` - Documentation build configuration
- `requirements.txt` - Documentation build dependencies
- `source/` - Sphinx documentation source files

#### **profiling/**
Performance profiling scripts.

**Files:**
- `main.py` - Main profiling script
- `batch.py` - Batch profiling operations

#### **assets/**
Test data and resources.

**Contents:**
- `test_garden.npz` - Sample test dataset

#### **results/**
Output directory for training results, checkpoints, and rendered images.

**Example subdirectories:**
- `26009_Record004/`
- `bonsai/`
- `garden/`

---

## Is gsplat/ the Only Important Part?

**For understanding the core algorithm:** Yes, `gsplat/` is the most critical folder. It contains:
1. The CUDA kernels that do the actual rendering
2. The Python API that users interact with
3. Training strategies and optimization logic
4. All core functionality

**However, the other folders serve important purposes:**
- **examples/** - Essential for learning *how* to use the library and seeing it in action
- **tests/** - Critical for understanding expected behavior and edge cases
- **docs/** - Useful for high-level understanding and API reference

**Recommended exploration order:**
1. `gsplat/rendering.py` - Start here to understand the main API
2. `gsplat/cuda/_wrapper.py` - See how Python calls CUDA
3. `gsplat/cuda/csrc/*.cu` - Dive into CUDA kernels (the actual computation)
4. `gsplat/strategy/` - Understand optimization strategies
5. `examples/simple_trainer.py` - See how it all fits together in practice

---

## Key Concepts

**3D Gaussian Splatting:**
- Represents scenes as collections of 3D Gaussian distributions
- Each Gaussian has position, covariance (shape/orientation), color, and opacity
- Rendering involves projecting 3D Gaussians to 2D and alpha-blending

**2D Gaussian Splatting (2DGS):**
- Variant that uses 2D Gaussians directly on surfaces
- Separate implementation files (*_2dgs.py)

**CUDA Acceleration:**
- Performance-critical operations written in CUDA
- Forward and backward passes for gradient-based optimization
- Tile-based rasterization for efficiency

**Training Strategies:**
- Adaptive densification (splitting/cloning Gaussians in high-gradient areas)
- Pruning (removing low-opacity or large Gaussians)
- MCMC-based sampling strategies
