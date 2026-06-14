# VGGT — Multiview Geometry for Food Volume Estimation

**Chenxu (Max) Lyu**, Christ's College, University of Cambridge  
Supervisor: Prof. Roberto Cipolla  
May 2026

> This repository is forked from [VGGT](https://github.com/facebookresearch/vggt) (Meta/Oxford). See [VGGT_README.md](VGGT_README.md) for the original documentation.  
> This is an **auxiliary repository** to the main project at [sam-3d-objects](../sam-3d-objects/). It serves as the VGGT package used by the SAM3D pipeline, and contains early exploration scripts for VGGT-based volume estimation.

---

## Overview

VGGT (Visual Geometry Grounded Transformer) is a feed-forward model that takes sparse multiview images and produces per-pixel 3D point maps along with camera extrinsics and intrinsics — all in a single forward pass. In this project, VGGT provides the multiview geometric grounding injected into SAM3D's generation pipeline.

This repo is installed as an **editable package** (`pip install -e .`) into the `vggt` conda environment, so any local changes are immediately reflected when `sam3d+vggt_method/vggt_inference.py` imports it:

```bash
conda activate vggt
pip install -e .
```

---

## Exploration and Baselines

### VGGT + Poisson Surface Reconstruction (Baseline)

The initial approach used VGGT point clouds directly as input to Poisson Surface Reconstruction to produce watertight meshes for volume estimation. This was evaluated on the NutritionVerse-3D dataset.

This baseline was found to be **brittle**: Poisson reconstruction frequently failed or produced degenerate meshes due to noise and incomplete coverage in sparse VGGT point clouds, especially for small or thin food items. This motivated moving to generative model-based approaches (SAM3D, TRELLIS) rather than explicit reconstruction.

Relevant scripts: `test_scripts/poisson_mesh_generation.py`, `test_scripts/poisson_mesh_generation_new.py`

### Parameterization Pipeline

After the Poisson baseline, a direct volume estimation pipeline from VGGT point clouds was developed — bypassing surface reconstruction entirely by fitting geometric primitives and using plate diameter as a metric anchor.

Relevant scripts: `parameterization/MeshVolume.py`, `parameterization/dataset_statistics.py`, `parameterization/dataset_statistics_real_dataset.py`

### VGGT → SAM3D Interface

Early prototype connecting VGGT output directly to the SAM3D pipeline. This evolved into the main `sam3d+vggt_method/vggt_inference.py` in the SAM3D repo.

Relevant scripts: `test_scripts/VGGT_interface_SAM3D.py`, `test_scripts/VGGT_COLMAP.py`

---

## Repository Structure

```
vggt/
│
├── test_scripts/               # Early exploration scripts
│   ├── poisson_mesh_generation.py      # Poisson Surface Reconstruction baseline (brittle)
│   ├── poisson_mesh_generation_new.py  # Improved Poisson with scale analysis
│   ├── VGGT_COLMAP.py                  # VGGT + COLMAP sparse reconstruction
│   └── VGGT_interface_SAM3D.py         # Early VGGT → SAM3D prototype
│
├── parameterization/           # Volume estimation from VGGT point clouds
│   ├── MeshVolume.py               # Point cloud → volume via geometric fitting
│   ├── dataset_statistics.py       # Volume evaluation on NutritionVerse-3D
│   ├── dataset_statistics_real_dataset.py  # Volume evaluation on RealFoodScenes
│   └── view_synthesis*.py          # Synthetic view rendering from meshes
│
├── ObjectCapture/              # Apple ObjectCapture baseline scripts
│   └── usdz_volume.py              # Volume extraction from ObjectCapture USDZ outputs
│
├── viewpoints/                 # Height estimation experiments (intrinsic vs extrinsic)
│
└── vggt/                       # Upstream VGGT model source (the importable package)
```

---

## Setup

Follow the upstream VGGT setup instructions in [VGGT_README.md](VGGT_README.md).

To use this repo as the `vggt` package for the SAM3D pipeline:
```bash
conda activate vggt
pip install -e .
```

This repo should be cloned at the same parent path as `sam-3d-objects/` (e.g., `/scratch/<user>/`) so that cross-repo path references resolve correctly.
