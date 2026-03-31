# Core Implementation of Mesh Optimization with Curvature-Aware Point Insertion

This repository provides the **core functions** of our proposed method for mesh refinement and point cloud optimization.

## Overview

The proposed method focuses on adaptive refinement of triangular meshes through:

* Curvature-guided point insertion
* Area-based refinement strategy
* Energy-based optimization for point distribution

The optimization improves:

* Triangle area regularity
* Point distribution uniformity
* Triangle shape quality (towards equilateral triangles)

---

## Code Structure

This repository contains only the **core components** corresponding to the key steps described in the paper:

```text
core/
 ├── curvature.py        # curvature estimation
 ├── insertion.py        # adaptive point insertion
 ├── optimization.py     # energy-based optimization
```

Each module is self-contained and reflects the main algorithmic contributions.

---

## Usage

The provided code is intended for **algorithmic reference and understanding**.

Users may integrate these core functions into their own pipelines for:

* Mesh processing
* Point cloud refinement
* Geometry optimization

---

## Important Notes

* This repository **does not include a complete runnable pipeline**
* Data preprocessing, mesh construction, and I/O operations are not provided
* Some implementation details (e.g., parameter tuning and auxiliary steps) are described in the paper

---

## Reproducibility

The full implementation, including:

* End-to-end pipeline
* Parameter configurations
* Experimental data

will be made publicly available upon acceptance of the paper.

---

## Requirements

The core functions are implemented in Python and may require:

```bash
pip install numpy scipy open3d
```

---

## Citation

If you find this work useful, please cite our paper (to be updated after publication).

---

## License

This project is released for academic research purposes only.
