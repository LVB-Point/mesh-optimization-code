# Mesh Optimization with Curvature-Aware Point Insertion

This repository contains the implementation of our proposed method for mesh refinement and point cloud optimization.

## Overview

This method performs adaptive point insertion on triangular meshes based on:

- Triangle curvature  
- Triangle area  

An energy-based optimization is then applied to improve point distribution, including:

- Area regularization  
- Distance uniformity  
- Triangle shape optimization (towards equilateral triangles)  

## Requirements

Install the required dependencies:

```bash
pip install numpy scipy open3d
```
## Usage

Prepare your input mesh file in .ply format
Put it in the same directory as the script

Run
```bash
python mesh_optimization.py input.ply
```
If no argument is provided, the script will try to read:input.ply

## Output

The optimized point cloud will be saved as:
output.ply

## Notes
- The triangulation is performed using 2D projection (XY plane)
- The optimization is based on numerical gradient descent
- This code is intended for research and academic use

## License

This project is released for academic and research purposes only.

## Citation

If you use this code, please cite our paper (to be updated after publication).
