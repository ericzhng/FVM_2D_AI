# Unstructured Mesh Generator

This project provides a Python implementation for generating and analyzing unstructured meshes in 2D and 3D using Delaunay triangulation. It includes:

- Mesh generation with random points
- Calculation of cell centers, face centers, volumes, areas, normals, and neighbors
- Visualization of 2D and 3D meshes with matplotlib

## Requirements
- numpy
- scipy
- matplotlib

Install dependencies with:

```
pip install numpy scipy matplotlib
```

## Usage
Run the script to generate and visualize a random 2D mesh:

```
python unstructured_mesh.py
```

To enable 3D mesh generation, uncomment the relevant section in the `__main__` block.

## File Overview
- `unstructured_mesh.py`: Main script containing the mesh generation and visualization code.

## License
MIT License
