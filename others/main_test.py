import numpy as np
from .unstructured_mesh import generate_random_mesh
from .shallow_water_2d import ShallowWaterSolver
from .OneDomain import OneDWaterDomain


# Example usage
def main():
    mesh = generate_random_mesh(100)
    bed = np.random.uniform(0, 0.5, mesh.n_cells)

    # Define boundary conditions
    boundary_conditions = {}

    for face_idx in mesh.boundary_faces:
        if mesh.face_centers[face_idx, 0] < 0.1:  # Left boundary
            boundary_conditions[face_idx] = "1d_coupling"
        elif mesh.face_centers[face_idx, 0] > 0.9:  # Right boundary
            boundary_conditions[face_idx] = "outflow"
        else:
            boundary_conditions[face_idx] = "reflective"
    # Initialize 1D domain
    x_1d = np.linspace(0, 0.1, 50)
    h_1d = np.ones_like(x_1d)
    u_1d = np.zeros_like(x_1d)
    bed_1d = np.linspace(0, 0.5, len(x_1d))
    one_d_domain = OneDWaterDomain(x_1d, h_1d, u_1d, bed_1d)
    solver = ShallowWaterSolver(mesh, bed, boundary_conditions, one_d_domain)
    solver.initialize(h0=1.0, u0=0.0, v0=0.0)
    solver.dt = solver.compute_dt()
    for _ in range(100):
        solver.step()
    solver.plot_solution()


if __name__ == "__main__":
    main()
