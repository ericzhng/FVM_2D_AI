import numpy as np
from src.mesh import Mesh


def setup_case_shallow_water(mesh: Mesh):
    """
    Sets up the initial conditions for the simulation.
    """
    U = np.zeros((mesh.nelem, 3))  # [h, hu, hv]
    for i in range(mesh.nelem):
        U[i, 0] = 5.0  # Water height (h)
        U[i, 1] = 10.0  # Momentum in x (hu)
        U[i, 2] = 10.0  # Momentum in y (hv)

    boundary_conditions = {
        "top": {"type": "wall"},
        "bottom": {"type": "wall"},
        "left": {"type": "inlet", "value": np.array([5.0, 10.0, 0.0])},  # [h, hu, hv]
        "right": {"type": "outlet", "value": np.array([5.0, 10.0, 0.0])},
    }

    return U, boundary_conditions

def setup_case_euler(mesh: Mesh, gamma=1.4):
    """
    Sets up the initial conditions for the 2D Euler simulation.
    """
    U = np.zeros((mesh.nelem, 4))  # [rho, rho*u, rho*v, E]

    # Example: Sod's shock tube rotated 45 degrees
    for i in range(mesh.nelem):
        x, y, _ = mesh.cell_centroids[i]
        if x + y < 0:
            rho = 1.0
            p = 1.0
            u = 0.75
            v = 0.0
        else:
            rho = 0.125
            p = 0.1
            u = 0.0
            v = 0.0
        
        E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)
        U[i] = np.array([rho, rho * u, rho * v, E])

    # Define boundary conditions for Euler equations - all transmissive
    boundary_conditions = {
        "top": {"type": "outlet"},
        "bottom": {"type": "outlet"},
        "left": {"type": "outlet"},
        "right": {"type": "outlet"}
    }

    return U, boundary_conditions

