import numpy as np
from src.mesh import Mesh


def setup_case(mesh: Mesh):
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
