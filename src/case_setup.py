import numpy as np
from src.mesh import Mesh


def setup_dam_break_scenario(mesh: Mesh):
    """
    Sets up the initial conditions for a dam break scenario.
    """
    U = np.zeros((mesh.nelem, 3))  # [h, hu, hv]
    for i in range(mesh.nelem):
        if mesh.cell_centroids[i][0] < 10.0:
            U[i, 0] = 2.0  # Water height (h)
        else:
            U[i, 0] = 1.0
        U[i, 1] = 0.0  # Momentum in x (hu)
        U[i, 2] = 0.0  # Momentum in y (hv)

    boundary_conditions = {
        "wall": "wall",
        "outflow": "outflow"
    }

    return U, boundary_conditions

