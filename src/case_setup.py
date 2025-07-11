import numpy as np


def dam_break_initial_conditions(elem_conn, node_coord):
    """
    Sets up the initial conditions for a dam break scenario.

    Args:
        elem_conn (np.ndarray): Element connectivity array.
        node_coord (np.ndarray): Node coordinates array.

    Returns:
        np.ndarray: An array of conserved variables (h, hu, hv).
    """
    nelem = len(elem_conn)
    U = np.zeros((nelem, 3))
    for i, elem in enumerate(elem_conn):
        centroid = np.mean(node_coord[elem - 1], axis=0)
        U[i] = [1.0, 1, 0.2] if centroid[0] < 50 else [0.5, 0.5, 0.5]
    return U


def initial_conditions(elem_conn, node_coord):
    """
    Sets initial condition
    """
    nelem = len(elem_conn)
    U = np.zeros((nelem, 3))
    U[:] = [1.0, 0.1, 0.1]  # Example uniform initial condition
    return U
