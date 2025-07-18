import numpy as np

def calculate_adaptive_dt(mesh, U, calculate_wave_speed, cfl_number):
    """
    Calculates the adaptive time step for the simulation based on the CFL condition.

    The time step is limited by the maximum wave speed in the domain and the
    characteristic length of the cells to ensure numerical stability.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        calculate_wave_speed (function): A function that calculates the maximum
                                       wave speed for a given state.
        cfl_number (float): The Courant-Friedrichs-Lewy (CFL) number.

    Returns:
        float: The calculated adaptive time step (dt).
    """
    max_wave_speed = 0.0
    for i in range(mesh.nelem):
        wave_speed = calculate_wave_speed(U[i])
        if wave_speed > max_wave_speed:
            max_wave_speed = wave_speed

    # Characteristic length (e.g., smallest cell diameter)
    # A simple approximation is the square root of the minimum cell volume (for 2D)
    char_length = np.sqrt(np.min(mesh.cell_volumes))

    # Avoid division by zero if the fluid is at rest
    if max_wave_speed > 1e-9:
        return cfl_number * char_length / max_wave_speed
    else:
        # Return a small, fixed time step if there is no flow
        return 1e-3
