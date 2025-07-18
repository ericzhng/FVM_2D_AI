import numpy as np

def calculate_adaptive_dt(mesh, U, calculate_wave_speed, cfl_number):
    """
    Calculates the adaptive time step based on the CFL condition.
    """
    max_wave_speed = 0.0
    for i in range(mesh.nelem):
        wave_speed = calculate_wave_speed(U[i])
        if wave_speed > max_wave_speed:
            max_wave_speed = wave_speed

    char_length = np.sqrt(np.min(mesh.cell_volumes))

    if max_wave_speed > 1e-9:
        return cfl_number * char_length / max_wave_speed
    else:
        return 1e-3
