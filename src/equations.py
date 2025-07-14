import numpy as np

# --- Euler Equations ---

def hllc_flux_euler(U_L, U_R, normal, gamma):
    """Calculates the HLLC flux for the 2D Euler Equations."""
    rho_L, rho_u_L, rho_v_L, E_L = U_L
    rho_R, rho_u_R, rho_v_R, E_R = U_R

    u_L = rho_u_L / rho_L
    v_L = rho_v_L / rho_L
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))

    u_R = rho_u_R / rho_R
    v_R = rho_v_R / rho_R
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))

    un_L = u_L * normal[0] + v_L * normal[1]
    un_R = u_R * normal[0] + v_R * normal[1]

    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)

    S_L = min(un_L - c_L, un_R - c_R)
    S_R = max(un_L + c_L, un_R + c_R)

    F_L = np.array([
        rho_L * un_L,
        rho_u_L * un_L + p_L * normal[0],
        rho_v_L * un_L + p_L * normal[1],
        (E_L + p_L) * un_L
    ])

    F_R = np.array([
        rho_R * un_R,
        rho_u_R * un_R + p_R * normal[0],
        rho_v_R * un_R + p_R * normal[1],
        (E_R + p_R) * un_R
    ])

    if S_L >= 0:
        return F_L
    if S_R <= 0:
        return F_R

    S_star = (p_R - p_L + rho_L * un_L * (S_L - un_L) - rho_R * un_R * (S_R - un_R)) / (rho_L * (S_L - un_L) - rho_R * (S_R - un_R))

    U_star_L = rho_L * (S_L - un_L) / (S_L - S_star) * np.array([1, S_star, v_L, E_L/rho_L + (S_star - un_L) * (S_star + p_L/(rho_L * (S_L - un_L)))])
    U_star_R = rho_R * (S_R - un_R) / (S_R - S_star) * np.array([1, S_star, v_R, E_R/rho_R + (S_star - un_R) * (S_star + p_R/(rho_R * (S_R - un_R)))])

    if S_star >= 0:
        return F_L + S_L * (U_star_L - U_L)
    else:
        return F_R + S_R * (U_star_R - U_R)

def calculate_adaptive_dt_euler(mesh, U, gamma, cfl_number):
    """
    Calculates the adaptive time step based on the CFL condition for Euler equations.
    """
    max_wave_speed = 0.0
    for i in range(mesh.nelem):
        rho, rho_u, rho_v, E = U[i]
        u = rho_u / rho
        v = rho_v / rho
        p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        c = np.sqrt(gamma * p / rho)
        wave_speed = np.sqrt(u**2 + v**2) + c
        if wave_speed > max_wave_speed:
            max_wave_speed = wave_speed

    char_length = np.sqrt(np.min(mesh.cell_volumes))

    if max_wave_speed > 1e-9:
        return cfl_number * char_length / max_wave_speed
    else:
        return 1e-3

def apply_boundary_condition_euler(U_inside, normal, bc_info, gamma):
    """Applies a boundary condition for Euler equations and returns the ghost cell value."""
    bc_type = bc_info.get("type", "wall")
    rho, rho_u, rho_v, E = U_inside
    u = rho_u / rho
    v = rho_v / rho

    if bc_type == "wall":
        un = u * normal[0] + v * normal[1]
        ut = u * -normal[1] + v * normal[0]
        un_ghost = -un
        ut_ghost = ut
        u_ghost = un_ghost * normal[0] - ut_ghost * normal[1]
        v_ghost = un_ghost * normal[1] + ut_ghost * normal[0]
        return np.array([rho, rho * u_ghost, rho * v_ghost, E])
    elif bc_type == "inlet":
        return bc_info.get("value", U_inside)
    elif bc_type == "outlet":
        return bc_info.get("value", U_inside)
    else:  # Default to outflow
        return U_inside


def hllc_flux_shallow_water(U_L, U_R, normal, g):
    """Calculates the HLLC flux for the 2D Shallow Water Equations."""
    h_L, hu_L, hv_L = U_L
    h_R, hu_R, hv_R = U_R

    u_L = hu_L / h_L if h_L > 1e-6 else 0
    v_L = hv_L / h_L if h_L > 1e-6 else 0
    u_R = hu_R / h_R if h_R > 1e-6 else 0
    v_R = hv_R / h_R if h_R > 1e-6 else 0

    un_L = u_L * normal[0] + v_L * normal[1]
    un_R = u_R * normal[0] + v_R * normal[1]

    c_L = np.sqrt(g * h_L) if h_L > 0 else 0
    c_R = np.sqrt(g * h_R) if h_R > 0 else 0

    S_L = min(un_L - c_L, un_R - c_R)
    S_R = max(un_L + c_L, un_R + c_R)

    F_L = np.array(
        [
            h_L * un_L,
            hu_L * un_L + 0.5 * g * h_L**2 * normal[0],
            hv_L * un_L + 0.5 * g * h_L**2 * normal[1],
        ]
    )
    F_R = np.array(
        [
            h_R * un_R,
            hu_R * un_R + 0.5 * g * h_R**2 * normal[0],
            hv_R * un_R + 0.5 * g * h_R**2 * normal[1],
        ]
    )

    if S_L >= 0:
        return F_L
    if S_R <= 0:
        return F_R

    # Avoid division by zero
    if (h_R * (un_R - S_R) - h_L * (un_L - S_L)) == 0:
        return 0.5 * (F_L + F_R)

    S_star = (
        S_R * h_R * (un_R - S_R)
        - S_L * h_L * (un_L - S_L)
        + 0.5 * g * (h_L**2 - h_R**2)
    ) / (h_R * (un_R - S_R) - h_L * (un_L - S_L))

    # Avoid division by zero
    if (S_L - S_star) == 0 or (S_R - S_star) == 0:
        return 0.5 * (F_L + F_R)

    U_star_L = (S_L * U_L - F_L) / (S_L - S_star)
    U_star_R = (S_R * U_R - F_R) / (S_R - S_star)

    if S_star >= 0:
        return F_L + S_L * (U_star_L - U_L)
    else:
        return F_R + S_R * (U_star_R - U_R)


def calculate_adaptive_dt_shallow_water(mesh, U, g, cfl_number):
    """
    Calculates the adaptive time step based on the CFL condition for Shallow Water.
    """
    max_wave_speed = 0.0
    for i in range(mesh.nelem):
        h = U[i, 0]
        hu = U[i, 1]
        hv = U[i, 2]
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        c = np.sqrt(g * h) if h > 0 else 0
        wave_speed = np.sqrt(u**2 + v**2) + c
        if wave_speed > max_wave_speed:
            max_wave_speed = wave_speed

    # Characteristic length (e.g., sqrt of cell area for 2D)
    char_length = np.sqrt(np.min(mesh.cell_volumes))

    if max_wave_speed > 1e-9:
        return cfl_number * char_length / max_wave_speed
    else:
        return 1e-3  # Default small dt if wave speed is zero

def apply_boundary_condition_shallow_water(U_inside, normal, bc_info):
    """Applies a boundary condition for shallow water and returns the ghost cell value."""
    bc_type = bc_info.get("type", "wall")
    if bc_type == "wall":
        h, hu, hv = U_inside
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        un = u * normal[0] + v * normal[1]
        ut = u * -normal[1] + v * normal[0]
        un_ghost = -un
        ut_ghost = ut
        u_ghost = un_ghost * normal[0] - ut_ghost * normal[1]
        v_ghost = un_ghost * normal[1] + ut_ghost * normal[0]
        return np.array([h, h * u_ghost, h * v_ghost])
    elif bc_type == "inlet":
        return bc_info.get("value", U_inside)
    elif bc_type == "outlet":
        return bc_info.get("value", U_inside)
    else:  # Default to outflow
        return U_inside
