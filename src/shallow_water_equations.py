import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class ShallowWaterEquations(BaseEquation):
    def __init__(self, g):
        self.g = g

    def hllc_flux(self, U_L, U_R, normal):
        g = self.g
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

        if (h_R * (un_R - S_R) - h_L * (un_L - S_L)) == 0:
            return 0.5 * (F_L + F_R)

        S_star = (
            S_R * h_R * (un_R - S_R)
            - S_L * h_L * (un_L - S_L)
            + 0.5 * g * (h_L**2 - h_R**2)
        ) / (h_R * (un_R - S_R) - h_L * (un_L - S_L))

        if (S_L - S_star) == 0 or (S_R - S_star) == 0:
            return 0.5 * (F_L + F_R)

        U_star_L = (S_L * U_L - F_L) / (S_L - S_star)
        U_star_R = (S_R * U_R - F_R) / (S_R - S_star)

        if S_star >= 0:
            return F_L + S_L * (U_star_L - U_L)
        else:
            return F_R + S_R * (U_star_R - U_R)

    def _calculate_wave_speed(self, U_cell):
        h = U_cell[0]
        hu = U_cell[1]
        hv = U_cell[2]
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        c = np.sqrt(self.g * h) if h > 0 else 0
        return np.sqrt(u**2 + v**2) + c

    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        return calculate_adaptive_dt(mesh, U, self._calculate_wave_speed, cfl_number)

    def apply_boundary_condition(self, U_inside, normal, bc_info):
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
        else:
            return U_inside
