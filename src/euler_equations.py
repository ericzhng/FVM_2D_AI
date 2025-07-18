import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class EulerEquations(BaseEquation):
    def __init__(self, gamma):
        self.gamma = gamma

    def hllc_flux(self, U_L, U_R, normal):
        gamma = self.gamma
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

        F_L = np.array(
            [
                rho_L * un_L,
                rho_u_L * un_L + p_L * normal[0],
                rho_v_L * un_L + p_L * normal[1],
                (E_L + p_L) * un_L,
            ]
        )

        F_R = np.array(
            [
                rho_R * un_R,
                rho_u_R * un_R + p_R * normal[0],
                rho_v_R * un_R + p_R * normal[1],
                (E_R + p_R) * un_R,
            ]
        )

        if S_L >= 0:
            return F_L
        if S_R <= 0:
            return F_R

        S_star = (
            p_R - p_L + rho_L * un_L * (S_L - un_L) - rho_R * un_R * (S_R - un_R)
        ) / (rho_L * (S_L - un_L) - rho_R * (S_R - un_R))

        p_star = p_L + rho_L * (un_L - S_L) * (un_L - S_star)

        U_star_L = (
            S_L * U_L
            - F_L
            + np.array([0, p_star * normal[0], p_star * normal[1], p_star * S_star])
        ) / (S_L - S_star)
        U_star_R = (
            S_R * U_R
            - F_R
            + np.array([0, p_star * normal[0], p_star * normal[1], p_star * S_star])
        ) / (S_R - S_star)

        if S_star >= 0:
            return F_L + S_L * (U_star_L - U_L)
        else:
            return F_R + S_R * (U_star_R - U_R)

    def _calculate_wave_speed(self, U_cell):
        rho, rho_u, rho_v, E = U_cell
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        c = np.sqrt(self.gamma * p / rho)
        return np.sqrt(u**2 + v**2) + c

    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        return calculate_adaptive_dt(mesh, U, self._calculate_wave_speed, cfl_number)

    def apply_boundary_condition(self, U_inside, normal, bc_info):
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
        else:
            return U_inside
