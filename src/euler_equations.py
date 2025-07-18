import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class EulerEquations(BaseEquation):
    def __init__(self, gamma=1.4):
        self.gamma = gamma

    def _cons_to_prim(self, U):
        rho, rho_u, rho_v, E = U
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _calculate_wave_speeds(self, q_L, q_R, normal):
        rho_L, u_L, v_L, p_L = q_L
        rho_R, u_R, v_R, p_R = q_R

        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)

        S_L = min(un_L - c_L, un_R - c_R)
        S_R = max(un_L + c_L, un_R + c_R)

        return S_L, S_R

    def _calculate_flux(self, U, q, normal):
        rho, rho_u, rho_v, E = U
        rho_q, u, v, p = q
        un = u * normal[0] + v * normal[1]
        return np.array(
            [
                rho * un,
                rho_u * un + p * normal[0],
                rho_v * un + p * normal[1],
                (E + p) * un,
            ]
        )

    def _calculate_S_star(self, U_L, U_R, q_L, q_R, S_L, S_R, normal):
        rho_L, u_L, v_L, p_L = q_L
        rho_R, u_R, v_R, p_R = q_R
        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        numerator = p_R - p_L + rho_L * un_L * (S_L - un_L) - rho_R * un_R * (S_R - un_R)
        denominator = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_U_star(self, U, q, S, S_star, normal, F):
        rho, u, v, p = q
        p_star = p + rho * (u * normal[0] + v * normal[1] - S) * (u * normal[0] + v * normal[1] - S_star)
        
        if (S - S_star) == 0:
            return U

        return (S * U - F + np.array([0, p_star * normal[0], p_star * normal[1], p_star * S_star])) / (S - S_star)

    def _calculate_wave_speed(self, U_cell):
        rho, rho_u, rho_v, E = U_cell
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        c = np.sqrt(self.gamma * p / rho)
        return np.sqrt(u**2 + v**2) + c

    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        return calculate_adaptive_dt(mesh, U, self._calculate_wave_speed, cfl_number)

    def _apply_wall_bc(self, U_inside, normal):
        rho, rho_u, rho_v, E = U_inside
        u = rho_u / rho
        v = rho_v / rho

        un = u * normal[0] + v * normal[1]
        ut = u * -normal[1] + v * normal[0]

        un_ghost = -un
        ut_ghost = ut

        u_ghost = un_ghost * normal[0] - ut_ghost * normal[1]
        v_ghost = un_ghost * normal[1] + ut_ghost * normal[0]

        return np.array([rho, rho * u_ghost, rho * v_ghost, E])