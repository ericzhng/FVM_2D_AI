import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class ShallowWaterEquations(BaseEquation):
    def __init__(self, g=9.81):
        self.g = g

    def _cons_to_prim(self, U):
        h, hu, hv = U
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        return np.array([h, u, v])

    def _calculate_wave_speeds(self, q_L, q_R, normal):
        h_L, u_L, v_L = q_L
        h_R, u_R, v_R = q_R

        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        c_L = np.sqrt(self.g * h_L) if h_L > 0 else 0
        c_R = np.sqrt(self.g * h_R) if h_R > 0 else 0

        # Roe average speeds
        roe_avg_h = 0.5 * (h_L + h_R)
        sqrt_h_L = np.sqrt(h_L)
        sqrt_h_R = np.sqrt(h_R)
        
        if (sqrt_h_L + sqrt_h_R) == 0:
            roe_avg_u = 0
            roe_avg_v = 0
        else:
            roe_avg_u = (sqrt_h_L * u_L + sqrt_h_R * u_R) / (sqrt_h_L + sqrt_h_R)
            roe_avg_v = (sqrt_h_L * v_L + sqrt_h_R * v_R) / (sqrt_h_L + sqrt_h_R)

        roe_avg_un = roe_avg_u * normal[0] + roe_avg_v * normal[1]
        roe_avg_c = np.sqrt(self.g * roe_avg_h)

        S_L = min(un_L - c_L, roe_avg_un - roe_avg_c)
        S_R = max(un_R + c_R, roe_avg_un + roe_avg_c)

        return S_L, S_R

    def _calculate_flux(self, U, q, normal):
        h, hu, hv = U
        h_q, u, v = q
        un = u * normal[0] + v * normal[1]
        return np.array(
            [
                h * un,
                hu * un + 0.5 * self.g * h**2 * normal[0],
                hv * un + 0.5 * self.g * h**2 * normal[1],
            ]
        )

    def _calculate_S_star(self, U_L, U_R, q_L, q_R, S_L, S_R, normal):
        h_L, u_L, v_L = q_L
        h_R, u_R, v_R = q_R
        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        numerator = S_R * h_R * (un_R - S_R) - S_L * h_L * (un_L - S_L) + 0.5 * self.g * (h_L**2 - h_R**2)
        denominator = h_R * (un_R - S_R) - h_L * (un_L - S_L)
        
        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _calculate_U_star(self, U, q, S, S_star, normal, F):
        if (S - S_star) == 0:
            return U
        return (S * U - F) / (S - S_star)

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

    def _apply_wall_bc(self, U_inside, normal):
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