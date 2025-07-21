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

    def _compute_flux(self, U, normal):
        h, hu, hv = U
        h_q, u, v = self._cons_to_prim(U)
        un = u * normal[0] + v * normal[1]
        return np.array(
            [
                h * un,
                hu * un + 0.5 * self.g * h**2 * normal[0],
                hv * un + 0.5 * self.g * h**2 * normal[1],
            ]
        )

    def _compute_wave_speed(self, U, normal):
        h = U[0]
        c = np.sqrt(self.g * h)
        return c

    def max_eigenvalue(self, U_cell):
        h = U_cell[0]
        hu = U_cell[1]
        hv = U_cell[2]
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        c = np.sqrt(self.g * h) if h > 0 else 0
        return np.sqrt(u**2 + v**2) + c

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
