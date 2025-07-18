from abc import ABC, abstractmethod
import numpy as np

class BaseEquation(ABC):
    def hllc_flux(self, U_L, U_R, normal):
        q_L = self._cons_to_prim(U_L)
        q_R = self._cons_to_prim(U_R)

        S_L, S_R = self._calculate_wave_speeds(q_L, q_R, normal)

        F_L = self._calculate_flux(U_L, q_L, normal)
        F_R = self._calculate_flux(U_R, q_R, normal)

        if S_L >= 0:
            return F_L
        if S_R <= 0:
            return F_R

        S_star = self._calculate_S_star(U_L, U_R, q_L, q_R, S_L, S_R, normal)

        U_star_L = self._calculate_U_star(U_L, q_L, S_L, S_star, normal, F_L)
        U_star_R = self._calculate_U_star(U_R, q_R, S_R, S_star, normal, F_R)

        if S_star >= 0:
            return F_L + S_L * (U_star_L - U_L)
        else:
            return F_R + S_R * (U_star_R - U_R)

    @abstractmethod
    def _cons_to_prim(self, U):
        pass

    @abstractmethod
    def _calculate_wave_speeds(self, q_L, q_R, normal):
        pass

    @abstractmethod
    def _calculate_flux(self, U, q, normal):
        pass

    @abstractmethod
    def _calculate_S_star(self, U_L, U_R, q_L, q_R, S_L, S_R, normal):
        pass

    @abstractmethod
    def _calculate_U_star(self, U, q, S, S_star, normal, F):
        pass

    @abstractmethod
    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        pass

    def apply_boundary_condition(self, U_inside, normal, bc_info):
        bc_type = bc_info.get("type", "wall")

        if bc_type == "wall":
            return self._apply_wall_bc(U_inside, normal)
        elif bc_type == "inlet":
            return bc_info.get("value", U_inside)
        elif bc_type == "outlet":
            return U_inside
        else:
            return self._apply_wall_bc(U_inside, normal)

    @abstractmethod
    def _apply_wall_bc(self, U_inside, normal):
        pass
