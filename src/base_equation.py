from abc import ABC, abstractmethod

class BaseEquation(ABC):
    @abstractmethod
    def hllc_flux(self, U_L, U_R, normal):
        pass

    @abstractmethod
    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        pass

    @abstractmethod
    def apply_boundary_condition(self, U_inside, normal, bc_info):
        pass
