from abc import ABC, abstractmethod
import numpy as np


class BaseEquation(ABC):
    """
    Abstract base class for defining the governing equations for a Finite Volume Method solver.
    This class provides the framework for the HLLC Riemann solver and requires subclasses
    to implement the specific physics (e.g., Euler, Shallow Water).
    """

    def hllc_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux across a face using the HLLC (Harten-Lax-van Leer-Contact)
        Riemann solver.

        Args:
            U_L (np.ndarray): State vector of the left cell.
            U_R (np.ndarray): State vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The numerical flux across the face.
        """
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
    def _calculate_flux(self, U, q, normal):
        """
        Calculates the physical flux for a given state.

        Args:
            U (np.ndarray): Conservative state vector.
            q (np.ndarray): Primitive state vector.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The physical flux.
        """
        pass

    @abstractmethod
    def _cons_to_prim(self, U):
        """
        Converts conservative variables to primitive variables.

        Args:
            U (np.ndarray): Conservative state vector.

        Returns:
            np.ndarray: Primitive state vector.
        """
        pass

    @abstractmethod
    def _calculate_wave_speeds(self, q_L, q_R, normal):
        """
        Calculates the wave speeds for the HLLC solver.

        Args:
            q_L (np.ndarray): Primitive state vector of the left cell.
            q_R (np.ndarray): Primitive state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            tuple: A tuple containing the left and right wave speeds (S_L, S_R).
        """
        pass

    @abstractmethod
    def _calculate_S_star(self, U_L, U_R, q_L, q_R, S_L, S_R, normal):
        """
        Calculates the contact wave speed (S_star) for the HLLC solver.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            q_L (np.ndarray): Primitive state vector of the left cell.
            q_R (np.ndarray): Primitive state vector of the right cell.
            S_L (float): Left wave speed.
            S_R (float): Right wave speed.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            float: The contact wave speed.
        """
        pass

    @abstractmethod
    def _calculate_U_star(self, U, q, S, S_star, normal, F):
        """
        Calculates the state in the star region of the Riemann fan.

        Args:
            U (np.ndarray): Conservative state vector.
            q (np.ndarray): Primitive state vector.
            S (float): Wave speed (S_L or S_R).
            S_star (float): Contact wave speed.
            normal (np.ndarray): Normal vector of the face.
            F (np.ndarray): Physical flux.

        Returns:
            np.ndarray: The state in the star region.
        """
        pass

    @abstractmethod
    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        """
        Calculates the adaptive time step based on the CFL condition.

        Args:
            mesh (Mesh): The mesh object.
            U (np.ndarray): The conservative state vector for all cells.
            cfl_number (float): The Courant-Friedrichs-Lewy number.

        Returns:
            float: The adaptive time step.
        """
        pass

    def apply_boundary_condition(self, U_inside, normal, bc_info):
        """
        Applies a boundary condition to a ghost cell state.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.
            bc_info (dict): Dictionary containing boundary condition information.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        bc_type = bc_info.get("type", "wall")

        if bc_type == "wall":
            return self._apply_wall_bc(U_inside, normal)
        elif bc_type == "inlet":
            return bc_info.get("value", U_inside)
        elif bc_type == "outlet":
            return U_inside  # For outlet, ghost cell has the same state as the interior
        else:
            # Default to wall boundary condition if type is unknown
            return self._apply_wall_bc(U_inside, normal)

    @abstractmethod
    def _apply_wall_bc(self, U_inside, normal):
        """
        Applies a solid wall boundary condition.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        pass
