from abc import ABC, abstractmethod
import numpy as np


class BaseEquation(ABC):
    """
    Abstract base class for defining the governing equations for a Finite Volume Method solver.
    This class provides the framework for the HLLC Riemann solver and requires subclasses
    to implement the specific physics (e.g., Euler, Shallow Water).
    """

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

    def cons_to_prim_batch(self, U_batch):
        """
        Converts a batch of conservative variables to primitive variables.

        Args:
            U_batch (np.ndarray): Array of conservative state vectors.

        Returns:
            np.ndarray: Array of primitive state vectors.
        """
        return np.apply_along_axis(self._cons_to_prim, 1, U_batch)

    @abstractmethod
    def _compute_flux(self, U, normal):
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
    def max_eigenvalue(self, U):
        """
        Calculates the maximum wave speed in a cell for the CFL condition.
        """
        pass

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
