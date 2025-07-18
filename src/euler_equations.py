import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class EulerEquations(BaseEquation):
    """
    Represents the 2D Euler equations for compressible fluid flow.

    This class provides the specific implementation for the Euler equations,
    including the conversion between conservative and primitive variables,
    flux calculation, and wave speed estimation required by the HLLC solver.

    Attributes:
        gamma (float): The ratio of specific heats (adiabatic index).
    """

    def __init__(self, gamma=1.4):
        """
        Initializes the EulerEquations object.

        Args:
            gamma (float, optional): The ratio of specific heats. Defaults to 1.4.
        """
        self.gamma = gamma

    def _cons_to_prim(self, U):
        """
        Converts conservative variables to primitive variables for the Euler equations.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            np.ndarray: Primitive state vector [rho, u, v, p].
        """
        rho, rho_u, rho_v, E = U
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _calculate_wave_speeds(self, q_L, q_R, normal):
        """
        Estimates the wave speeds (S_L, S_R) for the HLLC solver using the Roe average.

        Args:
            q_L (np.ndarray): Primitive state vector of the left cell.
            q_R (np.ndarray): Primitive state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            tuple: A tuple containing the left and right wave speeds (S_L, S_R).
        """
        rho_L, u_L, v_L, p_L = q_L
        rho_R, u_R, v_R, p_R = q_R

        # Normal velocities
        un_L = u_L * normal[0] + v_L * normal[1]
        un_R = u_R * normal[0] + v_R * normal[1]

        # Sound speeds
        c_L = np.sqrt(self.gamma * p_L / rho_L)
        c_R = np.sqrt(self.gamma * p_R / rho_R)

        # Estimate of wave speeds (Davis, 1988)
        S_L = min(un_L - c_L, un_R - c_R)
        S_R = max(un_L + c_L, un_R + c_R)

        return S_L, S_R

    def _calculate_flux(self, U, q, normal):
        """
        Calculates the physical flux for the Euler equations.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].
            q (np.ndarray): Primitive state vector [rho, u, v, p].
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The physical flux vector.
        """
        rho, rho_u, rho_v, E = U
        _, u, v, p = q
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
        """
        Calculates the contact wave speed (S_star) for the HLLC solver.
        """
        _, _, _, p_L = q_L
        _, _, _, p_R = q_R
        rho_L, rho_u_L, rho_v_L, _ = U_L
        rho_R, rho_u_R, rho_v_R, _ = U_R

        un_L = (rho_u_L * normal[0] + rho_v_L * normal[1]) / rho_L
        un_R = (rho_u_R * normal[0] + rho_v_R * normal[1]) / rho_R

        numerator = p_R - p_L + rho_L * un_L * (S_L - un_L) - rho_R * un_R * (S_R - un_R)
        denominator = rho_L * (S_L - un_L) - rho_R * (S_R - un_R)

        # Avoid division by zero
        if abs(denominator) < 1e-9:
            return 0.0

        return numerator / denominator

    def _calculate_U_star(self, U, q, S, S_star, normal, F):
        """
        Calculates the state in the star region (U_star) for the HLLC solver.
        """
        rho, _, _, p = q
        un = (U[1] * normal[0] + U[2] * normal[1]) / rho

        # Pressure in the star region
        p_star = p + rho * (un - S) * (un - S_star)

        # Avoid division by zero
        if abs(S - S_star) < 1e-9:
            return U

        # State vector in the star region
        U_star = (
            S * U
            - F
            + np.array([0, p_star * normal[0], p_star * normal[1], p_star * S_star])
        ) / (S - S_star)

        return U_star

    def _calculate_wave_speed(self, U_cell):
        """
        Calculates the maximum wave speed in a cell for the CFL condition.
        """
        rho, rho_u, rho_v, E = U_cell
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        c = np.sqrt(self.gamma * p / rho)
        return np.sqrt(u**2 + v**2) + c

    def calculate_adaptive_dt(self, mesh, U, cfl_number):
        """
        Calculates the adaptive time step for the Euler equations.
        """
        return calculate_adaptive_dt(mesh, U, self._calculate_wave_speed, cfl_number)

    def _apply_wall_bc(self, U_inside, normal):
        """
        Applies a solid wall (reflective) boundary condition for the Euler equations.

        This condition reflects the velocity normal to the wall while keeping the
        tangential velocity and thermodynamic properties the same.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        rho, rho_u, rho_v, E = U_inside
        u = rho_u / rho
        v = rho_v / rho

        # Decompose velocity into normal and tangential components
        un = u * normal[0] + v * normal[1]
        ut = u * -normal[1] + v * normal[0]

        # Reflect the normal velocity
        un_ghost = -un
        ut_ghost = ut

        # Recompose the ghost velocity vector
        u_ghost = un_ghost * normal[0] - ut_ghost * normal[1]
        v_ghost = un_ghost * normal[1] + ut_ghost * normal[0]

        # Ghost cell state with reflected velocity
        return np.array([rho, rho * u_ghost, rho * v_ghost, E])
