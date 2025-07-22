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
        rho = max(rho, 1e-6)  # Avoid division by zero
        # Calculate primitive variables
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _compute_flux(self, U, normal):
        """
        Calculates the physical flux vectors (F and G) for the Euler equations.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the F and G flux vectors.
        """
        rho, u, v, p = self._cons_to_prim(U)
        vnL = u * normal[0] + v * normal[1]
        HL = (U[3] + p) / rho

        # Left and Right fluxes (normal flux)
        F = np.array(
            [
                rho * vnL,
                rho * vnL * u + p * normal[0],
                rho * vnL * v + p * normal[1],
                rho * vnL * HL,
            ]
        )

        return F

    def max_eigenvalue(self, U):
        """
        Calculates the maximum wave speed in a cell for the CFL condition.
        """
        rho, u, v, p = self._cons_to_prim(U)
        c = np.sqrt(self.gamma * p / rho)
        return np.sqrt(u**2 + v**2) + c

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

    def _compute_flux_2d(
        self,
        U,
    ):
        """
        Calculates the physical flux vectors (F and G) for the Euler equations.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the F and G flux vectors.
        """
        rho, rho_u, rho_v, E = U
        rho = max(rho, 1e-9)  # Avoid division by zero
        u = rho_u / rho
        v = rho_v / rho
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))

        F = np.array(
            [
                rho_u,
                rho_u * u + p,
                rho_v * u,
                (E + p) * u,
            ]
        )

        G = np.array(
            [
                rho_v,
                rho_u * v,
                rho_v * v + p,
                (E + p) * v,
            ]
        )

        return F, G

    def hllc_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux across a face using the HLLC (Harten-Lax-van Leer-Contact)
        Riemann solver.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The numerical flux across the face.
        """
        # Compute HLLC flux
        nx, ny = normal  # Normal vectors

        # Left state
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny
        aL = np.sqrt(self.gamma * pL / rL)
        HL = (U_L[3] + pL) / rL

        # Right state
        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny
        aR = np.sqrt(self.gamma * pR / rR)
        HR = (U_R[3] + pR) / rR

        # Left and Right fluxes (normal flux)
        FL = np.array(
            [rL * vnL, rL * vnL * uL + pL * nx, rL * vnL * vL + pL * ny, rL * vnL * HL]
        )
        FR = np.array(
            [rR * vnR, rR * vnR * uR + pR * nx, rR * vnR * vR + pR * ny, rR * vnR * HR]
        )

        # Compute guess pressure from PVRS Riemann solver
        PPV = max(
            0, 0.5 * (pL + pR) + 0.5 * (vnL - vnR) * (0.25 * (rL + rR) * (aL + aR))
        )
        pmin = min(pL, pR)
        pmax = max(pL, pR)
        Qmax = pmax / pmin
        Quser = 2.0  # Parameter manually set

        if (Qmax <= Quser) and (pmin <= PPV <= pmax):
            # Select PVRS Riemann solver
            pM = PPV
        elif PPV < pmin:
            # Select Two-Rarefaction Riemann solver
            PQ = (pL / pR) ** ((self.gamma - 1) / (2 * self.gamma))
            uM = (PQ * vnL / aL + vnR / aR + 2 / (self.gamma - 1) * (PQ - 1)) / (
                PQ / aL + 1 / aR
            )
            PTL = 1 + (self.gamma - 1) / 2 * (vnL - uM) / aL
            PTR = 1 + (self.gamma - 1) / 2 * (uM - vnR) / aR
            pM = 0.5 * (
                pL * PTL ** (2 * self.gamma / (self.gamma - 1))
                + pR * PTR ** (2 * self.gamma / (self.gamma - 1))
            )
        else:
            # Use Two-Shock Riemann solver with PVRS as estimate
            GEL = np.sqrt(
                (2 / (self.gamma + 1) / rL)
                / ((self.gamma - 1) / (self.gamma + 1) * pL + PPV)
            )
            GER = np.sqrt(
                (2 / (self.gamma + 1) / rR)
                / ((self.gamma - 1) / (self.gamma + 1) * pR + PPV)
            )
            pM = (GEL * pL + GER * pR - (vnR - vnL)) / (GEL + GER)

        # Estimate wave speeds: SL, SR, SM
        zL = (
            np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pL - 1))
            if pM > pL
            else 1
        )
        zR = (
            np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (pM / pR - 1))
            if pM > pR
            else 1
        )
        SL = vnL - aL * zL
        SR = vnR + aR * zR
        SM = (pL - pR + rR * vnR * (SR - vnR) - rL * vnL * (SL - vnL)) / (
            rR * (SR - vnR) - rL * (SL - vnL)
        )

        # Compute the HLLC flux
        if 0 <= SL:  # Right-going supersonic flow
            HLLC = FL
        elif SL <= 0 <= SM:  # Subsonic flow to the right
            U_star_L = (
                rL
                * (SL - vnL)
                / (SL - SM)
                * np.array(
                    [
                        1,
                        SM * nx + uL * abs(ny),
                        SM * ny + vL * abs(nx),
                        U_L[3] / rL + (SM - vnL) * (SM + pL / (rL * (SL - vnL))),
                    ]
                )
            )
            HLLC = FL + SL * (U_star_L - U_L)
        elif SM <= 0 <= SR:  # Subsonic flow to the left
            U_star_R = (
                rR
                * (SR - vnR)
                / (SR - SM)
                * np.array(
                    [
                        1,
                        SM * nx + uR * abs(ny),
                        SM * ny + vR * abs(nx),
                        U_R[3] / rR + (SM - vnR) * (SM + pR / (rR * (SR - vnR))),
                    ]
                )
            )
            HLLC = FR + SR * (U_star_R - U_R)
        else:  # Left-going supersonic flow
            HLLC = FR

        return HLLC

    def roe_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux across a face using the HLLC (Harten-Lax-van Leer-Contact)
        Riemann solver.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The numerical flux across the face.
        """
        # Compute Roe flux
        nx, ny = normal  # Normal vectors
        tx, ty = -ny, nx  # Tangent vectors

        # Left state
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny
        vtL = uL * tx + vL * ty
        HL = (U_L[3] + pL) / rL

        # Right state
        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny
        vtR = uR * tx + vR * ty
        HR = (U_R[3] + pR) / rR

        # Roe Averages
        RT = np.sqrt(rR / rL)
        r = RT * rL
        u = (uL + RT * uR) / (1 + RT)
        v = (vL + RT * vR) / (1 + RT)
        H = (HL + RT * HR) / (1 + RT)
        a = np.sqrt((self.gamma - 1) * (H - (u**2 + v**2) / 2))
        vn = u * nx + v * ny
        vt = u * tx + v * ty

        # Wave Strengths
        dr = rR - rL
        dp = pR - pL
        dvn = vnR - vnL
        dvt = vtR - vtL
        dV = np.array(
            [
                (dp - r * a * dvn) / (2 * a**2),
                r * dvt / a,
                dr - dp / (a**2),
                (dp + r * a * dvn) / (2 * a**2),
            ]
        )

        # Wave Speed
        ws = np.array([abs(vn - a), abs(vn), abs(vn), abs(vn + a)])

        # Harten's Entropy Fix
        dws = np.array([1 / 5, 0, 0, 1 / 5])
        ws[0] = (ws[0] ** 2 / dws[0] + dws[0]) / 2 if ws[0] < dws[0] else ws[0]
        ws[3] = (ws[3] ** 2 / dws[3] + dws[3]) / 2 if ws[3] < dws[3] else ws[3]

        # Right Eigenvectors
        Rv = np.array(
            [
                [1, 0, 1, 1],
                [u - a * nx, a * tx, u, u + a * nx],
                [u - a * ny, a * ty, u, u + a * ny],
                [H - vn * a, vt * a, (u**2 + v**2) / 2, H + vn * a],
            ]
        )

        # Left and Right fluxes
        FL = np.array(
            [rL * vnL, rL * vnL * uL + pL * nx, rL * vnL * vL + pL * ny, rL * vnL * HL]
        )
        FR = np.array(
            [rR * vnR, rR * vnR * uR + pR * nx, rR * vnR * vR + pR * ny, rR * vnR * HR]
        )

        # Dissipation Term
        Roe = (FL + FR - Rv @ (ws * dV)) / 2

        return Roe
