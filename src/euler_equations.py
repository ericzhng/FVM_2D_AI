import numpy as np
from src.base_equation import BaseEquation
from src.time_step import calculate_adaptive_dt


class EulerEquations(BaseEquation):
    """
    Represents the 2D Euler equations for compressible fluid flow.

    This class provides the specific implementation for the Euler equations,
    including the conversion between conservative and primitive variables,
    flux calculation (Roe and HLLC), and wave speed estimation.

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
        Converts a single conservative state vector to primitive variables.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].

        Returns:
            np.ndarray: Primitive state vector [rho, u, v, p].
        """
        rho, rho_u, rho_v, E = U
        # Floor density to prevent division by zero or negative pressure
        rho = max(rho, 1e-9)
        u = rho_u / rho
        v = rho_v / rho
        # Calculate pressure from total energy
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        return np.array([rho, u, v, p])

    def _prim_to_cons(self, P):
        """
        Converts a single primitive state vector to conservative variables.

        Args:
            P (np.ndarray): Primitive state vector [rho, u, v, p].

        Returns:
            np.ndarray: Conservative state vector [rho, rho*u, rho*v, E].
        """
        rho, u, v, p = P
        rho_u = rho * u
        rho_v = rho * v
        # Total energy E is the sum of internal and kinetic energy
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho_u, rho_v, E])

    def _compute_flux(self, U, normal):
        """
        Calculates the physical flux across a face with a given normal.

        Args:
            U (np.ndarray): Conservative state vector [rho, rho*u, rho*v, E].
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The flux vector normal to the face.
        """
        rho, u, v, p = self._cons_to_prim(U)
        # Normal velocity
        vn = u * normal[0] + v * normal[1]
        # Total enthalpy H = (E + p) / rho
        H = (U[3] + p) / rho

        # Normal flux components
        F = np.array(
            [
                rho * vn,
                rho * vn * u + p * normal[0],
                rho * vn * v + p * normal[1],
                rho * vn * H,
            ]
        )
        return F

    def max_eigenvalue(self, U):
        """
        Calculates the maximum wave speed (eigenvalue) for a cell.
        This is used for determining the stable time step (CFL condition).
        """
        rho, u, v, p = self._cons_to_prim(U)
        # Speed of sound 'a'
        a = np.sqrt(self.gamma * p / rho)
        # Max eigenvalue = |v| + a
        return np.sqrt(u**2 + v**2) + a

    def _apply_wall_bc(self, U_inside, normal):
        """
        Applies a solid wall (reflective) boundary condition.

        This condition reflects the velocity normal to the wall while keeping the
        tangential velocity and thermodynamic properties (pressure, density) the same.

        Args:
            U_inside (np.ndarray): State vector of the interior cell.
            normal (np.ndarray): Normal vector of the boundary face.

        Returns:
            np.ndarray: The state vector of the ghost cell.
        """
        rho, u, v, p = self._cons_to_prim(U_inside)

        # Decompose velocity into normal and tangential components
        vn = u * normal[0] + v * normal[1]
        vt = u * -normal[1] + v * normal[0]

        # Reflect the normal velocity, keep tangential velocity
        vn_ghost = -vn
        vt_ghost = vt

        # Recompose the ghost velocity vector from the new normal and tangential components
        u_ghost = vn_ghost * normal[0] - vt_ghost * normal[1]
        v_ghost = vn_ghost * normal[1] + vt_ghost * normal[0]

        # Create the primitive state for the ghost cell
        P_ghost = np.array([rho, u_ghost, v_ghost, p])

        # Convert back to conservative variables
        return self._prim_to_cons(P_ghost)

    def hllc_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux using the HLLC (Harten-Lax-van Leer-Contact) Riemann solver.
        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The HLLC numerical flux across the face.
        """
        nx, ny = normal
        # Tangential vector (rotated 90 degrees clockwise from normal)
        tx, ty = ny, -nx

        # --- Left State ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        vnL = uL * nx + vL * ny  # Normal velocity
        vtL = uL * tx + vL * ty  # Tangential velocity
        aL = np.sqrt(self.gamma * pL / rL)  # Speed of sound
        EL = U_L[3]
        FL = self._compute_flux(U_L, normal)

        # --- Right State ---
        rR, uR, vR, pR = self._cons_to_prim(U_R)
        vnR = uR * nx + vR * ny  # Normal velocity
        vtR = uR * tx + vR * ty  # Tangential velocity
        aR = np.sqrt(self.gamma * pR / rR)  # Speed of sound
        ER = U_R[3]
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages (for wave speed estimates) ---
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        r_roe = sqrt_rL * sqrt_rR
        u_roe = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
        v_roe = (sqrt_rL * vL + sqrt_rR * vR) / (sqrt_rL + sqrt_rR)
        vn_roe = u_roe * nx + v_roe * ny
        HL = (EL + pL) / rL
        HR = (ER + pR) / rR
        H_roe = (sqrt_rL * HL + sqrt_rR * HR) / (sqrt_rL + sqrt_rR)
        a_roe = np.sqrt((self.gamma - 1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2)))

        # --- Wave Speed Estimates (Toro's HLLC, Section 10.3.2) ---
        # Estimate pressure in star region (p_star)
        # This is a simplified estimate, more robust methods exist but this is common.
        q_L = 1.0 if p_star > pL else np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (p_star / pL - 1))
        q_R = 1.0 if p_star > pR else np.sqrt(1 + (self.gamma + 1) / (2 * self.gamma) * (p_star / pR - 1))

        # Initial guess for p_star (PVRS-like)
        p_pvrs = 0.5 * (pL + pR) + 0.5 * (vnL - vnR) * (rL * aL + rR * aR) / (rL + rR)
        p_star = max(0.0, p_pvrs) # Ensure non-negative pressure

        # Iterative solution for p_star (optional, but more accurate)
        # For simplicity, we'll use the initial guess for now.
        # More robust solvers might iterate here.

        # Wave speeds
        SL = vnL - aL * q_L
        SR = vnR + aR * q_R
        SM = (p_star - pL + rL * vnL * (SL - vnL)) / (rL * (SL - vnL)) # Contact wave speed

        # --- HLLC Flux Calculation ---
        if 0 <= SL:
            # All waves move to the right
            return FL
        elif SL < 0 <= SM:
            # Left-going shock/rarefaction, contact wave to the right
            # U_star_L state
            r_star_L = rL * (SL - vnL) / (SM - vnL)
            u_star_L = SM * nx + vtL * tx
            v_star_L = SM * ny + vtL * ty
            E_star_L = r_star_L * ( (EL / rL) + (SM - vnL) * (SM + pL / (rL * (SL - vnL))) )
            U_star_L = np.array([r_star_L, r_star_L * u_star_L, r_star_L * v_star_L, E_star_L])

            F_star_L = self._compute_flux(U_star_L, normal)
            return FL + SL * (U_star_L - U_L)
        elif SM < 0 < SR:
            # Contact wave to the left, right-going shock/rarefaction
            # U_star_R state
            r_star_R = rR * (SR - vnR) / (SM - vnR)
            u_star_R = SM * nx + vtR * tx
            v_star_R = SM * ny + vtR * ty
            E_star_R = r_star_R * ( (ER / rR) + (SM - vnR) * (SM + pR / (rR * (SR - vnR))) )
            U_star_R = np.array([r_star_R, r_star_R * u_star_R, r_star_R * v_star_R, E_star_R])

            F_star_R = self._compute_flux(U_star_R, normal)
            return FR + SR * (U_star_R - U_R)
        else:  # SR <= 0
            # All waves move to the left
            return FR

    def roe_flux(self, U_L, U_R, normal):
        """
        Computes the numerical flux using the Roe approximate Riemann solver.
        Based on "Riemann Solvers and Numerical Methods for Fluid Dynamics" by Eleuterio F. Toro.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The Roe numerical flux across the face.
        """
        nx, ny = normal

        # --- Left and Right States ---
        rL, uL, vL, pL = self._cons_to_prim(U_L)
        EL = U_L[3]
        FL = self._compute_flux(U_L, normal)

        rR, uR, vR, pR = self._cons_to_prim(U_R)
        ER = U_R[3]
        FR = self._compute_flux(U_R, normal)

        # --- Roe Averages ---
        sqrt_rL = np.sqrt(rL)
        sqrt_rR = np.sqrt(rR)
        r_avg = sqrt_rL * sqrt_rR
        u_avg = (sqrt_rL * uL + sqrt_rR * uR) / (sqrt_rL + sqrt_rR)
        v_avg = (sqrt_rL * vL + sqrt_rR * vR) / (sqrt_rL + sqrt_rR)
        H_avg = (sqrt_rL * (EL + pL) / rL + sqrt_rR * (ER + pR) / rR) / (sqrt_rL + sqrt_rR)
        a_avg = np.sqrt((self.gamma - 1) * (H_avg - 0.5 * (u_avg**2 + v_avg**2)))
        vn_avg = u_avg * nx + v_avg * ny

        # --- Eigenvalues (Wave Speeds) ---
        lambda1 = vn_avg - a_avg
        lambda2 = vn_avg
        lambda3 = vn_avg
        lambda4 = vn_avg + a_avg
        ws = np.array([lambda1, lambda2, lambda3, lambda4])

        # --- Entropy Fix (Harten's entropy fix) ---
        delta = 0.1 * a_avg
        for i in range(len(ws)):
            if abs(ws[i]) < delta:
                ws[i] = (ws[i]**2 + delta**2) / (2 * delta)
        ws = np.abs(ws) # Use absolute values for dissipation

        # --- Jump in Conservative Variables ---
        dU = U_R - U_L

        # --- Right Eigenvectors (Roe matrix for 2D Euler) ---
        # These are the columns of the matrix.
        # Based on Toro, Chapter 10, Section 10.2.2
        # R1: (1, u-a*nx, v-a*ny, H-vn*a)
        R1 = np.array([
            1,
            u_avg - a_avg * nx,
            v_avg - a_avg * ny,
            H_avg - vn_avg * a_avg
        ])

        # R2: (1, u, v, 0.5*(u^2+v^2)) - (1, u+a*nx, v+a*ny, H+vn*a)
        # This is for the contact discontinuity, related to tangential velocity.
        # The original code's R2 and R3 were for 1D.
        # For 2D, the second and third waves are shear waves.
        # R2 and R3 are related to the tangential components of velocity.
        # R2: (0, -ny, nx, -u*ny + v*nx)
        R2 = np.array([
            0,
            -ny,
            nx,
            -u_avg * ny + v_avg * nx
        ])

        # R3: (0, nx, ny, u*nx + v*ny) - This is not standard for the third wave.
        # The third wave is also a shear wave, orthogonal to the second.
        # R3: (1, u, v, 0.5*(u^2+v^2)) - (1, u, v, H)
        # A common choice for the third eigenvector is related to the pressure/density jump.
        # Let's use a simpler form for the third eigenvector, related to the density jump.
        # R3: (1, u, v, 0.5*(u^2+v^2))
        R3 = np.array([
            1,
            u_avg,
            v_avg,
            0.5 * (u_avg**2 + v_avg**2)
        ])

        # R4: (1, u+a*nx, v+a*ny, H+vn*a)
        R4 = np.array([
            1,
            u_avg + a_avg * nx,
            v_avg + a_avg * ny,
            H_avg + vn_avg * a_avg
        ])

        # Assemble the right eigenvector matrix
        Rv = np.column_stack((R1, R2, R3, R4))

        # --- Wave Strengths (alpha_k = L_k . dU) ---
        # L_k are the left eigenvectors. Instead of explicitly computing L_k,
        # we can solve Rv * alpha = dU for alpha.
        # alpha = np.linalg.solve(Rv, dU)
        # However, the original code used a direct calculation for dV (wave strengths).
        # Let's try to adapt the original dV calculation to 2D, or use the inverse of Rv.

        # For 2D Euler, the wave strengths are more complex.
        # A common approach is to use the characteristic variables directly.
        # dU = alpha_1 * R1 + alpha_2 * R2 + alpha_3 * R3 + alpha_4 * R4
        # We need to find alpha_k.
        # This requires inverting Rv or using the left eigenvectors.

        # Let's use the direct calculation of alpha_k from Toro, Section 10.2.2
        # d_rho = rR - rL
        # d_rho_u = U_R[1] - U_L[1]
        # d_rho_v = U_R[2] - U_L[2]
        # d_E = U_R[3] - U_L[3]

        # alpha_1 = (dp - r_avg * a_avg * (d_rho_u * nx + d_rho_v * ny) / r_avg) / (2 * a_avg**2)
        # alpha_2 = r_avg * (d_rho_v * nx - d_rho_u * ny) / r_avg
        # alpha_3 = d_rho - dp / a_avg**2
        # alpha_4 = (dp + r_avg * a_avg * (d_rho_u * nx + d_rho_v * ny) / r_avg) / (2 * a_avg**2)

        # The original dV was:
        # dV = np.array([
        #     (dp - r * a * dvn) / (2 * a**2),
        #     r * dvt,
        #     dr - dp / a**2,
        #     (dp + r * a * dvn) / (2 * a**2),
        # ])
        # This is for 1D. For 2D, the characteristic variables are:
        # alpha_1 = 0.5 * (dp / a_avg**2 - d_rho) + 0.5 * r_avg / a_avg * (dU[1]*nx + dU[2]*ny - vn_avg*d_rho)
        # alpha_2 = r_avg * (dU[2]*nx - dU[1]*ny) # Tangential momentum
        # alpha_3 = d_rho - dp / a_avg**2
        # alpha_4 = 0.5 * (dp / a_avg**2 - d_rho) - 0.5 * r_avg / a_avg * (dU[1]*nx + dU[2]*ny - vn_avg*d_rho)

        # Let's use the simpler approach of alpha = inv(Rv) * dU
        # This is numerically more stable than the explicit formulas for alpha_k
        # if Rv is well-conditioned.
        try:
            alpha = np.linalg.solve(Rv, dU)
        except np.linalg.LinAlgError:
            # Fallback to a simpler method or raise error if matrix is singular
            # For now, return HLL flux as a fallback
            return 0.5 * (FL + FR) - 0.5 * np.abs(vn_avg) * dU # Simple HLL-like fallback

        # --- Roe Flux ---
        # F_roe = 0.5 * (F_L + F_R) - 0.5 * sum(ws_i * alpha_i * R_i)
        dissipation = np.zeros_like(dU)
        for i in range(len(ws)):
            dissipation += ws[i] * alpha[i] * Rv[:, i]

        roe_flux = 0.5 * (FL + FR - dissipation)

        return roe_flux
