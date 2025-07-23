
# HLLC Riemann Solver for 2D Euler Equations

The HLLC (Harten-Lax-van Leer-Contact) Riemann solver is an extension of the HLL solver that incorporates the contact discontinuity, leading to improved resolution of shear waves and contact surfaces.

The Euler equations in conservative form are: $$ \frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} + \frac{\partial \mathbf{G}(\mathbf{U})}{\partial y} = 0 $$ where $\mathbf{U} = [\rho, \rho u, \rho v, E]^T$ is the vector of conservative variables, and $\mathbf{F}$ and $\mathbf{G}$ are the flux vectors in the x and y directions, respectively.

For a face with normal vector $\mathbf{n} = (n_x, n_y)$, the flux across the face is given by $\mathbf{F}_n = \mathbf{F} n_x + \mathbf{G} n_y$.

The numerical flux $F_{HLLC}$ is given by: 

$$ F_{HLLC} = \begin{cases} F_L & \text{if } 0 \le S_L \ F_L + S_L (U_L^* - U_L) & \text{if } S_L < 0 \le S_M \ F_R + S_R (U_R^* - U_R) & \text{if } S_M < 0 < S_R \ F_R & \text{if} S_R \le 0 \end{cases} $$

1. Primitive Variables: Convert conservative variables $U = [\rho, \rho u, \rho v, E]^T$ to primitive variables $P = [\rho, u, v, p]^T$: $$ u = \frac{\rho u}{\rho}, \quad v = \frac{\rho v}{\rho} $$ $$ p = (\gamma - 1) \left( E - \frac{1}{2} \rho (u^2 + v^2) \right) $$ Speed of sound: $a = \sqrt{\frac{\gamma p}{\rho}}$

2. Normal and Tangential Velocities: For a given normal vector $\mathbf{n} = (n_x, n_y)$ and tangential vector $\mathbf{t} = (n_y, -n_x)$: $$ v_n = u n_x + v n_y $$ $$ v_t = u n_y - v n_x $$

3. Wave Speed Estimates ($S_L, S_R, S_M$): The wave speeds are estimated using the Roe averages. 

    Roe-averaged density: $\tilde{\rho} = \sqrt{\rho_L \rho_R}$ 

    Roe-averaged velocities: $\tilde{u} = \frac{\sqrt{\rho_L} u_L + \sqrt{\rho_R} u_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$, $\tilde{v} = \frac{\sqrt{\rho_L} v_L + \sqrt{\rho_R} v_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$

    Roe-averaged enthalpy: $\tilde{H} = \frac{\sqrt{\rho_L} H_L + \sqrt{\rho_R} H_R}{\sqrt{\rho_L} + \sqrt{\rho_R}}$, where $H = (E+p)/\rho$ 

    Roe-averaged speed of sound: $\tilde{a} = \sqrt{(\gamma - 1) \left( \tilde{H} - \frac{1}{2} (\tilde{u}^2 + \tilde{v}^2) \right)}$ 

    Roe-averaged normal velocity: $\tilde{v}_n = \tilde{u} n_x + \tilde{v} n_y$

    Initial estimates for $S_L$ and $S_R$ (Davis-Einfeldt estimates): $$ S_L = \min(v_{nL} - a_L, \tilde{v}n - \tilde{a}) $$ $$ S_R = \max(v{nR} + a_R, \tilde{v}_n + \tilde{a}) $$

    The contact wave speed $S_M$ is derived from the pressure balance across the contact discontinuity. A common approach is to use an iterative solver for the pressure in the star region, or a simplified estimate. The code uses a simplified estimate for $p$: $$ p = \max\left(0, \frac{p_L (\rho_R S_R - \rho_L S_L) + \rho_L \rho_R (S_R - S_L) v_{nL} v_{nR}}{(\rho_R S_R - \rho_L S_L) + \rho_L \rho_R (S_R - S_L)} \right) $$ 
    
    Then, $S_M$ can be calculated from the jump conditions.

4. Star States ($U_L^, U_R^*$):* The star states are calculated based on the jump conditions across the waves. For example, for $U_L$: 
    $$ \rho_L = \rho_L \frac{S_L - v_{nL}}{S_L - S_M} $$ $$ u_L^* = S_M n_x + v_{tL} t_x $$ $$ v_L^* = S_M n_y + v_{tL} t_y $$ $$ E_L^* = \rho_L^* \left( \frac{E_L}{\rho_L} + (S_M - v_{nL}) \left( S_M + \frac{p_L}{\rho_L (S_L - v_{nL})} \right) \right) $$

    And similarly for $U_R^*$.


# Roe Riemann Solver for 2D Euler Equations

The Roe solver is an approximate Riemann solver that linearizes the Euler equations and solves the resulting linear Riemann problem exactly. It is known for its high resolution of contact discontinuities.

The Roe flux $F_{Roe}$ is given by: $$ F_{Roe} = \frac{1}{2} (F_L + F_R) - \frac{1}{2} \sum_{k=1}^4 |\tilde{\lambda}_k| \tilde{\alpha}_k \tilde{r}_k $$ where $\tilde{\lambda}_k$ are the Roe-averaged eigenvalues, $\tilde{\alpha}_k$ are the wave strengths, and $\tilde{r}_k$ are the Roe-averaged right eigenvectors.

1. Roe Averages: The Roe averages are computed as described for the HLLC solver.

2. Eigenvalues (Wave Speeds): The eigenvalues of the Roe matrix are: $$ \tilde{\lambda}_1 = \tilde{v}_n - \tilde{a} $$ $$ \tilde{\lambda}_2 = \tilde{v}_n $$ $$ \tilde{\lambda}_3 = \tilde{v}_n $$ $$ \tilde{\lambda}_4 = \tilde{v}_n + \tilde{a} $$ An entropy fix (e.g., Harten's entropy fix) is applied to the eigenvalues to prevent expansion shocks.

3. Right Eigenvectors: The right eigenvectors $\tilde{r}_k$ are the columns of the Roe matrix. For 2D Euler equations, these are: $$ \tilde{r}_1 = \begin{pmatrix} 1 \ \tilde{u} - \tilde{a} n_x \ \tilde{v} - \tilde{a} n_y \ \tilde{H} - \tilde{v}_n \tilde{a} \end{pmatrix} $$ $$ \tilde{r}_2 = \begin{pmatrix} 0 \ -n_y \ n_x \ -\tilde{u} n_y + \tilde{v} n_x \end{pmatrix} $$ $$ \tilde{r}_3 = \begin{pmatrix} 1 \ \tilde{u} \ \tilde{v} \ \frac{1}{2}(\tilde{u}^2 + \tilde{v}^2) \end{pmatrix} $$ $$ \tilde{r}_4 = \begin{pmatrix} 1 \ \tilde{u} + \tilde{a} n_x \ \tilde{v} + \tilde{a} n_y \ \tilde{H} + \tilde{v}_n \tilde{a} \end{pmatrix} $$

4. Wave Strengths ($\tilde{\alpha}_k$): The wave strengths are determined by projecting the jump in conservative variables $\Delta U = U_R - U_L$ onto the left eigenvectors, or by solving the linear system: $$ \Delta U = \sum_{k=1}^4 \tilde{\alpha}_k \tilde{r}_k $$ This can be solved by $\tilde{\alpha} = \tilde{R}^{-1} \Delta U$, where $\tilde{R}$ is the matrix of right eigenvectors.


These formulas provide a theoretical basis for the implemented HLLC and Roe Riemann solvers. The code implements these concepts, with careful attention to the 2D nature of the problem and the specific forms of the equations and boundary conditions.
