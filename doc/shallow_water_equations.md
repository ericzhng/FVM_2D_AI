# 2D Shallow Water Equations for Finite Volume Solver

## Governing Equations
The 2D shallow water equations in conservative form are:

\[
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}}{\partial x} + \frac{\partial \mathbf{G}}{\partial y} = \mathbf{S}
\]

Where:
- Conserved variables: \(\mathbf{U} = \begin{bmatrix} h \\ hu \\ hv \end{bmatrix}\)
  - \(h\): water depth
  - \(hu\): x-momentum
  - \(hv\): y-momentum
- Fluxes:
  - \(\mathbf{F} = \begin{bmatrix} hu \\ hu^2 + \frac{1}{2}gh^2 \\ huv \end{bmatrix}\)
  - \(\mathbf{G} = \begin{bmatrix} hv \\ huv \\ hv^2 + \frac{1}{2}gh^2 \end{bmatrix}\)
- Source term: \(\mathbf{S} = \begin{bmatrix} 0 \\ -gh \frac{\partial z}{\partial x} \\ -gh \frac{\partial z}{\partial y} \end{bmatrix}\)
  - \(g\): gravitational acceleration
  - \(z\): bed elevation

## Riemann Problem
For the Riemann problem, define a discontinuity across a cell interface:
- Left state: \(\mathbf{U}_L = \begin{bmatrix} h_L \\ (hu)_L \\ (hv)_L \end{bmatrix}\)
- Right state: \(\mathbf{U}_R = \begin{bmatrix} h_R \\ (hu)_R \\ (hv)_R \end{bmatrix}\)
- Initial condition: Specify different \(h\), \(hu\), \(hv\) on either side of a line (e.g., \(x = 0\)).

## Roe Flux
The Roe flux for the interface between left and right states is:

\[
\mathbf{F}_{Roe} = \frac{1}{2} \left( \mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R) \right) - \frac{1}{2} \sum_{k=1}^3 |\tilde{\lambda}_k| \tilde{\mathbf{K}}_k \tilde{\alpha}_k
\]

Where:
- \(\tilde{\lambda}_k\): eigenvalues of the Roe-averaged Jacobian
- \(\tilde{\mathbf{K}}_k\): eigenvectors
- \(\tilde{\alpha}_k\): wave strengths
- Roe averages:
  - \(\tilde{u} = \frac{\sqrt{h_L}u_L + \sqrt{h_R}u_R}{\sqrt{h_L} + \sqrt{h_R}}\)
  - \(\tilde{v} = \frac{\sqrt{h_L}v_L + \sqrt{h_R}v_R}{\sqrt{h_L} + \sqrt{h_R}}\)
  - \(\tilde{h} = \sqrt{h_L h_R}\)
- Eigenvalues: \(\tilde{\lambda}_1 = \tilde{u} - \sqrt{gh}\), \(\tilde{\lambda}_2 = \tilde{u}\), \(\tilde{\lambda}_3 = \tilde{u} + \sqrt{gh}\)
- Eigenvectors: \(\tilde{\mathbf{K}}_1 = \begin{bmatrix} 1 \\ \�o\tilde{u} - \sqrt{gh} \\ \tilde{v} \end{bmatrix}\), \(\tilde{\mathbf{K}}_2 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}\), \(\tilde{\mathbf{K}}_3 = \begin{bmatrix} 1 \\ \tilde{u} + \sqrt{gh} \\ \tilde{v} \end{bmatrix}\)
- Wave strengths: Solve \(\mathbf{U}_R - \mathbf{U}_L = \sum_{k=1}^3 \tilde{\alpha}_k \tilde{\mathbf{K}}_k\)

## MUSCL Reconstruction
For second-order accuracy, use MUSCL (Monotonic Upstream-centered Scheme for Conservation Laws):
- Reconstruct \(\mathbf{U}\) at cell interfaces using linear interpolation.
- Slope limiter (e.g., minmod, superbee) to ensure monotonicity:
  \[
  \mathbf{U}_{i+1/2,L} = \mathbf{U}_i + \frac{1}{2} \phi(\mathbf{r}) (\mathbf{U}_i - \mathbf{U}_{i-1})
  \]
  \[
  \mathbf{U}_{i+1/2,R} = \mathbf{U}_{i+1} - \frac{1}{2} \phi(\mathbf{r}) (\mathbf{U}_{i+2} - \mathbf{U}_{i+1})
  \]
  Where \(\phi(\mathbf{r})\) is the limiter function, and \(\mathbf{r}\) is the ratio of consecutive gradients.

## Mesh Handling
- Import unstructured mesh (quadrilateral/triangular) from a standard format (e.g., Gmsh .msh).
- Store cell connectivity, face normals, and areas.
- Compute fluxes across each face using Roe solver and MUSCL-reconstructed states.