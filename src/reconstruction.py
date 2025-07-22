import numpy as np
from src.mesh import Mesh


# --- Limiter Functions ---
def barth_jespersen_limiter(r):
    """Barth-Jespersen limiter function."""
    return np.minimum(1, r)


def minmod_limiter(r):
    """Minmod limiter function."""
    return np.maximum(0, np.minimum(1, r))


def superbee_limiter(r):
    """Superbee limiter function."""
    return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))


LIMITERS = {
    "barth_jespersen": barth_jespersen_limiter,
    "minmod": minmod_limiter,
    "superbee": superbee_limiter,
}


def compute_gradients_gaussian(mesh: Mesh, U, over_relaxation=1.2):
    """
    Computes gradients at cell centroids using the Gaussian method.

    This method iterates over each face of a cell, calculates the value of the
    variable at the face, and then uses the divergence theorem to approximate
    the gradient.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        over_relaxation (float, optional): Over-relaxation factor for non-orthogonal
                                         correction. Defaults to 1.2.

    Returns:
        np.ndarray: The gradients of the state variables at each cell centroid.
    """
    nvars = U.shape[1]
    gradients = np.zeros((mesh.nelem, nvars, 2))  # For 2D gradients (x, y)

    for i in range(mesh.nelem):
        grad_sum = np.zeros((nvars, 2))
        for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
            face_normal = mesh.face_normals[i, j]
            face_area = mesh.face_areas[i, j]

            if neighbor_idx != -1:
                d_i, d_j = mesh.face_to_cell_distances[i, j]

                if d_i + d_j > 1e-9:
                    w_i = d_j / (d_i + d_j)
                    w_j = d_i / (d_i + d_j)
                    U_face = w_i * U[i] + w_j * U[neighbor_idx]
                else:
                    # Fallback to simple average if distances are zero
                    U_face = 0.5 * (U[i] + U[neighbor_idx])

                # Non-orthogonal correction for unstructured meshes
                d = mesh.cell_centroids[neighbor_idx] - mesh.cell_centroids[i]
                if np.linalg.norm(d) > 1e-9:
                    e = d / np.linalg.norm(d)
                    k = face_normal / np.linalg.norm(face_normal)
                    if abs(np.dot(d, k)) > 1e-9:
                        # The correction term is a scalar, but was calculated as a vector.
                        # The correction vector is dotted with the cell-to-cell vector 'd'
                        # to get a scalar correction value.
                        correction_vector = (e - k * np.dot(e, k)) / np.dot(d, k)
                        for m in range(nvars):
                            non_orth_correction = (
                                U[neighbor_idx, m] - U[i, m]
                            ) * np.dot(correction_vector, d)
                            U_face[m] += over_relaxation * non_orth_correction
            else:
                # Boundary face: use the interior cell value
                U_face = U[i]

            grad_sum[:, 0] += U_face * face_normal[0] * face_area
            grad_sum[:, 1] += U_face * face_normal[1] * face_area

        if mesh.cell_volumes[i] > 1e-9:
            gradients[i] = grad_sum / mesh.cell_volumes[i]

    return gradients


def compute_limiters(mesh: Mesh, U, gradients, limiter_type="barth_jespersen"):
    """
    Computes the slope limiter for each cell to ensure monotonicity.

    This function prevents spurious oscillations (Gibbs phenomenon) near
    discontinuities by limiting the gradient of the solution.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        gradients (np.ndarray): The gradients of the state variables.
        limiter_type (str, optional): The type of limiter to use.
                                    Defaults to "barth_jespersen".

    Returns:
        np.ndarray: The limiter values (phi) for each cell and variable.
    """
    nvars = U.shape[1]
    limiter_func = LIMITERS.get(limiter_type, barth_jespersen_limiter)
    limiters = np.ones((mesh.nelem, nvars))

    for i in range(mesh.nelem):
        cell_neighbors = mesh.cell_neighbors[i]
        nfaces = cell_neighbors.shape[0]
        U_i = U[i]
        grad_i = gradients[i]

        # Determine the max and min values among the cell and its neighbors
        U_max = U_i.copy()
        U_min = U_i.copy()
        for neighbor_idx in cell_neighbors:
            if neighbor_idx != -1:
                U_neighbor = U[neighbor_idx]
                U_max = np.maximum(U_max, U_neighbor)
                U_min = np.minimum(U_min, U_neighbor)

        # Check against extrapolated values at face midpoints
        for j in range(nfaces):
            face_midpoint = mesh.face_midpoints[i, j]
            r_if = face_midpoint - mesh.cell_centroids[i]

            # Extrapolate value to the face midpoint
            U_face_extrap = U_i + np.array(
                [np.dot(grad_i[k], r_if[:2]) for k in range(nvars)]
            )

            # Calculate the limiter ratio (r)
            for k in range(nvars):
                diff = U_face_extrap[k] - U_i[k]
                if abs(diff) > 1e-9:
                    if diff > 0:
                        r = (U_max[k] - U_i[k]) / diff
                    else:
                        r = (U_min[k] - U_i[k]) / diff
                    limiters[i, k] = min(limiters[i, k], limiter_func(r))

    return limiters


def reconst_func(
    mesh: Mesh,
    U,
    equation,
    boundary_conditions,
    limiter_type,
    flux_type,
    over_relaxation=1.2,
) -> np.ndarray:
    """
    Performs MUSCL reconstruction to determine the states at the left and right
    sides of each internal face.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The array of conservative state vectors for all cells.
        equation: The equation object (e.g., EulerEquations).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the reconstructed states
        at the left (U_L) and right (U_R) sides of each internal face.
    """
    # --- Second-Order Reconstruction ---
    nvars = U.shape[1]

    gradients = compute_gradients_gaussian(mesh, U, over_relaxation)
    limiters = compute_limiters(mesh, U, gradients, limiter_type=limiter_type)

    res = U.copy()

    # --- Flux Integration Loop ---
    for i in range(mesh.nelem):
        for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
            face_normal = mesh.face_normals[i, j, 0:2]
            face_area = mesh.face_areas[i, j]

            if neighbor_idx != -1:
                # --- Interior Face ---
                face_midpoint = mesh.face_midpoints[i, j]
                d_i = face_midpoint - mesh.cell_centroids[i]
                d_j = mesh.cell_centroids[neighbor_idx] - face_midpoint

                delta_U_i = np.array(
                    [np.dot(gradients[i, k], d_i[:2]) for k in range(nvars)]
                )
                delta_U_j = np.array(
                    [np.dot(gradients[neighbor_idx, k], d_j[:2]) for k in range(nvars)]
                )

                U_L = U[i] + limiters[i] * delta_U_i
                U_R = U[neighbor_idx] + limiters[neighbor_idx] * delta_U_j

                if flux_type == "roe":
                    flux = equation.roe_flux(U_L, U_R, face_normal)
                elif flux_type == "hllc":
                    flux = equation.hllc_flux(U_L, U_R, face_normal)
            else:
                # --- Boundary Face ---
                face_tuple = tuple(sorted(mesh.elem_faces[i][j]))
                bc_name = mesh.boundary_faces.get(face_tuple, {}).get("name", "wall")
                bc_info = boundary_conditions.get(bc_name, {"type": "wall"})

                # Get ghost cell state based on boundary condition
                U_ghost = equation.apply_boundary_condition(U[i], face_normal, bc_info)

                # Flux is computed between the interior cell and the ghost cell
                if flux_type == "roe":
                    flux = equation.roe_flux(U[i], U_ghost, face_normal)
                elif flux_type == "hllc":
                    flux = equation.hllc_flux(U[i], U_ghost, face_normal)

            res[i] -= (face_area / mesh.cell_volumes[i]) * flux
            res[neighbor_idx] += (face_area / mesh.cell_volumes[i]) * flux

    return res
