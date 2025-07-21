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
    gradients = np.zeros((mesh.nelem, U.shape[1], 2))  # For 2D gradients (x, y)

    for i in range(mesh.nelem):
        grad_sum = np.zeros((U.shape[1], 2))
        for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
            face_normal = mesh.face_normals[i, j, :2]
            face_area = mesh.face_areas[i, j]

            if neighbor_idx != -1:
                # For non-uniform meshes, a distance-weighted interpolation is more accurate.
                face_nodes_tags = mesh.elem_faces[i][j]
                node_coords = [
                    mesh.node_coords[np.where(mesh.node_tags == tag)[0][0]]
                    for tag in face_nodes_tags
                ]
                face_midpoint = np.mean(node_coords, axis=0)

                d_i = np.linalg.norm(face_midpoint - mesh.cell_centroids[i])
                d_j = np.linalg.norm(face_midpoint - mesh.cell_centroids[neighbor_idx])

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
                    k = np.append(k, 0)
                    if abs(np.dot(d, k)) > 1e-9:
                        non_orth_correction = (
                            (U[neighbor_idx] - U[i])
                            * (e - k * np.dot(e, k))
                            / np.dot(d, k)
                        )
                        U_face += over_relaxation * non_orth_correction
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
    limiter_func = LIMITERS.get(limiter_type, barth_jespersen_limiter)
    limiters = np.ones((mesh.nelem, U.shape[1]))

    for i in range(mesh.nelem):
        U_i = U[i]
        grad_i = gradients[i]

        # Determine the max and min values among the cell and its neighbors
        U_max = U_i.copy()
        U_min = U_i.copy()
        for neighbor_idx in mesh.cell_neighbors[i]:
            if neighbor_idx != -1:
                U_neighbor = U[neighbor_idx]
                U_max = np.maximum(U_max, U_neighbor)
                U_min = np.minimum(U_min, U_neighbor)

        # Check against extrapolated values at face midpoints
        for j, face_nodes in enumerate(mesh.elem_faces[i]):
            node_coords = [
                mesh.node_coords[np.where(mesh.node_tags == tag)[0][0]]
                for tag in face_nodes
            ]
            face_midpoint = np.mean(node_coords, axis=0)
            r_if = face_midpoint - mesh.cell_centroids[i]

            # Extrapolate value to the face midpoint
            U_face_extrap = U_i + np.array(
                [np.dot(grad_i[k], r_if[:2]) for k in range(U.shape[1])]
            )

            # Calculate the limiter ratio (r)
            for k in range(U.shape[1]):
                diff = U_face_extrap[k] - U_i[k]
                if abs(diff) > 1e-9:
                    if diff > 0:
                        r = (U_max[k] - U_i[k]) / diff
                    else:
                        r = (U_min[k] - U_i[k]) / diff
                    limiters[i, k] = min(limiters[i, k], limiter_func(r))

    return limiters


def muscl_reconstruction(
    U_i,
    U_j,
    grad_i,
    grad_j,
    limiter_i,
    limiter_j,
    centroid_i,
    centroid_j,
    face_midpoint,
):
    """
    Reconstructs the state variables at the face midpoint using the MUSCL scheme.

    This provides second-order spatial accuracy by extrapolating the cell-centered
    values to the faces using the computed gradients and limiters.

    Args:
        U_i, U_j (np.ndarray): State vectors of the left and right cells.
        grad_i, grad_j (np.ndarray): Gradients in the left and right cells.
        limiter_i, limiter_j (np.ndarray): Limiter values for the left and right cells.
        centroid_i, centroid_j (np.ndarray): Centroids of the left and right cells.
        face_midpoint (np.ndarray): Midpoint of the face.

    Returns:
        tuple: A tuple containing the reconstructed state vectors at the left and
               right sides of the face (U_L, U_R).
    """
    r_if = face_midpoint - centroid_i
    r_jf = face_midpoint - centroid_j

    delta_U_i = np.array([np.dot(grad_i[k], r_if[:2]) for k in range(U_i.shape[0])])
    delta_U_j = np.array([np.dot(grad_j[k], r_jf[:2]) for k in range(U_j.shape[0])])

    U_L = U_i + limiter_i * delta_U_i
    U_R = U_j + limiter_j * delta_U_j

    return U_L, U_R


def solve(
    mesh: Mesh,
    U,
    boundary_conditions,
    equation,
    t_end,
    over_relaxation=1.2,
    limiter="barth_jespersen",
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
):
    """
    Main solver loop for the Finite Volume Method.

    This function orchestrates the time-stepping process, including gradient
    computation, slope limiting, MUSCL reconstruction, and flux calculation.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The initial conservative state vector.
        boundary_conditions (dict): A dictionary defining the boundary conditions.
        equation (BaseEquation): The equation system to be solved.
        t_end (float, optional): The end time of the simulation. Defaults to 2.0.
        over_relaxation (float, optional): Over-relaxation factor. Defaults to 1.2.
        limiter (str, optional): The type of slope limiter. Defaults to "barth_jespersen".
        use_adaptive_dt (bool, optional): Whether to use adaptive time stepping.
                                        Defaults to True.
        cfl (float, optional): The CFL number for adaptive time stepping. Defaults to 0.5.
        dt_initial (float, optional): The initial time step. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the history of the state vector and the history
               of the time steps.
    """
    t = 0.0
    history = [U.copy()]
    dt_history = []
    dt = dt_initial

    while t < t_end:
        if use_adaptive_dt:
            dt = equation.calculate_adaptive_dt(mesh, U, cfl)

        # Ensure the last time step does not overshoot t_end
        if t + dt > t_end:
            dt = t_end - t

        # --- Second-Order Reconstruction ---
        gradients = compute_gradients_gaussian(mesh, U, over_relaxation)
        limiters = compute_limiters(mesh, U, gradients, limiter_type=limiter)

        U_new = U.copy()

        # --- Flux Integration Loop ---
        for i in range(mesh.nelem):
            for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
                face_normal = mesh.face_normals[i, j, :2]
                face_area = mesh.face_areas[i, j]

                if neighbor_idx != -1:
                    # --- Interior Face ---
                    face_nodes_tags = mesh.elem_faces[i][j]
                    node_coords = [
                        mesh.node_coords[np.where(mesh.node_tags == tag)[0][0]]
                        for tag in face_nodes_tags
                    ]
                    face_midpoint = np.mean(node_coords, axis=0)

                    U_L, U_R = muscl_reconstruction(
                        U[i],
                        U[neighbor_idx],
                        gradients[i],
                        gradients[neighbor_idx],
                        limiters[i],
                        limiters[neighbor_idx],
                        mesh.cell_centroids[i],
                        mesh.cell_centroids[neighbor_idx],
                        face_midpoint,
                    )
                    flux = equation.hllc_flux(U_L, U_R, face_normal)
                else:
                    # --- Boundary Face ---
                    face_tuple = tuple(sorted(mesh.elem_faces[i][j]))
                    bc_name = mesh.boundary_faces.get(face_tuple, {}).get(
                        "name", "wall"
                    )
                    bc_info = boundary_conditions.get(bc_name, {"type": "wall"})

                    # Get ghost cell state based on boundary condition
                    U_ghost = equation.apply_boundary_condition(
                        U[i], face_normal, bc_info
                    )

                    # Flux is computed between the interior cell and the ghost cell
                    flux = equation.hllc_flux(U[i], U_ghost, face_normal)

                # Update the solution
                U_new[i] -= (dt / mesh.cell_volumes[i]) * flux * face_area

        U = U_new
        t += dt
        history.append(U.copy())
        dt_history.append(dt)
        print(f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s")

    return history, dt_history
