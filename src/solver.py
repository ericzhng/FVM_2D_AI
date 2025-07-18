import numpy as np
from src.mesh import Mesh


# --- Limiter Functions ---
def barth_jespersen_limiter(r):
    return np.minimum(1, r)


def minmod_limiter(r):
    return np.maximum(0, np.minimum(1, r))


def superbee_limiter(r):
    return np.maximum(0, np.maximum(np.minimum(2 * r, 1), np.minimum(r, 2)))


LIMITERS = {
    "barth_jespersen": barth_jespersen_limiter,
    "minmod": minmod_limiter,
    "superbee": superbee_limiter,
}


def compute_gradients_gaussian(mesh: Mesh, U, over_relaxation=1.2):
    """
    Computes gradients using the Gaussian method with over-relaxed non-orthogonal correction.
    """
    gradients = np.zeros((mesh.nelem, U.shape[1], 2))  # For 2D gradients (x, y)

    for i in range(mesh.nelem):
        grad_sum = np.zeros((U.shape[1], 2))
        for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
            face_normal = mesh.face_normals[i, j, :2]
            face_area = mesh.face_areas[i, j]

            if neighbor_idx != -1:
                U_face = 0.5 * (U[i] + U[neighbor_idx])
                # Non-orthogonal correction for unstructured meshes
                d = mesh.cell_centroids[neighbor_idx] - mesh.cell_centroids[i]
                if np.linalg.norm(d) > 1e-9:
                    e = d / np.linalg.norm(d)
                    k = face_normal / np.linalg.norm(face_normal)
                    k = np.append(k, 0)
                    # Check for zero dot product before division
                    if np.dot(d, k) != 0:
                        non_orth_correction = (
                            (U[neighbor_idx] - U[i])
                            * (e - k * np.dot(e, k))
                            / np.dot(d, k)
                        )
                        U_face += over_relaxation * non_orth_correction
            else:
                # A simple approach: use the interior value at the boundary face
                U_face = U[i]

            grad_sum[:, 0] += U_face * face_normal[0] * face_area
            grad_sum[:, 1] += U_face * face_normal[1] * face_area

        if mesh.cell_volumes[i] > 1e-9:
            gradients[i] = grad_sum / mesh.cell_volumes[i]

    return gradients


def compute_limiters(mesh: Mesh, U, gradients, limiter_type="barth_jespersen"):
    """
    Computes the slope limiter for each cell to prevent oscillations.
    """
    limiter_func = LIMITERS.get(limiter_type, barth_jespersen_limiter)
    limiters = np.ones((mesh.nelem, U.shape[1]))

    for i in range(mesh.nelem):
        U_i = U[i]
        grad_i = gradients[i]

        U_max = U_i.copy()
        U_min = U_i.copy()
        for neighbor_idx in mesh.cell_neighbors[i]:
            if neighbor_idx != -1:
                U_neighbor = U[neighbor_idx]
                U_max = np.maximum(U_max, U_neighbor)
                U_min = np.minimum(U_min, U_neighbor)

        for j, face_nodes in enumerate(mesh.elem_faces[i]):
            node_coords = [
                mesh.node_coords[np.where(mesh.node_tags == tag)[0][0]]
                for tag in face_nodes
            ]
            face_midpoint = np.mean(node_coords, axis=0)
            r_if = face_midpoint - mesh.cell_centroids[i]

            U_face_extrap = U_i + np.array(
                [np.dot(grad_i[k], r_if[:2]) for k in range(U.shape[1])]
            )

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
    Reconstructs the values at the face midpoint using MUSCL with a slope limiter.
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
    t_end=2.0,
    over_relaxation=1.2,
    limiter="barth_jespersen",
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
):
    """
    Solves the given equations using a Finite Volume Method.
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

        gradients = compute_gradients_gaussian(mesh, U, over_relaxation)
        limiters = compute_limiters(mesh, U, gradients, limiter_type=limiter)
        U_new = U.copy()

        for i in range(mesh.nelem):
            for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
                face_normal = mesh.face_normals[i, j, :2]
                face_area = mesh.face_areas[i, j]

                if neighbor_idx != -1:
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
                    face_tuple = mesh.elem_faces[i][j]
                    bc_name = mesh.boundary_faces.get(face_tuple, {}).get(
                        "name", "wall"
                    )
                    bc_info = boundary_conditions.get(bc_name, {"type": "wall"})
                    U_ghost = equation.apply_boundary_condition(
                        U[i], face_normal, bc_info
                    )
                    flux = equation.hllc_flux(U[i], U_ghost, face_normal)

                U_new[i] -= (dt / mesh.cell_volumes[i]) * flux * face_area

        U = U_new
        t += dt
        history.append(U.copy())
        dt_history.append(dt)
        print(f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s")

    return history, dt_history
