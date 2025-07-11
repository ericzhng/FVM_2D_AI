import numpy as np
from src.mesh import get_face_normal_and_length, compute_cell_centroids


# Minmod limiter for MUSCL
def minmod(a, b):
    if a * b > 0:
        return min(abs(a), abs(b)) * np.sign(a)
    return 0.0


# HLLC flux calculation for 2D Shallow Water Equations
def hllc_flux(U_L, U_R, normal, g):
    h_L, hu_L, hv_L = U_L
    h_R, hu_R, hv_R = U_R

    u_L = hu_L / h_L if h_L > 1e-6 else 0
    v_L = hv_L / h_L if h_L > 1e-6 else 0
    u_R = hu_R / h_R if h_R > 1e-6 else 0
    v_R = hv_R / h_R if h_R > 1e-6 else 0

    # Rotate velocities to be normal to the face
    un_L = u_L * normal[0] + v_L * normal[1]
    un_R = u_R * normal[0] + v_R * normal[1]

    # Wave speed estimates (Roe averages)
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    S_L = min(un_L - c_L, un_R - c_R)
    S_R = max(un_L + c_L, un_R + c_R)

    # Flux vectors in the normal direction
    F_L = np.array(
        [
            h_L * un_L,
            hu_L * un_L + 0.5 * g * h_L**2 * normal[0],
            hv_L * un_L + 0.5 * g * h_L**2 * normal[1],
        ]
    )
    F_R = np.array(
        [
            h_R * un_R,
            hu_R * un_R + 0.5 * g * h_R**2 * normal[0],
            hv_R * un_R + 0.5 * g * h_R**2 * normal[1],
        ]
    )

    if S_L >= 0:
        return F_L
    elif S_R <= 0:
        return F_R
    else:
        S_star = (
            S_R * h_R * (un_R - S_R)
            - S_L * h_L * (un_L - S_L)
            + 0.5 * g * (h_L**2 - h_R**2)
        ) / (h_R * (un_R - S_R) - h_L * (un_L - S_L))

        U_star_L = (S_L * U_L - F_L) / (S_L - S_star)
        U_star_R = (S_R * U_R - F_R) / (S_R - S_star)

        if S_star >= 0:
            return F_L + S_L * (U_star_L - U_L)
        else:
            return F_R + S_R * (U_star_R - U_R)


def compute_gradients(
    U, all_neighbors, cell_centroids, elem_conn, node_coord, cell_areas
):
    """Computes the gradients for all cells using the Green-Gauss theorem."""
    nelem = len(elem_conn)
    gradients = np.zeros((nelem, 3, 2))  # (nelem, nvars, 2 for x and y)
    for i in range(nelem):
        grad_sum = np.zeros((3, 2))
        for j in all_neighbors[i]:
            normal, length = get_face_normal_and_length(i, j, elem_conn, node_coord)
            if length is not None and length > 0:
                # Average value at the face
                U_face = 0.5 * (U[i] + U[j])
                grad_sum[:, 0] += U_face * normal[0] * length
                grad_sum[:, 1] += U_face * normal[1] * length
        if cell_areas[i] > 1e-9:
            gradients[i] = grad_sum / cell_areas[i]
    return gradients


# MUSCL reconstruction
def muscl_reconstruction(U, gradients, cell_centroids, i, j, elem_conn, node_coord):
    """
    Reconstructs the values at the face between cell i and j.
    """
    # Find face midpoint
    elem1_nodes = set(elem_conn[i])
    elem2_nodes = set(elem_conn[j])
    common_nodes_tags = list(elem1_nodes.intersection(elem2_nodes))
    if len(common_nodes_tags) != 2:
        return U[i], U[j]  # Should not happen

    p1 = node_coord[common_nodes_tags[0] - 1][:2]
    p2 = node_coord[common_nodes_tags[1] - 1][:2]
    face_midpoint = (p1 + p2) / 2.0

    # Vectors from centroids to face midpoint
    r_if = face_midpoint - cell_centroids[i]
    r_jf = face_midpoint - cell_centroids[j]

    # Gradients for cell i and j
    grad_i = gradients[i]
    grad_j = gradients[j]

    # Extrapolate to the face midpoint
    delta_U_i = np.array([np.dot(grad_i[k], r_if) for k in range(3)])
    delta_U_j = np.array([np.dot(grad_j[k], r_jf) for k in range(3)])

    # Simple limiter to ensure positivity of h
    alpha_i = 1.0
    if U[i][0] + delta_U_i[0] < 1e-6:
        alpha_i = -U[i][0] / (delta_U_i[0] + 1e-9)

    alpha_j = 1.0
    if U[j][0] + delta_U_j[0] < 1e-6:
        alpha_j = -U[j][0] / (delta_U_j[0] + 1e-9)

    limiter = min(1.0, alpha_i, alpha_j)

    U_L = U[i] + limiter * delta_U_i
    U_R = U[j] + limiter * delta_U_j

    return U_L, U_R


# Main solver loop
def solve(
    U, elem_conn, node_coord, all_neighbors, cell_areas, cell_centroids, g, dt, t_end
):
    t = 0.0
    history = []
    nelem = len(elem_conn)
    while t < t_end:
        U_new = U.copy()
        gradients = compute_gradients(
            U, all_neighbors, cell_centroids, elem_conn, node_coord, cell_areas
        )
        for i in range(nelem):
            neighbors = all_neighbors[i]
            for j in neighbors:
                normal, length = get_face_normal_and_length(i, j, elem_conn, node_coord)
                if length is not None and length > 0:
                    U_L, U_R = muscl_reconstruction(
                        U,
                        gradients,
                        cell_centroids,
                        i,
                        j,
                        elem_conn,
                        node_coord,
                    )

                    flux = hllc_flux(U_L, U_R, normal, g)

                    U_new[i] -= (dt / cell_areas[i]) * flux * length
            # Apply boundary conditions (example: reflective boundaries)
            # Note: This is a placeholder and needs proper implementation
            boundary_info = get_boundary_faces(i, elem_conn)

            inlet_values = (1.0, 1.0, 0.2)

            for edge, boundary_tag in boundary_info:
                # Need to find a neighbor index for boundary faces.  Using a dummy value for now.
                #  The correct approach depends on how your mesh is structured.
                #  In many cases, you might have a "ghost cell" or a special index to represent boundaries.
                neighbor_idx = (
                    -1
                )  #  Placeholder: Replace with your boundary representation
                normal, length = get_face_normal_and_length(
                    i, neighbor_idx, elem_conn, node_coord, edge
                )
                # normal, length = get_face_normal_and_length(
                #     i, boundary_idx, elem_conn, node_coord
                # )

                bc_type = boundary_tag  # Use the tag directly
                if bc_type == "inlet":
                    U_ghost = apply_boundary_condition(
                        U[i], normal, bc_type, inlet_values=inlet_values
                    )
                else:
                    U_ghost = apply_boundary_condition(U[i], normal, bc_type)
                if length is not None and length > 0:
                    U_ghost = apply_boundary_condition(U[i], normal, "reflective")
                    flux = hllc_flux(U[i], U_ghost, normal, g)

        U[:] = U_new
        t += dt
        if int(t / dt) % 10 == 0:  # Print progress
            print(f"t = {t:.2f}")
            history.append(U.copy())
    return history


def get_boundary_faces(elem_idx, elem_conn):
    """
    Identifies boundary faces for a given element.

    Args:
        elem_idx (int): Index of the element.
        elem_conn (np.ndarray): Element connectivity array.

    Returns:
        list: List of node indices forming boundary faces.  Returns empty list if no boundary face found.
    """
    elem_nodes = elem_conn[elem_idx]
    boundary_info = []
    for i in range(4):
        node1 = elem_nodes[i]
        node2 = elem_nodes[(i + 1) % 4]
        edge = tuple(sorted((node1, node2)))
        if edge in boundary_edges:
            # Assuming boundary_edges dictionary has edge: tag mapping
            boundary_info.append((edge, boundary_edges[edge]))
    return boundary_info


def apply_boundary_condition(U_inside, normal, bc_type, **kwargs):
    """
    Applies a boundary condition and returns the ghost cell value.
    """
    if bc_type == "inlet":
        if "inlet_values" in kwargs:
            h_in, u_in, v_in = kwargs["inlet_values"]
            U_ghost = np.array([h_in, h_in * u_in, h_in * v_in])
        else:
            U_ghost = U_inside  # Default to interior value if no inlet values provided
    elif bc_type == "outlet":  # Extrapolate
        U_ghost = U_inside
    elif bc_type == "wall":  # Reflective
        U_ghost = U_inside  # Assuming no-penetration for simplicity
    else:  # Default or unknown BC
        U_ghost = U_inside

    return U_ghost


def initialize_boundary_edges(elem_conn, boundary_nodes):
    # Create a dictionary mapping boundary edges (tuples of sorted node tags) to boundary tags.
    # Placeholder:  This needs to be filled using mesh data.
    return {(1, 2): 1, (3, 4): 2}  # Example: Replace with actual boundary data.
