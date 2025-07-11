import numpy as np
import gmsh

# Initialize Gmsh for mesh reading
gmsh.initialize()
gmsh.open("data/rectangle_mesh.msh")  # Replace with your mesh file

print("\n--- Mesh Information ---")
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
print(f"Total number of nodes: {len(node_tags)}")

# Get all elements in the model
element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
print(f"Total number of element types found: {len(element_types)}")
gmsh.finalize()


# Extract quadrilateral element data (assuming they are the second type found)
elem_tags_quad = element_tags[1]
elem_conn = np.array(element_node_tags[1]).reshape(-1, 4)
node_coord = np.array(node_coords).reshape(-1, 3)
nelem = len(elem_tags_quad)


# --- Geometric Computations ---


def get_neighbors(elem_conn):
    """Computes neighbors for all elements."""
    nelem = elem_conn.shape[0]

    face_to_elem = {}
    for i, elem_nodes in enumerate(elem_conn):
        for j in range(4):
            node1 = elem_nodes[j]
            node2 = elem_nodes[(j + 1) % 4]
            face = tuple(sorted((node1, node2)))
            if face not in face_to_elem:
                face_to_elem[face] = []
            face_to_elem[face].append(i)

    all_neighbors = [[] for _ in range(nelem)]
    for i in range(nelem):
        elem_nodes = elem_conn[i]
        neighbors = []
        for j in range(4):
            node1 = elem_nodes[j]
            node2 = elem_nodes[(j + 1) % 4]
            face = tuple(sorted((node1, node2)))
            for neighbor_idx in face_to_elem[face]:
                if neighbor_idx != i:
                    neighbors.append(neighbor_idx)
        all_neighbors[i] = list(set(neighbors))
    return all_neighbors


def get_face_normal_and_length(elem_idx, neighbor_idx, elem_conn, node_coord):
    """Computes the normal and length of the face between two elements."""
    elem1_nodes = set(elem_conn[elem_idx])
    elem2_nodes = set(elem_conn[neighbor_idx])

    common_nodes_tags = list(elem1_nodes.intersection(elem2_nodes))
    if len(common_nodes_tags) != 2:
        return np.array([0, 0]), 0.0

    p1 = node_coord[common_nodes_tags[0] - 1]
    p2 = node_coord[common_nodes_tags[1] - 1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    length = np.sqrt(dx**2 + dy**2)

    if length < 1e-9:
        return np.array([0, 0]), 0

    # Normal vector, normalized
    normal = np.array([dy / length, -dx / length])

    # Ensure the normal points from elem_idx to neighbor_idx
    centroid1 = np.mean(node_coord[elem_conn[elem_idx] - 1], axis=0)
    face_midpoint = (p1 + p2) / 2.0

    vec_to_face = face_midpoint - centroid1

    if np.dot(normal, vec_to_face[:2]) < 0:
        normal = -normal

    return normal, length


def compute_cell_areas(elem_conn, node_coord):
    """Computes the areas of all cells."""
    areas = np.zeros(len(elem_conn))
    for i, elem_nodes in enumerate(elem_conn):
        nodes = node_coord[np.array(elem_nodes) - 1]
        x = nodes[:, 0]
        y = nodes[:, 1]
        areas[i] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return areas


def compute_cell_centroids(elem_conn, node_coord):
    """Computes the centroids of all cells."""
    centroids = np.zeros((len(elem_conn), 2))
    for i, elem_nodes in enumerate(elem_conn):
        nodes = node_coord[np.array(elem_nodes) - 1]
        centroids[i] = np.mean(nodes[:, :2], axis=0)
    return centroids


# Pre-compute geometric properties
cell_areas = compute_cell_areas(elem_conn, node_coord)
cell_centroids = compute_cell_centroids(elem_conn, node_coord)
all_neighbors = get_neighbors(elem_conn)


# Constants
g = 9.81
dt = 0.01
t_end = 1.0

# Conserved variables: h, hu, hv
U = np.zeros((nelem, 3))

# Initialize Riemann problem
for i, elem in enumerate(elem_conn):
    centroid = np.mean(node_coord[elem - 1], axis=0)
    U[i] = [1.0, 0.0, 0.0] if centroid[0] < 50 else [0.5, 0.0, 0.0]


# Minmod limiter for MUSCL
def minmod(a, b):
    if a * b > 0:
        return min(abs(a), abs(b)) * np.sign(a)
    return 0.0


# HLLC flux calculation for 2D Shallow Water Equations
def hllc_flux(U_L, U_R, normal):
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
def solve():
    t = 0.0
    history = []
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
                        U, gradients, cell_centroids, i, j, elem_conn, node_coord
                    )

                    flux = hllc_flux(U_L, U_R, normal)

                    U_new[i] -= (dt / cell_areas[i]) * flux * length

        U[:] = U_new
        t += dt
        if int(t / dt) % 10 == 0:  # Print progress
            print(f"t = {t:.4f}")
            history.append(U.copy())
    return history


import matplotlib

matplotlib.use("Qt5Agg")  # Or 'TkAgg', 'Qt4Agg', etc.
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation


def visualize_mesh(elem_conn, node_coord, all_neighbors, cell_areas):
    """Visualizes the mesh, labels, normals, and face lengths."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot elements
    for i, elem_nodes in enumerate(elem_conn):
        nodes = node_coord[np.array(elem_nodes) - 1]
        polygon = Polygon(nodes[:, :2], edgecolor="b", facecolor="none", lw=0.5)
        ax.add_patch(polygon)
        # Plot element labels (centroids)
        centroid = np.mean(nodes, axis=0)
        ax.text(centroid[0], centroid[1], str(i), color="blue", fontsize=8, ha="center")
        ax.text(
            centroid[0],
            centroid[1] - 2,
            f"Area: {cell_areas[i]:.2f}",
            color="black",
            fontsize=6,
            ha="center",
        )

    # Plot node labels
    for i, coord in enumerate(node_coord):
        ax.text(coord[0], coord[1], str(i + 1), color="red", fontsize=6, ha="center")

    # Plot face normals for each element
    for i in range(nelem):
        for j in all_neighbors[i]:
            normal, length = get_face_normal_and_length(i, j, elem_conn, node_coord)
            if length is not None and length > 0:
                # Find face midpoint
                elem1_nodes = set(elem_conn[i])
                elem2_nodes = set(elem_conn[j])
                common_nodes_tags = list(elem1_nodes.intersection(elem2_nodes))
                p1 = node_coord[common_nodes_tags[0] - 1]
                p2 = node_coord[common_nodes_tags[1] - 1]
                midpoint = (p1 + p2) / 2.0

                # Scale for visibility
                normal_scaled = normal * 2

                # Plot normal vector
                ax.quiver(
                    midpoint[0],
                    midpoint[1],
                    normal_scaled[0],
                    normal_scaled[1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="green",
                    width=0.005,  # Make arrows thinner
                )

    ax.set_aspect("equal", "box")
    ax.set_title("Mesh Visualization with Normals")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.show()


def visualize_results(U, elem_conn, node_coord, t):
    """Visualizes the final simulation results."""
    fig, ax = plt.subplots(figsize=(12, 10))

    h = U[:, 0]
    hu = U[:, 1]
    hv = U[:, 2]
    u = np.divide(hu, h, out=np.zeros_like(hu), where=h > 1e-6)
    v = np.divide(hv, h, out=np.zeros_like(hv), where=h > 1e-6)

    # Create a collection of polygons for the mesh
    patches = []
    for elem_nodes in elem_conn:
        nodes = node_coord[np.array(elem_nodes) - 1]
        polygon = Polygon(nodes[:, :2], closed=True)
        patches.append(polygon)

    p = PatchCollection(patches, alpha=0.9)
    p.set_array(h)
    ax.add_collection(p)
    fig.colorbar(p, ax=ax, label="Water Height (h)")

    # Overlay velocity vectors
    centroids = compute_cell_centroids(elem_conn, node_coord)
    ax.quiver(centroids[:, 0], centroids[:, 1], u, v, color="white", scale=50)

    ax.set_aspect("equal", "box")
    ax.set_title(f"Final Simulation Results at t = {t:.4f}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.show(block=True)


def create_animation(history, elem_conn, node_coord, interval=100):
    """Creates an animation of the simulation history."""
    fig, ax = plt.subplots(figsize=(12, 10))

    patches = []
    for elem_nodes in elem_conn:
        nodes = node_coord[np.array(elem_nodes) - 1]
        polygon = Polygon(nodes[:, :2], closed=True)
        patches.append(polygon)
    p = PatchCollection(patches, alpha=0.9)
    # ax.add_collection(p)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Solution at t = {0:.6f}")
    ax.grid(True)

    def update_frame(frame):
        U = history[frame]
        h = U[:, 0]
        p.set_array(h)
        ax.set_title(f"Simulation at t = {frame * dt * 10:.4f}")
        return [p]

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(history),
        interval=interval,
        blit=False,
        repeat=True,
        repeat_delay=3000,
    )
    anim.save("shallow_water.gif", writer="imagemagick")


# # Visualize the mesh before solving
# visualize_mesh(elem_conn, node_coord, all_neighbors, cell_areas)

# Run solver
history = solve()

# Create animation
create_animation(history, elem_conn, node_coord)

# # Visualize final results
# visualize_results(history[-1], elem_conn, node_coord, t_end)
