import numpy as np
from scipy import sparse
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

for i, elem_type in enumerate(element_types):
    # Get information about the element type (name, dimension, number of nodes)
    elem_name, elem_dim, order, num_nodes_per_elem, _, _ = (
        gmsh.model.mesh.getElementProperties(elem_type)
    )

    if num_nodes_per_elem == 4 or num_nodes_per_elem == 2:
        print(
            f"\nElement Type: {elem_name} (Type ID: {elem_type}, Dimension: {elem_dim}, Nodes per element: {num_nodes_per_elem})"
        )
        print(f"Number of elements of this type: {len(element_tags[i])}")
        # # Iterate through elements of the current type
        # current_node_idx = 0
        # for j, tag in enumerate(element_tags[i]):
        #     # Extract node tags for the current element
        #     nodes = element_node_tags[i][
        #         current_node_idx : current_node_idx + num_nodes_per_elem
        #     ]
        #     print(f"{tag:<10} | {' '.join(map(str, nodes))}")
        #     current_node_idx += num_nodes_per_elem

# Finalize Gmsh
gmsh.finalize()


# Constants
g = 9.81  # Gravity
dt = 0.01  # Time step
t_end = 1.0  # End time

# Conserved variables: h, hu, hv
elem_tags = element_tags[1]
elem_conn = element_node_tags[1].reshape(-1, 4)
node_tags
node_coord = node_coords.reshape(-1, 3)
bc_tags = element_tags[0]
bc_conn = element_node_tags[0].reshape(-1, 2)

nelem = len(elem_tags)
U = np.zeros((nelem, 3))  # [h, hu, hv]

# Initialize Riemann problem (discontinuity at x=0)
for i, elem in enumerate(elem_conn):
    centroid = np.mean(node_coord[elem - 1], axis=0)
    U[i] = [1.0, 0.0, 0.0] if centroid[0] < 50 else [0.5, 0.0, 0.0]


# Minmod limiter for MUSCL
def minmod(a, b):
    if a * b > 0:
        return min(abs(a), abs(b)) * np.sign(a)
    return 0.0


# Roe flux calculation
def roe_flux(U_L, U_R):
    h_L, hu_L, hv_L = U_L
    h_R, hu_R, hv_R = U_R
    u_L, v_L = hu_L / h_L if h_L > 0 else 0, hv_L / h_L if h_L > 0 else 0
    u_R, v_R = hu_R / h_R if h_R > 0 else 0, hv_R / h_R if h_R > 0 else 0

    # Roe averages
    sqrt_h_L, sqrt_h_R = np.sqrt(h_L), np.sqrt(h_R)
    h_tilde = sqrt_h_L * sqrt_h_R
    u_tilde = (sqrt_h_L * u_L + sqrt_h_R * u_R) / (sqrt_h_L + sqrt_h_R)
    v_tilde = (sqrt_h_L * v_L + sqrt_h_R * v_R) / (sqrt_h_L + sqrt_h_R)

    # Eigenvalues
    c = np.sqrt(g * h_tilde)
    lambda_1, lambda_2, lambda_3 = u_tilde - c, u_tilde, u_tilde + c

    # Eigenvectors
    K1 = np.array([1, u_tilde - c, v_tilde])
    K2 = np.array([0, 0, 1])
    K3 = np.array([1, u_tilde + c, v_tilde])

    # Wave strengths
    delta_U = U_R - U_L
    alpha_2 = delta_U[2] - v_tilde * delta_U[0]
    alpha_1 = (delta_U[0] * (u_tilde + c) - delta_U[1] + alpha_2 * v_tilde) / (2 * c)
    alpha_3 = delta_U[0] - alpha_1

    # Fluxes
    F_L = np.array([hu_L, hu_L * u_L + 0.5 * g * h_L**2, hu_L * v_L])
    F_R = np.array([hu_R, hu_R * u_R + 0.5 * g * h_R**2, hu_R * v_R])

    # Roe flux
    return 0.5 * (F_L + F_R) - 0.5 * (
        abs(lambda_1) * alpha_1 * K1
        + abs(lambda_2) * alpha_2 * K2
        + abs(lambda_3) * alpha_3 * K3
    )


# MUSCL reconstruction
def muscl_reconstruction(U, cell_neighbors, i):
    U_L, U_R = U[i], U[i]
    for j in cell_neighbors[i]:
        r = (U[j] - U[i]) / (U[i] - U[cell_neighbors[i][0]] + 1e-10)
        phi = minmod(1, r)
        U_L += 0.5 * phi * (U[i] - U[j])
        U_R -= 0.5 * phi * (U[cell_neighbors[j][0]] - U[i])
    return U_L, U_R


# Main solver loop
def solve():
    t = 0.0
    while t < t_end:
        U_new = U.copy()
        for i, elem in enumerate(elem_conn):
            # Assume cell_neighbors stores indices of neighboring cells
            cell_neighbors = get_neighbors(i)  # Implement based on mesh connectivity
            for j in cell_neighbors:
                U_L, U_R = muscl_reconstruction(U, cell_neighbors, i)
                flux = roe_flux(U_L, U_R)
                normal = get_face_normal(i, j)  # Implement based on mesh
                U_new[i] -= dt / cell_area[i] * flux.dot(normal)

        U[:] = U_new
        t += dt


# Placeholder for neighbor and normal calculations
def get_neighbors(i):
    # Implement mesh-based neighbor search
    return []


def get_face_normal(i, j):
    # Implement based on mesh geometry
    return np.array([1, 0])


# Cell areas (placeholder)
cell_area = np.ones(len(elements[1]))  # Compute from mesh

# Run solver
solve()
