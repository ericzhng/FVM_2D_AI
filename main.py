from src.mesh import read_mesh, get_neighbors, compute_cell_areas, compute_cell_centroids
from src.solver import solve
from src.case_setup import dam_break_initial_conditions
from src.visualization import visualize_mesh, create_animation, visualize_results

# --- Parameters ---
g = 9.81
dt = 0.01
t_end = 1.0
mesh_file = "data/rectangle_mesh.msh"

# --- Mesh Reading and Geometric Computations ---
node_tags, node_coord, elem_tags_quad, elem_conn = read_mesh(mesh_file)
cell_areas = compute_cell_areas(elem_conn, node_coord)
cell_centroids = compute_cell_centroids(elem_conn, node_coord)
all_neighbors = get_neighbors(elem_conn)

# --- Initial Conditions ---
U = dam_break_initial_conditions(elem_conn, node_coord)

# --- Visualization (optional) ---
# visualize_mesh(elem_conn, node_coord, all_neighbors, cell_areas)

# --- Solve ---
history = solve(U, elem_conn, node_coord, all_neighbors, cell_areas, cell_centroids, g, dt, t_end)

# --- Post-processing ---
create_animation(history, elem_conn, node_coord, dt)
# visualize_results(history[-1], elem_conn, node_coord, t_end)
