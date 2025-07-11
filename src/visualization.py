import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import numpy as np
from src.mesh import compute_cell_centroids, get_face_normal_and_length


def visualize_mesh(elem_conn, node_coord, all_neighbors, cell_areas):
    """Visualizes the mesh, labels, normals, and face lengths."""
    fig, ax = plt.subplots(figsize=(12, 12))
    nelem = len(elem_conn)

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


def create_animation(history, elem_conn, node_coord, dt, interval=100):
    """Creates an animation of the simulation history."""
    fig, ax = plt.subplots(figsize=(12, 10))

    patches = []
    for elem_nodes in elem_conn:
        nodes = node_coord[np.array(elem_nodes) - 1]
        polygon = Polygon(nodes[:, :2], closed=True)
        patches.append(polygon)
    p = PatchCollection(patches, alpha=0.9)
    ax.add_collection(p)
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
    plt.show()
    # anim.save("shallow_water.gif", writer="imagemagick")
