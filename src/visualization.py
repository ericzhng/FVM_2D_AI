import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np
from src.mesh import Mesh
from matplotlib.patches import Polygon


def plot_mesh(mesh: Mesh):
    """Visualizes the mesh, labels, normals, and face lengths."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot elements
    for i, elem_nodes_tags in enumerate(mesh.elem_conn):
        node_indices = [
            np.where(mesh.node_tags == tag)[0][0] for tag in elem_nodes_tags
        ]
        nodes = mesh.node_coords[np.array(node_indices)]
        polygon = Polygon(nodes[:, :2], edgecolor="b", facecolor="none", lw=0.5)
        ax.add_patch(polygon)
        # Plot element labels (centroids)
        ax.text(
            mesh.cell_centroids[i, 0],
            mesh.cell_centroids[i, 1],
            str(i),
            color="blue",
            fontsize=8,
            ha="center",
        )
        ax.text(
            mesh.cell_centroids[i, 0],
            mesh.cell_centroids[i, 1] - 1.2,
            f"A: {mesh.cell_volumes[i]:.2f}",
            color="black",
            fontsize=6,
            ha="center",
        )

    # Plot node labels
    for i, coord in enumerate(mesh.node_coords):
        ax.text(
            coord[0],
            coord[1],
            str(mesh.node_tags[i]),
            color="red",
            fontsize=6,
            ha="center",
        )

    # Plot face normals for each element
    for i in range(mesh.nelem):
        for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
            if neighbor_idx != -1:  # Only plot internal faces
                face_nodes_tags = mesh.elem_faces[i][j]
                node_indices = [
                    np.where(mesh.node_tags == tag)[0][0] for tag in face_nodes_tags
                ]
                nodes = mesh.node_coords[np.array(node_indices)]
                midpoint = np.mean(nodes, axis=0)
                normal = mesh.face_normals[i, j]

                # Scale for visibility
                normal_scaled = normal * 1.2

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
    plt.show(block=False)


def plot_simulation_step(mesh: Mesh, U, title=""):
    """
    Plots the water height on the mesh for a single time step.
    """
    node_coords = np.array(mesh.node_coords)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    h = U[:, 0]

    triangles = []
    facecolors = []
    for i, conn in enumerate(mesh.elem_conn):
        node_indices = [np.where(mesh.node_tags == tag)[0][0] for tag in conn]
        if len(node_indices) == 4:
            triangles.append([node_indices[0], node_indices[1], node_indices[2]])
            triangles.append([node_indices[0], node_indices[2], node_indices[3]])
            facecolors.extend([h[i], h[i]])
        else:
            triangles.append(node_indices)
            facecolors.append(h[i])

    plt.figure(figsize=(12, 5))
    plt.tripcolor(
        x, y, triangles=triangles, facecolors=facecolors, shading="flat", cmap="viridis"
    )
    plt.colorbar(label="Water Height (h)")
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def create_animation(mesh: Mesh, history, dt_history, filename="shallow_water.gif"):
    """
    Creates and saves an animation of the simulation history.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    node_coords = np.array(mesh.node_coords)
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    triangles = []
    for conn in mesh.elem_conn:
        node_indices = [np.where(mesh.node_tags == tag)[0][0] for tag in conn]
        if len(node_indices) == 4:
            triangles.append([node_indices[0], node_indices[1], node_indices[2]])
            triangles.append([node_indices[0], node_indices[2], node_indices[3]])
        else:
            triangles.append(node_indices)

    h_initial = history[0][:, 0]
    facecolors_initial = []
    for i, h_val in enumerate(h_initial):
        if len(mesh.elem_conn[i]) == 4:
            facecolors_initial.extend([h_val, h_val])
        else:
            facecolors_initial.append(h_val)

    tpc = ax.tripcolor(
        x,
        y,
        triangles=triangles,
        facecolors=facecolors_initial,
        shading="flat",
        cmap="viridis",
    )
    fig.colorbar(tpc, ax=ax, label="Water Height (h)")
    time_text = ax.set_title(f"Simulation at t = {0.0:.4f}s")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_aspect("equal", adjustable="box")

    def update_frame(frame):
        U = history[frame]
        h = U[:, 0]
        facecolors = []
        for i, h_val in enumerate(h):
            if len(mesh.elem_conn[i]) == 4:
                facecolors.extend([h_val, h_val])
            else:
                facecolors.append(h_val)
        tpc.set_array(facecolors)

        current_time = sum(dt_history[:frame])
        ax.set_title(f"Simulation at t = {current_time:.4f}s")
        return [tpc, time_text]

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(history),
        interval=100,
        blit=False,
        repeat=True,
        repeat_delay=3000,
    )

    plt.show()
