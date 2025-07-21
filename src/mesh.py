import numpy as np
import gmsh

import matplotlib.tri as tri
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class Mesh:
    """
    A class to represent a computational mesh for 1D, 2D, or 3D simulations,
    providing geometric and connectivity information for Finite Volume Methods.
    """

    def __init__(self):
        """
        Initializes the Mesh object with empty attributes.
        """
        self.mesh_file = np.array([])
        self.dim = 0
        self.node_tags = np.array([])
        self.node_coords = np.array([])
        self.elem_tags = np.array([])
        self.elem_conn = np.array([])
        self.nelem = 0
        self.nnode = 0
        self.cell_volumes = np.array([])
        self.cell_centroids = np.array([])
        self.face_normals = np.array([])
        self.face_tangentials = np.array([])
        self.face_areas = np.array([])
        self.boundary_faces = {}
        self.cell_neighbors = np.array([])
        self.elem_faces = np.array([])

    def read_mesh(self, mesh_file):
        """
        Reads the mesh file using gmsh, determines the highest dimension,
        and extracts node and element information.

        Args:
            mesh_file (str): Path to the mesh file (e.g., .msh).
        """
        self.mesh_file = mesh_file
        gmsh.initialize()
        gmsh.open(self.mesh_file)

        self.node_tags, self.node_coords, _ = gmsh.model.mesh.getNodes()
        self.node_coords = np.array(self.node_coords).reshape(-1, 3)
        self.nnode = len(self.node_tags)

        elem_types, elem_tags, node_connectivity = gmsh.model.mesh.getElements()

        max_dim = 0
        main_elem_type_idx = -1

        for i, e_type in enumerate(elem_types):
            _, dim, _, _, _, _ = gmsh.model.mesh.getElementProperties(e_type)
            if dim > max_dim:
                max_dim = dim
                main_elem_type_idx = i

        self.dim = max_dim

        if main_elem_type_idx != -1:
            main_elem_type = elem_types[main_elem_type_idx]
            self.elem_tags = elem_tags[main_elem_type_idx]

            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                main_elem_type
            )
            self.elem_conn = np.array(node_connectivity[main_elem_type_idx]).reshape(
                -1, num_nodes
            )
            self.nelem = len(self.elem_tags)

        self._get_boundary_info()
        gmsh.finalize()

    def analyze_mesh(self):
        """
        Analyzes the mesh to compute all geometric and connectivity properties
        required for a Finite Volume Method solver.
        """
        if self.node_tags is None:
            raise RuntimeError("Mesh data has not been read. Call read_mesh() first.")

        self._compute_cell_centroids()
        self._compute_mesh_properties()
        self._compute_cell_volumes()

    def _get_boundary_info(self):
        """
        Extracts boundary faces and their corresponding physical group tags.
        """
        boundary_dim = self.dim - 1
        if boundary_dim < 0:
            return

        physical_groups = gmsh.model.getPhysicalGroups(dim=boundary_dim)
        for dim, tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity in entities:
                b_elem_types, b_elem_tags, b_node_tags = gmsh.model.mesh.getElements(
                    dim, entity
                )
                for i, elem_type in enumerate(b_elem_types):
                    _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                        elem_type
                    )

                    faces_nodes = np.array(b_node_tags[i]).reshape(-1, num_nodes)
                    for face_nodes in faces_nodes:
                        face = tuple(sorted(face_nodes))
                        self.boundary_faces[face] = {"name": name, "tag": tag}

    def _compute_cell_centroids(self):
        """Computes the centroid of each element."""
        self.cell_centroids = np.zeros((self.nelem, 3))
        for i, elem_nodes_tags in enumerate(self.elem_conn):
            node_indices = [
                np.where(self.node_tags == tag)[0][0] for tag in elem_nodes_tags
            ]
            nodes = self.node_coords[np.array(node_indices)]
            self.cell_centroids[i] = np.mean(nodes, axis=0)

    def _compute_cell_volumes(self):
        """Computes the volume/area of each element."""
        self.cell_volumes = np.zeros(self.nelem)
        for i, elem_nodes_tags in enumerate(self.elem_conn):
            node_indices = [
                np.where(self.node_tags == tag)[0][0] for tag in elem_nodes_tags
            ]
            nodes = self.node_coords[np.array(node_indices)]

            if self.dim == 1:
                self.cell_volumes[i] = np.linalg.norm(nodes[1] - nodes[0])
            elif self.dim == 2:
                x = nodes[:, 0]
                y = nodes[:, 1]
                self.cell_volumes[i] = 0.5 * np.abs(
                    np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
                )
            elif self.dim == 3:
                # Volume calculation using divergence theorem, based on face properties
                volume = 0.0
                for j, face_nodes in enumerate(self.elem_faces[i]):
                    node_indices_face = [
                        np.where(self.node_tags == tag)[0][0] for tag in face_nodes
                    ]
                    face_midpoint = np.mean(
                        self.node_coords[np.array(node_indices_face)],
                        axis=0,
                    )
                    face_normal = self.face_normals[i, j]
                    face_area = self.face_areas[i, j]
                    volume += np.dot(face_midpoint, face_normal) * face_area
                self.cell_volumes[i] = volume / 3.0

    def _compute_mesh_properties(self):
        """
        Computes cell neighbors and face properties (normals, tangentials, areas).
        """
        face_to_elems = {}
        face_definitions = {
            "tet": [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
            "hex": [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
            ],
            "wedge": [[0, 1, 2], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
        }

        num_nodes_per_elem = self.elem_conn.shape[1]

        if self.dim == 2:
            faces_per_elem = num_nodes_per_elem
            face_nodes_def = [
                [i, (i + 1) % num_nodes_per_elem] for i in range(num_nodes_per_elem)
            ]
        elif self.dim == 3:
            if num_nodes_per_elem == 4:
                face_nodes_def = face_definitions["tet"]
            elif num_nodes_per_elem == 8:
                face_nodes_def = face_definitions["hex"]
            elif num_nodes_per_elem == 6:
                face_nodes_def = face_definitions["wedge"]
            else:
                raise NotImplementedError(
                    f"3D elements with {num_nodes_per_elem} nodes are not supported."
                )
            faces_per_elem = len(face_nodes_def)
        else:
            faces_per_elem = 0

        self.cell_neighbors = -np.ones((self.nelem, faces_per_elem), dtype=int)
        self.face_normals = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_tangentials = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_areas = np.zeros((self.nelem, faces_per_elem))
        self.face_midpoints = np.zeros((self.nelem, faces_per_elem, 3))
        self.face_to_cell_distances = np.zeros((self.nelem, faces_per_elem, 2))
        self.elem_faces = [[] for _ in range(self.nelem)]

        for i in range(self.nelem):
            elem_nodes = self.elem_conn[i]
            for j in range(faces_per_elem):
                face_node_indices = face_nodes_def[j]
                face_nodes = tuple(sorted(elem_nodes[k] for k in face_node_indices))

                self.elem_faces[i].append(face_nodes)
                if face_nodes not in face_to_elems:
                    face_to_elems[face_nodes] = []
                face_to_elems[face_nodes].append(i)

        for i in range(self.nelem):
            for j, face_nodes in enumerate(self.elem_faces[i]):
                elems = face_to_elems[face_nodes]
                neighbor_idx = -1
                if len(elems) > 1:
                    neighbor_idx = elems[0] if elems[1] == i else elems[1]
                self.cell_neighbors[i, j] = neighbor_idx

                node_indices = [
                    np.where(self.node_tags == tag)[0][0] for tag in face_nodes
                ]
                nodes = self.node_coords[np.array(node_indices)]
                face_midpoint = np.mean(nodes, axis=0)
                self.face_midpoints[i, j] = face_midpoint

                d_i = np.linalg.norm(face_midpoint - self.cell_centroids[i])
                d_j = (
                    np.linalg.norm(face_midpoint - self.cell_centroids[neighbor_idx])
                    if neighbor_idx != -1
                    else 0
                )
                self.face_to_cell_distances[i, j] = [d_i, d_j]

                if self.dim == 2:
                    p1, p2 = nodes[0], nodes[1]
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    length = np.sqrt(dx * dx + dy * dy)
                    self.face_areas[i, j] = length
                    normal = (
                        np.array([dy / length, -dx / length, 0])
                        if length > 1e-9
                        else np.zeros(3)
                    )
                    tangent = (
                        np.array([dx / length, dy / length, 0])
                        if length > 1e-9
                        else np.zeros(3)
                    )

                    if np.dot(normal, face_midpoint - self.cell_centroids[i]) < 0:
                        normal = -normal
                    self.face_normals[i, j] = normal
                    self.face_tangentials[i, j] = tangent

                elif self.dim == 3 and len(nodes) >= 3:
                    v1 = nodes[1] - nodes[0]
                    v2 = nodes[2] - nodes[0]
                    normal = np.cross(v1, v2)
                    area = np.linalg.norm(normal) / 2.0
                    self.face_areas[i, j] = area

                    if area > 1e-9:
                        normal /= 2.0 * area
                        tangent = v1 / np.linalg.norm(v1)
                    else:
                        normal = np.zeros(3)
                        tangent = np.zeros(3)

                    face_midpoint = np.mean(nodes, axis=0)
                    if np.dot(normal, face_midpoint - self.cell_centroids[i]) < 0:
                        normal = -normal
                    self.face_normals[i, j] = normal
                    self.face_tangentials[i, j] = tangent

    def get_mesh_quality(self, metric="aspect_ratio"):
        """
        Computes mesh quality for each element.
        """
        quality = np.zeros(self.nelem)
        if self.dim == 1:
            return np.ones(self.nelem)

        for i, elem_nodes_tags in enumerate(self.elem_conn):
            node_indices = [
                np.where(self.node_tags == tag)[0][0] for tag in elem_nodes_tags
            ]
            nodes = self.node_coords[np.array(node_indices)]

            num_nodes = len(nodes)
            edge_lengths = np.array(
                [
                    np.linalg.norm(nodes[j] - nodes[(j + 1) % num_nodes])
                    for j in range(num_nodes)
                ]
            )

            if self.dim == 2:
                if min(edge_lengths) > 1e-9:
                    quality[i] = max(edge_lengths) / min(edge_lengths)
                else:
                    quality[i] = float("inf")
            elif self.dim == 3:
                # Simple aspect ratio for 3D: longest edge / shortest edge
                all_edge_lengths = []
                for face in self.elem_faces[i]:
                    face_node_indices = [
                        np.where(self.node_tags == tag)[0][0] for tag in face
                    ]
                    face_nodes_coords = self.node_coords[np.array(face_node_indices)]
                    for k in range(len(face_nodes_coords)):
                        p1 = face_nodes_coords[k]
                        p2 = face_nodes_coords[(k + 1) % len(face_nodes_coords)]
                        all_edge_lengths.append(np.linalg.norm(p1 - p2))

                if min(all_edge_lengths) > 1e-9:
                    quality[i] = max(all_edge_lengths) / min(all_edge_lengths)
                else:
                    quality[i] = float("inf")
        return quality

    def summary(self):
        """
        Prints a summary of the mesh information.
        """
        print("\n--- Mesh Summary ---")
        print(f"Mesh File: {self.mesh_file}")
        print(f"Mesh Dimension: {self.dim}D")
        print(f"Number of Nodes: {self.nnode}")
        print(f"Number of Elements: {self.nelem}")

        if self.nelem > 0:
            print(f"Element Type: {self.elem_conn.shape[1]}-node elements")
            avg_quality = np.mean(self.get_mesh_quality())
            print(f"Average Mesh Quality (Aspect Ratio): {avg_quality:.4f}")

        num_boundary_sets = len(
            np.unique([f["tag"] for f in self.boundary_faces.values()])
        )
        print(f"Number of Boundary Face Sets: {num_boundary_sets}")
        if num_boundary_sets > 0:
            for tag in np.unique([f["tag"] for f in self.boundary_faces.values()]):
                name = [
                    f["name"] for f in self.boundary_faces.values() if f["tag"] == tag
                ][0]
                count = len(
                    [f for f in self.boundary_faces.values() if f["tag"] == tag]
                )
                print(f"  - Boundary '{name}' (tag {tag}): {count} faces")
        print("--------------------\n")

    def get_mesh_data(self):
        """
        Returns all the computed mesh data in a dictionary.
        """
        return {
            "dimension": self.dim,
            "node_tags": self.node_tags,
            "node_coords": self.node_coords,
            "elem_tags": self.elem_tags,
            "elem_conn": self.elem_conn,
            "cell_volumes": self.cell_volumes,
            "cell_centroids": self.cell_centroids,
            "cell_neighbors": self.cell_neighbors,
            "boundary_faces": self.boundary_faces,
            "face_areas": self.face_areas,
            "face_normals": self.face_normals,
            "face_tangentials": self.face_tangentials,
        }


def plot_mesh(mesh: Mesh):
    """
    Visualizes the computational mesh, including element and node labels, and face normals.

    This function is useful for debugging and verifying the mesh structure.

    Args:
        mesh (Mesh): The mesh object to visualize.
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot elements and their labels
    for i, elem_nodes_tags in enumerate(mesh.elem_conn):
        node_indices = [
            np.where(mesh.node_tags == tag)[0][0] for tag in elem_nodes_tags
        ]
        nodes = mesh.node_coords[np.array(node_indices)]
        polygon = Polygon(nodes[:, :2], edgecolor="b", facecolor="none", lw=0.5)
        ax.add_patch(polygon)
        ax.text(
            mesh.cell_centroids[i, 0],
            mesh.cell_centroids[i, 1],
            f"{i} (A={mesh.cell_volumes[i]:.2f})",
            color="blue",
            fontsize=8,
            ha="center",
        )
    # Plot node labels
    for i, coord in enumerate(mesh.node_coords):
        ax.text(
            coord[0],
            coord[1],
            str(mesh.node_tags[i]),
            color="red",
            fontsize=8,
            ha="center",
        )

    # Plot face normals
    for i in range(mesh.nelem):
        for j, _ in enumerate(mesh.elem_faces[i]):
            midpoint = mesh.face_midpoints[i, j]
            normal = mesh.face_normals[i, j]
            face_to_cell_distances = mesh.face_to_cell_distances[i, j][0]

            # Scale for visibility
            normal_scaled = normal * face_to_cell_distances * 0.5

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
                width=0.003,
            )

    ax.set_aspect("equal", "box")
    ax.set_title("Mesh Visualization")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    try:
        mesh_file = "./data/complex_shape_mesh.msh"

        # New workflow
        mesh = Mesh()
        mesh.read_mesh(mesh_file)
        mesh.analyze_mesh()

        mesh.summary()

        mesh_data = mesh.get_mesh_data()
        print("\n--- Mesh Data Export ---")
        print(f"First 5 node coordinates:\n{mesh_data['node_coords'][:5]}")
        print(f"First 5 element connectivities:\n{mesh_data['elem_conn'][:5]}")

    except FileNotFoundError:
        print(f"Error: Mesh file not found at {mesh_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

    plot_mesh(mesh)
