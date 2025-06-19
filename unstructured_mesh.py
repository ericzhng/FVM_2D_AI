import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class UnstructuredMesh:
    def __init__(self, points, connectivity, dim=2):
        self.dim = dim  # 2D or 3D
        self.points = np.array(points)  # Node coordinates
        self.connectivity = np.array(connectivity)  # Cell connectivity
        self.n_cells = len(connectivity)
        self.n_points = len(points)
        self.cell_centers = None
        self.face_centers = None
        self.cell_volumes = None
        self.face_areas = None
        self.face_normals = None
        self.face_tangents = None
        self.cell_neighbors = None
        self.face_neighbors = None
        self.faces = None
        self._build_mesh()

    def _build_mesh(self):
        self._compute_faces()
        self._compute_cell_centers()
        self._compute_face_centers()
        self._compute_cell_volumes()
        self._compute_face_areas()
        self._compute_normals_tangents()
        self._compute_neighbors()

    def _compute_faces(self):
        faces = set()
        face_to_cells = {}
        if self.dim == 2:
            for cell_idx, cell in enumerate(self.connectivity):
                for i in range(len(cell)):
                    face = tuple(sorted([cell[i], cell[(i + 1) % len(cell)]]))
                    faces.add(face)
                    if face not in face_to_cells:
                        face_to_cells[face] = []
                    face_to_cells[face].append(cell_idx)
        elif self.dim == 3:
            for cell_idx, cell in enumerate(self.connectivity):
                for face_indices in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                    face = tuple(sorted([cell[i] for i in face_indices]))
                    faces.add(face)
                    if face not in face_to_cells:
                        face_to_cells[face] = []
                    face_to_cells[face].append(cell_idx)
        self.faces = np.array(list(faces))
        self.face_neighbors = [face_to_cells[face] for face in faces]

    def _compute_cell_centers(self):
        self.cell_centers = np.zeros((self.n_cells, self.dim))
        for i, cell in enumerate(self.connectivity):
            self.cell_centers[i] = np.mean(self.points[cell], axis=0)

    def _compute_face_centers(self):
        self.face_centers = np.zeros((len(self.faces), self.dim))
        for i, face in enumerate(self.faces):
            self.face_centers[i] = np.mean(self.points[face], axis=0)

    def _compute_cell_volumes(self):
        self.cell_volumes = np.zeros(self.n_cells)
        if self.dim == 2:
            for i, cell in enumerate(self.connectivity):
                x, y = self.points[cell, 0], self.points[cell, 1]
                self.cell_volumes[i] = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
        elif self.dim == 3:
            for i, cell in enumerate(self.connectivity):
                v0, v1, v2, v3 = self.points[cell]
                self.cell_volumes[i] = np.abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0

    def _compute_face_areas(self):
        self.face_areas = np.zeros(len(self.faces))
        if self.dim == 2:
            for i, face in enumerate(self.faces):
                p1, p2 = self.points[face]
                self.face_areas[i] = np.linalg.norm(p2 - p1)
        elif self.dim == 3:
            for i, face in enumerate(self.faces):
                v0, v1, v2 = self.points[face]
                self.face_areas[i] = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2.0

    def _compute_normals_tangents(self):
        self.face_normals = np.zeros((len(self.faces), self.dim))
        self.face_tangents = np.zeros((len(self.faces), self.dim))
        if self.dim == 2:
            for i, face in enumerate(self.faces):
                p1, p2 = self.points[face]
                tangent = p2 - p1
                self.face_tangents[i] = tangent / np.linalg.norm(tangent)
                self.face_normals[i] = np.array([-self.face_tangents[i][1], self.face_tangents[i][0]])
                if len(self.face_neighbors[i]) == 2:
                    cell1, cell2 = self.face_neighbors[i]
                    vec = self.cell_centers[cell2] - self.cell_centers[cell1]
                    if np.dot(vec, self.face_normals[i]) < 0:
                        self.face_normals[i] = -self.face_normals[i]
        elif self.dim == 3:
            for i, face in enumerate(self.faces):
                v0, v1, v2 = self.points[face]
                tangent = v1 - v0
                self.face_tangents[i] = tangent / np.linalg.norm(tangent)
                normal = np.cross(v1 - v0, v2 - v0)
                self.face_normals[i] = normal / np.linalg.norm(normal)
                if len(self.face_neighbors[i]) == 2:
                    cell1, cell2 = self.face_neighbors[i]
                    vec = self.cell_centers[cell2] - self.cell_centers[cell1]
                    if np.dot(vec, self.face_normals[i]) < 0:
                        self.face_normals[i] = -self.face_normals[i]

    def _compute_neighbors(self):
        self.cell_neighbors = [[] for _ in range(self.n_cells)]
        for i, face in enumerate(self.faces):
            cells = self.face_neighbors[i]
            if len(cells) == 2:
                self.cell_neighbors[cells[0]].append((cells[1], i))
                self.cell_neighbors[cells[1]].append((cells[0], i))

    def plot_mesh(self):
        if self.dim == 2:
            fig, ax = plt.subplots(figsize=(10, 10))
            for cell_idx, cell in enumerate(self.connectivity):
                cell_points = self.points[cell]
                cell_points = np.append(cell_points, [cell_points[0]], axis=0)
                ax.plot(cell_points[:, 0], cell_points[:, 1], 'b-')
                ax.text(self.cell_centers[cell_idx, 0], self.cell_centers[cell_idx, 1], 
                        str(cell_idx), color='red', fontsize=10, ha='center', va='center')
            for face_idx, face in enumerate(self.faces):
                p1, p2 = self.points[face]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
                ax.text(self.face_centers[face_idx, 0], self.face_centers[face_idx, 1], 
                        str(face_idx), color='blue', fontsize=8, ha='center', va='center')
            ax.set_aspect('equal')
            plt.title('Unstructured 2D Mesh with Cell and Face Indices')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
        elif self.dim == 3:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            for cell_idx, cell in enumerate(self.connectivity):
                for edge in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
                    p1, p2 = self.points[cell[edge[0]]], self.points[cell[edge[1]]]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', alpha=0.3)
                ax.text(self.cell_centers[cell_idx, 0], self.cell_centers[cell_idx, 1], 
                        self.cell_centers[cell_idx, 2], str(cell_idx), color='red', fontsize=10, ha='center', va='center')
            for face_idx, face in enumerate(self.faces):
                v0, v1, v2 = self.points[face]
                ax.plot([v0[0], v1[0], v2[0], v0[0]], [v0[1], v1[1], v2[1], v0[1]], 
                        [v0[2], v1[2], v2[2], v0[2]], 'k-', alpha=0.5)
                ax.text(self.face_centers[face_idx, 0], self.face_centers[face_idx, 1], 
                        self.face_centers[face_idx, 2], str(face_idx), color='blue', fontsize=8, ha='center', va='center')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Unstructured 3D Mesh with Cell and Face Indices')
            plt.show()

def generate_random_mesh(n_points, dim=2, bounds=None):
    np.random.seed(0)
    if dim == 2:
        bounds = bounds or (0, 1, 0, 1)
        x = np.random.uniform(bounds[0], bounds[1], n_points)
        y = np.random.uniform(bounds[2], bounds[3], n_points)
        points = np.vstack((x, y)).T
    elif dim == 3:
        bounds = bounds or (0, 1, 0, 1, 0, 1)
        x = np.random.uniform(bounds[0], bounds[1], n_points)
        y = np.random.uniform(bounds[2], bounds[3], n_points)
        z = np.random.uniform(bounds[4], bounds[5], n_points)
        points = np.vstack((x, y, z)).T
    else:
        raise ValueError("Dimension must be 2 or 3")
    tri = Delaunay(points)
    return UnstructuredMesh(points, tri.simplices, dim=dim)

# Example usage
if __name__ == "__main__":
    # 2D mesh
    mesh_2d = generate_random_mesh(11, dim=2)
    print("2D Mesh:")
    print("Number of cells:", mesh_2d.n_cells)
    print("Number of points:", mesh_2d.n_points)
    print("Number of faces:", len(mesh_2d.faces))
    print("\nSample cell centers:", mesh_2d.cell_centers[:2])
    print("\nSample face centers:", mesh_2d.face_centers[:2])
    print("\nSample cell volumes:", mesh_2d.cell_volumes[:2])
    print("\nSample face areas:", mesh_2d.face_areas[:2])
    print("\nSample face normals:", mesh_2d.face_normals[:2])
    print("\nSample cell neighbors (cell, face):", mesh_2d.cell_neighbors[0])
    print("\nSample face neighbors (cells):", mesh_2d.face_neighbors[:2])
    mesh_2d.plot_mesh()

    # # 3D mesh
    # mesh_3d = generate_random_mesh(6, dim=3)
    # print("\n3D Mesh:")
    # print("Number of cells:", mesh_3d.n_cells)
    # print("Number of points:", mesh_3d.n_points)
    # print("Number of faces:", len(mesh_3d.faces))
    # print("\nSample cell centers:", mesh_3d.cell_centers[:2])
    # print("\nSample face centers:", mesh_3d.face_centers[:2])
    # print("\nSample cell volumes:", mesh_3d.cell_volumes[:2])
    # print("\nSample face areas:", mesh_3d.face_areas[:2])
    # print("\nSample face normals:", mesh_3d.face_normals[:2])
    # print("\nSample cell neighbors (cell, face):", mesh_3d.cell_neighbors[0])
    # print("\nSample face neighbors (cells):", mesh_3d.face_neighbors[:2])
    # mesh_3d.plot_mesh()
