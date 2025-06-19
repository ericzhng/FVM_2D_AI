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

def test_mesh_generation():
    # Example usage
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


class ShallowWaterSolver:
    def __init__(self, mesh, bed_elevation, manning_coeff=0.03, g=9.81):
        self.mesh = mesh
        self.g = g
        self.manning = manning_coeff
        self.bed = bed_elevation  # Bed elevation at cell centers
        self.U = np.zeros((mesh.n_cells, 3))  # [h, hu, hv] at cell centers
        self.grad_U = np.zeros((mesh.n_cells, 3, 2))  # Gradients of [h, hu, hv]
        self.grad_bed = np.zeros((mesh.n_cells, 2))  # Bed slope
        self.dt = 0.0

    def initialize(self, h0, u0=0.0, v0=0.0):
        self.U[:, 0] = h0
        self.U[:, 1] = h0 * u0
        self.U[:, 2] = h0 * v0
        self._compute_gradients()

    def _compute_gradients(self):
        # Least squares gradient computation
        for i in range(self.mesh.n_cells):
            A = np.zeros((2, 2))
            b_h = np.zeros(2)
            b_hu = np.zeros(2)
            b_hv = np.zeros(2)
            b_bed = np.zeros(2)
            for neighbor_idx, face_idx in self.mesh.cell_neighbors[i]:
                dr = self.mesh.cell_centers[neighbor_idx] - self.mesh.cell_centers[i]
                A += np.outer(dr, dr)
                b_h += dr * (self.U[neighbor_idx, 0] - self.U[i, 0])
                b_hu += dr * (self.U[neighbor_idx, 1] - self.U[i, 1])
                b_hv += dr * (self.U[neighbor_idx, 2] - self.U[i, 2])
                b_bed += dr * (self.bed[neighbor_idx] - self.bed[i])
            if np.linalg.det(A) > 1e-10:
                inv_A = np.linalg.inv(A)
                self.grad_U[i, 0] = inv_A @ b_h
                self.grad_U[i, 1] = inv_A @ b_hu
                self.grad_U[i, 2] = inv_A @ b_hv
                self.grad_bed[i] = inv_A @ b_bed

    def _slope_limiter(self):
        # Minmod limiter
        for i in range(self.mesh.n_cells):
            for var in range(3):
                min_grad = np.zeros(2)
                max_grad = np.zeros(2)
                for neighbor_idx, _ in self.mesh.cell_neighbors[i]:
                    grad_n = self.grad_U[neighbor_idx, var]
                    min_grad = np.minimum(min_grad, grad_n)
                    max_grad = np.maximum(max_grad, grad_n)
                self.grad_U[i, var] = np.where(self.grad_U[i, var] > 0,
                                           np.minimum(self.grad_U[i, var], max_grad),
                                           np.maximum(self.grad_U[i, var], min_grad))

    def _reconstruct_face_values(self, face_idx):
        cells = self.mesh.face_neighbors[face_idx]
        U_L = np.zeros(3)
        U_R = np.zeros(3)
        if len(cells) == 2:
            i, j = cells
            dr_L = self.mesh.face_centers[face_idx] - self.mesh.cell_centers[i]
            dr_R = self.mesh.face_centers[face_idx] - self.mesh.cell_centers[j]
            U_L = self.U[i] + np.sum(self.grad_U[i] * dr_L, axis=1)
            U_R = self.U[j] + np.sum(self.grad_U[j] * dr_R, axis=1)
            # Non-orthogonality correction
            d_cc = self.mesh.cell_centers[j] - self.mesh.cell_centers[i]
            n = self.mesh.face_normals[face_idx]
            if np.dot(d_cc, n) == 0:
                return U_L, U_R
            alpha = np.dot(dr_L, n) / np.dot(d_cc, n)
            U_avg = (1 - alpha) * self.U[i] + alpha * self.U[j]
            U_L += (U_avg - U_L) * np.dot(d_cc, n) / np.dot(dr_L, n)
            U_R += (U_avg - U_R) * np.dot(d_cc, n) / np.dot(dr_R, n)
        else:
            i = cells[0]
            dr = self.mesh.face_centers[face_idx] - self.mesh.cell_centers[i]
            U_L = self.U[i] + np.sum(self.grad_U[i] * dr, axis=1)
            U_R = U_L  # Boundary condition (simplified)
        return U_L, U_R

    def _numerical_flux(self, U_L, U_R, n):
        h_L, hu_L, hv_L = U_L
        h_R, hu_R, hv_R = U_R
        u_L = hu_L / h_L if h_L > 1e-6 else 0
        v_L = hv_L / h_L if h_L > 1e-6 else 0
        u_R = hu_R / h_R if h_R > 1e-6 else 0
        v_R = hv_R / h_R if h_R > 1e-6 else 0
        vel_L = np.array([u_L, v_L])
        vel_R = np.array([u_R, v_R])
        # Roe average
        h_avg = 0.5 * (h_L + h_R)
        u_avg = (np.sqrt(h_L) * u_L + np.sqrt(h_R) * u_R) / (np.sqrt(h_L) + np.sqrt(h_R))
        v_avg = (np.sqrt(h_L) * v_L + np.sqrt(h_R) * v_R) / (np.sqrt(h_L) + np.sqrt(h_R))
        c_avg = np.sqrt(self.g * h_avg)
        # Eigenvalues
        vel_n = u_avg * n[0] + v_avg * n[1]
        lambda1 = vel_n - c_avg
        lambda2 = vel_n
        lambda3 = vel_n + c_avg
        # Numerical flux (Roe solver)
        F_L = np.array([hu_L, hu_L * u_L + 0.5 * self.g * h_L**2, hu_L * v_L])
        F_R = np.array([hu_R, hu_R * u_R + 0.5 * self.g * h_R**2, hu_R * v_R])
        if abs(lambda1) > abs(lambda3):
            return F_L if lambda1 > 0 else F_R
        else:
            return 0.5 * (F_L + F_R - np.abs(vel_n) * (U_R - U_L))

    def _source_terms(self, i):
        h, hu, hv = self.U[i]
        u = hu / h if h > 1e-6 else 0
        v = hv / h if h > 1e-6 else 0
        # Bed slope source term
        S_slope = np.zeros(3)
        S_slope[1] = -self.g * h * self.grad_bed[i, 0]
        S_slope[2] = -self.g * h * self.grad_bed[i, 1]
        # Bed friction (Manning)
        vel_mag = np.sqrt(u**2 + v**2)
        if vel_mag > 1e-6:
            cf = self.g * self.manning**2 / h**(1/3)
            S_friction = np.array([0, -cf * hu * vel_mag, -cf * hv * vel_mag])
        else:
            S_friction = np.zeros(3)
        return S_slope + S_friction

    def compute_dt(self):
        max_speed = 0
        for i in range(self.mesh.n_cells):
            h = self.U[i, 0]
            u = self.U[i, 1] / h if h > 1e-6 else 0
            v = self.U[i, 2] / h if h > 1e-6 else 0
            c = np.sqrt(self.g * h)
            max_speed = max(max_speed, np.sqrt(u**2 + v**2) + c)
        return 0.5 * np.min(self.mesh.cell_volumes) / max_speed

    def step(self):
        self._compute_gradients()
        self._slope_limiter()
        dU = np.zeros((self.mesh.n_cells, 3))
        for face_idx in range(len(self.mesh.faces)):
            U_L, U_R = self._reconstruct_face_values(face_idx)
            n = self.mesh.face_normals[face_idx]
            flux = self._numerical_flux(U_L, U_R, n) * self.mesh.face_areas[face_idx]
            cells = self.mesh.face_neighbors[face_idx]
            if len(cells) == 2:
                dU[cells[0]] -= flux
                dU[cells[1]] += flux
            else:
                dU[cells[0]] -= flux
        for i in range(self.mesh.n_cells):
            dU[i] += self._source_terms(i) * self.mesh.cell_volumes[i]
            self.U[i] += (self.dt / self.mesh.cell_volumes[i]) * dU[i]

    def plot_solution(self):
        plt.figure(figsize=(10, 10))
        plt.tricontourf(self.mesh.cell_centers[:, 0], self.mesh.cell_centers[:, 1], self.U[:, 0], cmap='viridis')
        plt.colorbar(label='Water Depth (h)')
        for cell_idx, cell in enumerate(self.mesh.connectivity):
            cell_points = self.mesh.points[cell]
            cell_points = np.append(cell_points, [cell_points[0]], axis=0)
            plt.plot(cell_points[:, 0], cell_points[:, 1], 'k-', alpha=0.2)
        plt.title('Shallow Water Solution: Water Depth')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.show()

# Example usage
if __name__ == "__main__":
    mesh = generate_random_mesh(1000)
    bed = np.random.uniform(100, 200, mesh.n_cells)  # Random bed elevation
    solver = ShallowWaterSolver(mesh, bed)
    solver.initialize(h0=1.0, u0=0.0, v0=0.0)
    solver.dt = solver.compute_dt()
    for _ in range(100):
        solver.step()
    solver.plot_solution()
