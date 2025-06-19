import numpy as np
import matplotlib.pyplot as plt


class ShallowWaterSolver:
    def __init__(
        self,
        mesh,
        bed_elevation,
        boundary_conditions,
        one_d_domain=None,
        manning_coeff=0.03,
        g=9.81,
        h_dry=1e-6,
    ):
        self.mesh = mesh
        self.g = g
        self.manning = manning_coeff
        self.h_dry = h_dry
        self.bed = np.array(bed_elevation)
        self.U = np.zeros((mesh.n_cells, 3))
        self.grad_U = np.zeros((mesh.n_cells, 3, 2))
        self.grad_bed = np.zeros((mesh.n_cells, 2))
        self.is_wet = np.ones(mesh.n_cells, dtype=bool)
        self.boundary_types = boundary_conditions
        self.one_d_domain = one_d_domain
        self.one_d_coupling_faces = []
        self.dt = 0.0
        if boundary_conditions:
            self.mesh.add_ghost_cells(boundary_conditions)

    def initialize(self, h0, u0=0.0, v0=0.0):
        self.U[:, 0] = np.maximum(h0, self.h_dry)
        self.U[:, 1] = h0 * u0
        self.U[:, 2] = h0 * v0
        self.is_wet = self.U[:, 0] > self.h_dry
        self._compute_gradients()

    def _compute_gradients(self):
        for i in range(self.mesh.n_cells - self.mesh.n_ghost):
            if not self.is_wet[i]:
                self.grad_U[i] = 0
                self.grad_bed[i] = 0
                continue
            A = np.zeros((2, 2))
            b_h = np.zeros(2)
            b_hu = np.zeros(2)
            b_hv = np.zeros(2)
            b_bed = np.zeros(2)
            for neighbor_idx, _ in self.mesh.cell_neighbors[i]:
                if (
                    neighbor_idx < self.mesh.n_cells - self.mesh.n_ghost
                    and self.is_wet[neighbor_idx]
                ):
                    dr = (
                        self.mesh.cell_centers[neighbor_idx] - self.mesh.cell_centers[i]
                    )
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
        for i in range(self.mesh.n_cells - self.mesh.n_ghost):
            if not self.is_wet[i]:
                continue
            for var in range(3):
                min_grad = np.zeros(2)
                max_grad = np.zeros(2)
                for neighbor_idx, _ in self.mesh.cell_neighbors[i]:
                    if (
                        neighbor_idx < self.mesh.n_cells - self.mesh.n_ghost
                        and self.is_wet[neighbor_idx]
                    ):
                        grad_n = self.grad_U[neighbor_idx, var]
                        min_grad = np.minimum(min_grad, grad_n)
                        max_grad = np.maximum(max_grad, grad_n)
                self.grad_U[i, var] = np.where(
                    self.grad_U[i, var] > 0,
                    np.minimum(self.grad_U[i, var], max_grad),
                    np.maximum(self.grad_U[i, var], min_grad),
                )

    def _set_ghost_values(self):
        for face_idx in self.mesh.boundary_faces:
            ghost_idx = self.mesh.face_neighbors[face_idx][1]
            real_idx = self.mesh.face_neighbors[face_idx][0]
            bc_type = self.boundary_types.get(face_idx, "reflective")
            if bc_type == "reflective":
                self.U[ghost_idx] = self.U[real_idx].copy()
                self.U[ghost_idx, 1] = -self.U[real_idx, 1]
                self.U[ghost_idx, 2] = -self.U[real_idx, 2]
                self.is_wet[ghost_idx] = self.is_wet[real_idx]
                self.bed[ghost_idx] = self.bed[real_idx]
            elif bc_type == "inflow":
                self.U[ghost_idx] = np.array(
                    [1.0, 0.5, 0.0]
                )  # Example: h=1, u=0.5, v=0
                self.is_wet[ghost_idx] = True
                self.bed[ghost_idx] = self.bed[real_idx]
            elif bc_type == "outflow":
                self.U[ghost_idx] = self.U[real_idx].copy()
                self.is_wet[ghost_idx] = self.is_wet[real_idx]
                self.bed[ghost_idx] = self.bed[real_idx]
            elif bc_type == "1d_coupling":
                self.one_d_coupling_faces.append(face_idx)
                self.U[ghost_idx] = np.array([self.h_dry, 0, 0])
                self.is_wet[ghost_idx] = False
                self.bed[ghost_idx] = self.bed[real_idx]

    def _reconstruct_face_values(self, face_idx):
        cells = self.mesh.face_neighbors[face_idx]
        i, j = cells
        is_wet_L = self.is_wet[i]
        is_wet_R = self.is_wet[j]
        dr_L = self.mesh.face_centers[face_idx] - self.mesh.cell_centers[i]
        dr_R = self.mesh.face_centers[face_idx] - self.mesh.cell_centers[j]
        U_L = (
            self.U[i] + np.sum(self.grad_U[i] * dr_L, axis=1)
            if is_wet_L
            else np.array([self.h_dry, 0, 0])
        )
        U_R = (
            self.U[j] + np.sum(self.grad_U[j] * dr_R, axis=1)
            if is_wet_R
            else np.array([self.h_dry, 0, 0])
        )
        U_L[0] = max(U_L[0], self.h_dry)
        U_R[0] = max(U_R[0], self.h_dry)
        if (
            i < self.mesh.n_cells - self.mesh.n_ghost
            and j < self.mesh.n_cells - self.mesh.n_ghost
        ):
            d_cc = self.mesh.cell_centers[j] - self.mesh.cell_centers[i]
            n = self.mesh.face_normals[face_idx]
            if np.dot(d_cc, n) != 0 and is_wet_L and is_wet_R:
                alpha = np.dot(dr_L, n) / np.dot(d_cc, n)
                U_avg = (1 - alpha) * self.U[i] + alpha * self.U[j]
                U_L += (U_avg - U_L) * np.dot(d_cc, n) / np.dot(dr_L, n)
                U_R += (U_avg - U_R) * np.dot(d_cc, n) / np.dot(dr_R, n)
        return U_L, U_R, is_wet_L, is_wet_R

    def _numerical_flux(self, U_L, U_R, n, is_wet_L, is_wet_R):
        if not is_wet_L and not is_wet_R:
            return np.zeros(3)
        h_L, hu_L, hv_L = U_L
        h_R, hu_R, hv_R = U_R
        u_L = hu_L / h_L if h_L > self.h_dry else 0
        v_L = hv_L / h_L if h_L > self.h_dry else 0
        u_R = hu_R / h_R if h_R > self.h_dry else 0
        v_R = hv_R / h_R if h_R > self.h_dry else 0
        vel_L = np.array([u_L, v_L])
        vel_R = np.array([u_R, v_R])
        h_avg = 0.5 * (h_L + h_R)
        u_avg = (
            (np.sqrt(h_L) * u_L + np.sqrt(h_R) * u_R) / (np.sqrt(h_L) + np.sqrt(h_R))
            if h_L > self.h_dry and h_R > self.h_dry
            else 0
        )
        v_avg = (
            (np.sqrt(h_L) * v_L + np.sqrt(h_R) * v_R) / (np.sqrt(h_L) + np.sqrt(h_R))
            if h_L > self.h_dry and h_R > self.h_dry
            else 0
        )
        c_avg = np.sqrt(self.g * h_avg) if h_avg > self.h_dry else 0
        vel_n = u_avg * n[0] + v_avg * n[1]
        lambda1 = vel_n - c_avg
        lambda2 = vel_n
        lambda3 = vel_n + c_avg
        F_L = (
            np.array([hu_L, hu_L * u_L + 0.5 * self.g * h_L**2, hu_L * v_L])
            if is_wet_L
            else np.zeros(3)
        )
        F_R = (
            np.array([hu_R, hu_R * u_R + 0.5 * self.g * h_R**2, hu_R * v_R])
            if is_wet_R
            else np.zeros(3)
        )
        if abs(lambda1) > abs(lambda3):
            return F_L if lambda1 > 0 else F_R
        else:
            return 0.5 * (F_L + F_R - np.abs(vel_n) * (U_R - U_L))

    def _source_terms(self, i):
        if not self.is_wet[i]:
            return np.zeros(3)
        h, hu, hv = self.U[i]
        u = hu / h if h > self.h_dry else 0
        v = hv / h if h > self.h_dry else 0
        S_slope = np.zeros(3)
        S_slope[1] = -self.g * h * self.grad_bed[i, 0]
        S_slope[2] = -self.g * h * self.grad_bed[i, 1]
        vel_mag = np.sqrt(u**2 + v**2)
        if vel_mag > 1e-6:
            cf = self.g * self.manning**2 / h ** (1 / 3)
            S_friction = np.array([0, -cf * hu * vel_mag, -cf * hv * vel_mag])
        else:
            S_friction = np.zeros(3)
        return S_slope + S_friction

    def _couple_1d_2d(self):
        if not self.one_d_domain or not self.one_d_coupling_faces:
            return
        for face_idx in self.one_d_coupling_faces:
            real_idx = self.mesh.face_neighbors[face_idx][0]
            ghost_idx = self.mesh.face_neighbors[face_idx][1]
            h_2d = self.U[real_idx, 0]
            u_2d = self.U[real_idx, 1] / h_2d if h_2d > self.h_dry else 0
            n = self.mesh.face_normals[face_idx]
            vel_n = u_2d * n[0]
            idx_1d = 0  # Assuming coupling at x=0 of 1D domain
            self.one_d_domain.h[idx_1d] = h_2d
            self.one_d_domain.u[idx_1d] = vel_n
            self.one_d_domain.q[idx_1d] = h_2d * vel_n
            h_1d = self.one_d_domain.h[idx_1d]
            q_1d = self.one_d_domain.q[idx_1d]
            u_1d = q_1d / h_1d if h_1d > 1e-6 else 0
            self.U[ghost_idx] = np.array([h_1d, h_1d * u_1d * n[0], h_1d * u_1d * n[1]])
            self.is_wet[ghost_idx] = h_1d > self.h_dry

    def compute_dt(self):
        max_speed = 0
        for i in range(self.mesh.n_cells - self.mesh.n_ghost):
            if not self.is_wet[i]:
                continue
            h = self.U[i, 0]
            u = self.U[i, 1] / h if h > self.h_dry else 0
            v = self.U[i, 2] / h if h > self.h_dry else 0
            c = np.sqrt(self.g * h)
            max_speed = max(max_speed, np.sqrt(u**2 + v**2) + c)
        dt_2d = (
            0.5 * np.min(self.mesh.cell_volumes) / max_speed if max_speed > 0 else 1e-3
        )
        dt_1d = self.one_d_domain.compute_dt() if self.one_d_domain else dt_2d
        return min(dt_2d, dt_1d)

    def step(self):
        self._compute_gradients()
        self._slope_limiter()
        self._set_ghost_values()
        self._couple_1d_2d()
        if self.one_d_domain:
            self.one_d_domain.step()
        dU = np.zeros((self.mesh.n_cells, 3))
        for face_idx in range(len(self.mesh.faces)):
            U_L, U_R, is_wet_L, is_wet_R = self._reconstruct_face_values(face_idx)
            n = self.mesh.face_normals[face_idx]
            flux = (
                self._numerical_flux(U_L, U_R, n, is_wet_L, is_wet_R)
                * self.mesh.face_areas[face_idx]
            )
            cells = self.mesh.face_neighbors[face_idx]
            if is_wet_L:
                dU[cells[0]] -= flux
            if is_wet_R and cells[1] < self.mesh.n_cells - self.mesh.n_ghost:
                dU[cells[1]] += flux
        for i in range(self.mesh.n_cells - self.mesh.n_ghost):
            if self.is_wet[i]:
                dU[i] += self._source_terms(i) * self.mesh.cell_volumes[i]
                self.U[i] += (self.dt / self.mesh.cell_volumes[i]) * dU[i]
                self.U[i, 0] = max(self.U[i, 0], self.h_dry)
                self.U[i, 1] = self.U[i, 1] if self.U[i, 0] > self.h_dry else 0
                self.U[i, 2] = self.U[i, 2] if self.U[i, 0] > self.h_dry else 0
                self.is_wet[i] = self.U[i, 0] > self.h_dry

    def plot_solution(self):
        plt.figure(figsize=(10, 10))
        plt.tricontourf(
            self.mesh.cell_centers[: self.mesh.n_cells - self.mesh.n_ghost, 0],
            self.mesh.cell_centers[: self.mesh.n_cells - self.mesh.n_ghost, 1],
            self.U[: self.mesh.n_cells - self.mesh.n_ghost, 0],
            cmap="viridis",
        )
        plt.colorbar(label="Water Depth (h)")
        for cell_idx, cell in enumerate(
            self.mesh.connectivity[: self.mesh.n_cells - self.mesh.n_ghost]
        ):
            cell_points = self.mesh.points[cell]
            cell_points = np.append(cell_points, [cell_points[0]], axis=0)
            plt.plot(cell_points[:, 0], cell_points[:, 1], "k-", alpha=0.2)
        plt.title("Shallow Water Solution: Water Depth")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.show()
        if self.one_d_domain:
            plt.figure(figsize=(10, 5))
            plt.plot(self.one_d_domain.x, self.one_d_domain.h, "b-")
            plt.title("1D Domain: Water Depth")
            plt.xlabel("X")
            plt.ylabel("Water Depth (h)")
            plt.show()
