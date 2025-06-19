import numpy as np
import matplotlib.pyplot as plt


class OneDWaterDomain:
    def __init__(self, x, h, u, bed, g=9.81):
        self.x = np.array(x)
        self.h = np.array(h)
        self.u = np.array(u)
        self.bed = np.array(bed)
        self.g = g
        self.q = self.h * self.u
        self.dx = np.diff(x).mean()
        self.dt = 0.0

    def compute_dt(self):
        max_speed = np.max(np.abs(self.u) + np.sqrt(self.g * self.h))
        return 0.5 * self.dx / max_speed if max_speed > 0 else 1e-3

    def step(self):
        flux = np.zeros((len(self.x), 2))
        for i in range(len(self.x) - 1):
            h_L, q_L = self.h[i], self.q[i]
            h_R, q_R = self.h[i + 1], self.q[i + 1]
            u_L = q_L / h_L if h_L > 1e-6 else 0
            u_R = q_R / h_R if h_R > 1e-6 else 0
            h_avg = 0.5 * (h_L + h_R)
            u_avg = (np.sqrt(h_L) * u_L + np.sqrt(h_R) * u_R) / (
                np.sqrt(h_L) + np.sqrt(h_R)
            )
            c_avg = np.sqrt(self.g * h_avg)
            F_L = np.array([q_L, q_L * u_L + 0.5 * self.g * h_L**2])
            F_R = np.array([q_R, q_R * u_R + 0.5 * self.g * h_R**2])
            flux[i] = 0.5 * (
                F_L
                + F_R
                - np.abs(u_avg) * (np.array([h_R, q_R]) - np.array([h_L, q_L]))
            )
        dU = np.zeros((len(self.x), 2))
        for i in range(1, len(self.x) - 1):
            dU[i] = -(flux[i] - flux[i - 1]) / self.dx
            dU[i, 1] -= self.g * self.h[i] * (self.bed[i + 1] - self.bed[i]) / self.dx
        self.h += self.dt * dU[:, 0]
        self.q += self.dt * dU[:, 1]
        self.h = np.maximum(self.h, 1e-6)
        self.u = self.q / self.h
        self.u[self.h <= 1e-6] = 0
