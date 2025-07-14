from src.mesh import Mesh
from src.solver import solve_shallow_water
from src.case_setup import setup_case
from src.visualization import plot_simulation_step, create_animation, plot_mesh
import numpy as np


def main():
    """
    Main function to run the FVM solver.
    """

    # --- 1. Initialize and Read Mesh ---
    mesh = Mesh()
    mesh.read_mesh("data/rectangle_mesh.msh")
    mesh.analyze_mesh()
    mesh.summary()
    # plot_mesh(mesh)  # Optional: visualize the mesh

    # --- 2. Set Up Case ---
    U_init, boundary_conditions = setup_case(mesh)

    # --- 3. Solve ---
    history, dt_history = solve_shallow_water(
        mesh,
        U_init,
        boundary_conditions,
        t_end=100.0,
        g=9.81,
        over_relaxation=1.2,
        limiter="minmod",  # Options: 'barth_jespersen', 'minmod', 'superbee'
        use_adaptive_dt=False,
        cfl=0.5,
        dt_initial=1,
    )

    # --- 4. Visualize ---
    create_animation(mesh, history, dt_history)
    # plot_simulation_step(mesh, history[-1], "Final State")


if __name__ == "__main__":
    main()
