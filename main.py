from src.mesh import Mesh
from src.solver import solve_shallow_water
from src.case_setup import setup_dam_break_scenario
from src.visualization import plot_simulation_step, create_animation, plot_mesh


def main():
    """
    Main function to run the FVM solver.
    """

    # --- 1. Initialize and Read Mesh ---
    mesh = Mesh()
    mesh.read_mesh("data/rectangle_mesh.msh")
    mesh.analyze_mesh()
    mesh.summary()
    plot_mesh(mesh)  # Optional: visualize the mesh

    # --- 2. Set Up Case ---
    U_initial, boundary_conditions = setup_dam_break_scenario(mesh)

    # --- 3. Solve ---
    history, dt_history = solve_shallow_water(
        mesh,
        U_initial,
        boundary_conditions,
        t_end=1.0,
        g=9.81,
        over_relaxation=1.2,
        limiter="minmod",  # Options: 'barth_jespersen', 'minmod', 'superbee'
        use_adaptive_dt=False,
        cfl=0.5,
    )

    # --- 4. Visualize ---
    # plot_simulation_step(mesh, history[-1], "Final State")
    create_animation(mesh, history, dt_history)


if __name__ == "__main__":
    main()
