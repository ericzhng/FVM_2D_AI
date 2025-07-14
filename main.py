from src.mesh import Mesh
from src.solver import solve
from src.case_setup import setup_case_shallow_water, setup_case_euler
from src.visualization import plot_simulation_step, create_animation, plot_mesh
import src.equations as equations
import numpy as np

class Equation: 
    def __init__(self, hllc_flux, calculate_adaptive_dt, apply_boundary_condition, **params):
        self.hllc_flux = lambda U_L, U_R, normal: hllc_flux(U_L, U_R, normal, **params)
        self.calculate_adaptive_dt = lambda mesh, U, cfl: calculate_adaptive_dt(mesh, U, cfl, **params)
        self.apply_boundary_condition = lambda U_inside, normal, bc_info: apply_boundary_condition(U_inside, normal, bc_info, **params)

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
    equation_type = "euler"  # Choose 'shallow_water' or 'euler'

    if equation_type == 'shallow_water':
        U_init, boundary_conditions = setup_case_shallow_water(mesh)
        equation = Equation(
            hllc_flux=equations.hllc_flux_shallow_water,
            calculate_adaptive_dt=equations.calculate_adaptive_dt_shallow_water,
            apply_boundary_condition=equations.apply_boundary_condition_shallow_water,
            g=9.81
        )
        t_end = 100.0
    elif equation_type == 'euler':
        U_init, boundary_conditions = setup_case_euler(mesh)
        equation = Equation(
            hllc_flux=equations.hllc_flux_euler,
            calculate_adaptive_dt=equations.calculate_adaptive_dt_euler,
            apply_boundary_condition=equations.apply_boundary_condition_euler,
            gamma=1.4
        )
        t_end = 10.0
    else:
        raise ValueError("Invalid equation type specified.")


    # --- 3. Solve ---
    history, dt_history = solve(
        mesh,
        U_init,
        boundary_conditions,
        equation,
        t_end=t_end,
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
