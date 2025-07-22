import numpy as np
from src.mesh import Mesh
from src.time_step import calculate_adaptive_dt
from src.reconstruction import reconst_func
from src.visualization import plot_simulation_step


def solve(
    mesh: Mesh,
    U,
    boundary_conditions,
    equation,
    t_end,
    limiter_type="barth_jespersen",
    flux_type="roe",
    over_relaxation=1.2,
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
    variable_to_plot=0,
):
    """
    Main solver loop for the Finite Volume Method.

    This function orchestrates the time-stepping process, including gradient
    computation, slope limiting, MUSCL reconstruction, and flux calculation.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The initial conservative state vector.
        boundary_conditions (dict): A dictionary defining the boundary conditions.
        equation (BaseEquation): The equation system to be solved.
        t_end (float, optional): The end time of the simulation. Defaults to 2.0.
        over_relaxation (float, optional): Over-relaxation factor. Defaults to 1.2.
        limiter (str, optional): The type of slope limiter. Defaults to "barth_jespersen".
        use_adaptive_dt (bool, optional): Whether to use adaptive time stepping.
                                        Defaults to True.
        cfl (float, optional): The CFL number for adaptive time stepping. Defaults to 0.5.
        dt_initial (float, optional): The initial time step. Defaults to 0.01.

    Returns:
        tuple: A tuple containing the history of the state vector and the history
               of the time steps.
    """
    history = [U.copy()]
    t = 0.0
    n = 0
    dt_history = []
    dt = dt_initial
    time_integration_method = "rk2"

    plot_simulation_step(
        mesh, equation.cons_to_prim_batch(U), f"t={t:.4f}", variable_to_plot
    )

    while t < t_end:
        if use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, cfl)
            # Compute time step
            dt = min(dt, t_end - t)

        t += dt
        n += 1

        # chosen between RK2 and 1st order update
        # RK2 method
        if time_integration_method == "rk2":
            # RK2 1st step
            U_star = U - dt * reconst_func(
                mesh,
                U,
                equation,
                boundary_conditions,
                limiter_type,
                flux_type,
                over_relaxation,
            )

            # RK2 2nd step / update U
            U_new = (
                U
                + U_star
                - dt
                * reconst_func(
                    mesh,
                    U,
                    equation,
                    boundary_conditions,
                    limiter_type,
                    flux_type,
                    over_relaxation,
                )
            ) / 2.0

        elif time_integration_method == "euler":  # 1st order update
            # 1st order update
            U_new = U - dt * reconst_func(
                mesh,
                U,
                equation,
                boundary_conditions,
                limiter_type,
                flux_type,
                over_relaxation,
            )

        # raise error if not the above
        else:
            raise NotImplementedError(
                f"Time integration method '{time_integration_method}' is not supported."
            )

        U = U_new
        t += dt
        history.append(U.copy())
        dt_history.append(dt)
        print(f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s")

    return history, dt_history
