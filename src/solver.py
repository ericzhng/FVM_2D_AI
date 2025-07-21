import numpy as np
from src.mesh import Mesh
from src.time_step import calculate_adaptive_dt
from src.reconstruction import reconst_func


def solve(
    mesh: Mesh,
    U,
    boundary_conditions,
    equation,
    t_end,
    over_relaxation=1.2,
    limiter="barth_jespersen",
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
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

    n_ghost = 1

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
            U_star = U - dt * reconst_func(mesh, U, equation, boundary_conditions, dt)
            bc_obj.enforce_bc(U_star)

            # RK2 2nd step / update U
            U_new = (U + U_star - dt * reconst_func(mesh, U, equation, boundary_conditions, dt)) / 2.0
            bc_obj.enforce_bc(U_new)

        elif time_integration_method == "euler":  # 1st order update
            # 1st order update
            U_new = U - dt * reconst_func(mesh, U, equation, boundary_conditions, dt)
            bc_obj.enforce_bc(U_new)

        # raise error if not the above
        else:
            raise NotImplementedError(
                f"Time integration method '{time_integration_method}' is not supported."
            )

        U_new = U.copy()

        # --- Flux Integration Loop ---
        for i in range(mesh.nelem):
            for j, neighbor_idx in enumerate(mesh.cell_neighbors[i]):
                face_normal = mesh.face_normals[i, j, :2]
                face_area = mesh.face_areas[i, j]

                if neighbor_idx != -1:
                    # --- Interior Face ---
                    face_nodes_tags = mesh.elem_faces[i][j]
                    node_coords = [
                        mesh.node_coords[np.where(mesh.node_tags == tag)[0][0]]
                        for tag in face_nodes_tags
                    ]
                    face_midpoint = np.mean(node_coords, axis=0)

                    U_L, U_R = muscl_reconstruction(
                        U[i],
                        U[neighbor_idx],
                        gradients[i],
                        gradients[neighbor_idx],
                        limiters[i],
                        limiters[neighbor_idx],
                        mesh.cell_centroids[i],
                        mesh.cell_centroids[neighbor_idx],
                        face_midpoint,
                    )
                    flux = equation.hllc_flux(U_L, U_R, face_normal)
                else:
                    # --- Boundary Face ---
                    face_tuple = tuple(sorted(mesh.elem_faces[i][j]))
                    bc_name = mesh.boundary_faces.get(face_tuple, {}).get(
                        "name", "wall"
                    )
                    bc_info = boundary_conditions.get(bc_name, {"type": "wall"})

                    # Get ghost cell state based on boundary condition
                    U_ghost = equation.apply_boundary_condition(
                        U[i], face_normal, bc_info
                    )

                    # Flux is computed between the interior cell and the ghost cell
                    flux = equation.hllc_flux(U[i], U_ghost, face_normal)

                # Update the solution
                U_new[i] -= (dt / mesh.cell_volumes[i]) * flux * face_area

        U = U_new
        t += dt
        history.append(U.copy())
        dt_history.append(dt)
        print(f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s")

    return history, dt_history
