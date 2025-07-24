import gmsh
import numpy as np
import os


def polygon_geo_to_triangular_mesh(
    node_coords,
    output_filename,
    mesh_size=0.1,
    mesh_algorithm="meshadapt",
):
    """
    Generates a triangular mesh for a polygon, with physical groups for the
    surface and boundaries, and more direct control over the mesh size.

    Args:
        node_coords (list of tuples): A list of (x, y, z) coordinates for the polygon vertices.
        output_filename (str): The path to save the output .msh file.
        mesh_size (float, optional): The desired edge length of the mesh elements.
                                     Defaults to 0.1.
        mesh_algorithm (str, optional): The 2D mesh algorithm to use.
                                        Options: "meshadapt", "del2d", "front2d".
                                        Defaults to "meshadapt".
    """
    gmsh.initialize()
    gmsh.model.add("polygon_mesh")

    # --- Geometry Definition ---
    points = []
    for x, y, z in node_coords:
        points.append(gmsh.model.geo.addPoint(x, y, z))

    lines = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        lines.append(gmsh.model.geo.addLine(p1, p2))

    curve_loop = gmsh.model.geo.addCurveLoop(lines)
    plane_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()

    # --- Physical Groups ---
    # This is crucial for applying boundary conditions in a solver.
    gmsh.model.addPhysicalGroup(2, [plane_surface], name="fluid")
    gmsh.model.addPhysicalGroup(1, lines, name="walls")

    # --- Mesh Size Control ---
    # By setting Min and Max to the same value, we enforce a more uniform mesh size.
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # --- Meshing Algorithm Selection ---
    algo_map = {"meshadapt": 1, "del2d": 5, "front2d": 6}
    gmsh.option.setNumber("Mesh.Algorithm", algo_map.get(mesh_algorithm.lower(), 1))

    # --- Generate and Save Mesh ---
    gmsh.model.mesh.generate(2)

    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gmsh.write(output_filename)
    print(
        f"Successfully generated mesh with algorithm '{mesh_algorithm}' and saved to {output_filename}"
    )

    gmsh.finalize()


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Define the vertices of a polygon (e.g., a more complex shape)
    polygon_vertices = [
        (0, 0, 0),
        (3, 0, 0),
        (3, 1, 0),
        (1, 1, 0),
        (1, 2, 0),
        (2, 2, 0),
        (2, 3, 0),
        (0, 3, 0),
    ]

    # 2. Define the output path
    output_file = "./data/complex_shape_mesh.msh"

    # 3. Set mesh parameters
    # This will now create a mesh with edge lengths very close to 0.1.
    target_edge_length = 1
    selected_algorithm = "meshadapt"

    # 4. Generate the mesh
    try:
        polygon_geo_to_triangular_mesh(
            polygon_vertices,
            output_file,
            mesh_size=target_edge_length,
            mesh_algorithm=selected_algorithm,
        )
        print("\nPhysical groups for 'fluid' and 'walls' have been added.")
        print("To visualize the mesh, you can open the .msh file in Gmsh.")

    except Exception as e:
        print(f"An error occurred: {e}")
