import gmsh
import sys


def create_rectangle_and_mesh(length, height, nx, ny, filename="data/euler_mesh.msh"):
    """
    Creates a rectangle and meshes it with a structured grid of quadrilateral elements.

    This function uses Gmsh's transfinite meshing algorithm to create a structured
    quadrilateral mesh with a precise number of elements along each axis.

    Args:
        length (float): The length of the rectangle along the x-axis.
        height (float): The height of the rectangle along the y-axis.
        nx (int): The number of elements along the length (x-axis).
        ny (int): The number of elements along the height (y-axis).
        filename (str): The path to save the output .msh file.
    """
    gmsh.initialize()
    gmsh.model.add("structured_rectangle")

    # Define the corners of the rectangle
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(length, 0, 0)
    p3 = gmsh.model.geo.addPoint(length, height, 0)
    p4 = gmsh.model.geo.addPoint(0, height, 0)

    # Define the boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Create a curve loop and a surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    s = gmsh.model.geo.addPlaneSurface([cl])

    # Synchronize the geometry
    gmsh.model.geo.synchronize()

    # --- Use the Transfinite algorithm to create a structured mesh ---

    # 1. Specify the number of nodes on each boundary curve.
    # The number of nodes is the number of elements + 1.
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)

    # 2. Specify the surface for the transfinite mesh and its corners.
    # The corners must be listed in a specific order.
    gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left", [p1, p2, p3, p4])

    # 3. Instruct Gmsh to recombine triangles into quadrilaterals.
    # This is essential for creating a quad mesh.
    gmsh.model.geo.mesh.setRecombine(2, s)

    # Create physical groups for boundaries and the main surface.
    # This is important for applying boundary conditions in the solver.
    gmsh.model.addPhysicalGroup(1, [l4], name="left")
    gmsh.model.addPhysicalGroup(1, [l2], name="right")
    gmsh.model.addPhysicalGroup(1, [l1], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l3], name="top")
    gmsh.model.addPhysicalGroup(2, [s], name="fluid")

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh
    gmsh.write(filename)
    print(f"Successfully created structured mesh.")
    print(f"Mesh saved to: {filename}")

    # Finalize Gmsh
    gmsh.finalize()


if __name__ == "__main__":
    # Define rectangle dimensions and the number of elements
    rect_length = 1.0
    rect_height = 1.0
    num_elements_x = 100
    num_elements_y = 100

    print(
        f"Creating a structured mesh of size {rect_length}x{rect_height} "
        f"with {num_elements_x}x{num_elements_y} elements."
    )
    create_rectangle_and_mesh(rect_length, rect_height, num_elements_x, num_elements_y)
    print("\nScript finished.")
