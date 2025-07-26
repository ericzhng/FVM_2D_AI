import gmsh
import sys
import os


def create_and_mesh_cube(
    length, width, height, nx, ny, nz, filename="data/cube_mesh.msh"
):
    """
    Creates a cube and meshes it with a structured grid of hexahedral elements.
    """

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    gmsh.initialize()
    gmsh.model.add("structured_cube")

    # Define the corners of the cube
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(length, 0, 0)
    p3 = gmsh.model.geo.addPoint(length, width, 0)
    p4 = gmsh.model.geo.addPoint(0, width, 0)
    p5 = gmsh.model.geo.addPoint(0, 0, height)
    p6 = gmsh.model.geo.addPoint(length, 0, height)
    p7 = gmsh.model.geo.addPoint(length, width, height)
    p8 = gmsh.model.geo.addPoint(0, width, height)

    # Define the boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p8)
    l8 = gmsh.model.geo.addLine(p8, p5)
    l9 = gmsh.model.geo.addLine(p1, p5)
    l10 = gmsh.model.geo.addLine(p2, p6)
    l11 = gmsh.model.geo.addLine(p3, p7)
    l12 = gmsh.model.geo.addLine(p4, p8)

    # Create curve loops and surfaces
    cl1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])  # Bottom
    s1 = gmsh.model.geo.addPlaneSurface([cl1])

    cl2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])  # Top
    s2 = gmsh.model.geo.addPlaneSurface([cl2])

    cl3 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])  # Front
    s3 = gmsh.model.geo.addPlaneSurface([cl3])

    cl4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])  # Right
    s4 = gmsh.model.geo.addPlaneSurface([cl4])

    cl5 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])  # Back
    s5 = gmsh.model.geo.addPlaneSurface([cl5])

    cl6 = gmsh.model.geo.addCurveLoop([l4, l9, -l8, -l12])  # Left
    s6 = gmsh.model.geo.addPlaneSurface([cl6])


    # Create a surface loop and a volume
    sl = gmsh.model.geo.addSurfaceLoop([s1, s2, s3, s4, s5, s6])
    v = gmsh.model.geo.addVolume([sl])

    # Use the Transfinite algorithm for a structured mesh
    gmsh.model.geo.mesh.setTransfiniteCurve(l1, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l5, nx + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l7, nx + 1)

    gmsh.model.geo.mesh.setTransfiniteCurve(l2, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l6, ny + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l8, ny + 1)

    gmsh.model.geo.mesh.setTransfiniteCurve(l9, nz + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l10, nz + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l11, nz + 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(l12, nz + 1)

    gmsh.model.geo.mesh.setTransfiniteSurface(s1)
    gmsh.model.geo.mesh.setRecombine(2, s1)
    gmsh.model.geo.mesh.setTransfiniteSurface(s2)
    gmsh.model.geo.mesh.setRecombine(2, s2)
    gmsh.model.geo.mesh.setTransfiniteSurface(s3)
    gmsh.model.geo.mesh.setRecombine(2, s3)
    gmsh.model.geo.mesh.setTransfiniteSurface(s4)
    gmsh.model.geo.mesh.setRecombine(2, s4)
    gmsh.model.geo.mesh.setTransfiniteSurface(s5)
    gmsh.model.geo.mesh.setRecombine(2, s5)
    gmsh.model.geo.mesh.setTransfiniteSurface(s6)
    gmsh.model.geo.mesh.setRecombine(2, s6)

    gmsh.model.geo.mesh.setTransfiniteVolume(v)
    gmsh.model.mesh.setRecombine(3, v)

    gmsh.model.geo.synchronize()

    # Create physical groups
    gmsh.model.addPhysicalGroup(2, [s6], name="left")
    gmsh.model.addPhysicalGroup(2, [s4], name="right")
    gmsh.model.addPhysicalGroup(2, [s1], name="bottom")
    gmsh.model.addPhysicalGroup(2, [s2], name="top")
    gmsh.model.addPhysicalGroup(2, [s3], name="front")
    gmsh.model.addPhysicalGroup(2, [s5], name="back")
    gmsh.model.addPhysicalGroup(3, [v], name="fluid")

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh
    gmsh.write(filename)
    print(f"Successfully created structured mesh with {nx}x{ny}x{nz} elements.")
    print(f"Mesh saved to: {filename}")

    # Finalize Gmsh
    gmsh.finalize()


if __name__ == "__main__":
    # Define cube dimensions and the number of elements
    cube_length = 1.0
    cube_width = 1.0
    cube_height = 1.0
    num_elements_x = 10
    num_elements_y = 10
    num_elements_z = 10

    print(
        f"Creating a structured mesh of size {cube_length}x{cube_width}x{cube_height} "
        f"with {num_elements_x}x{num_elements_y}x{num_elements_z} elements."
    )
    create_and_mesh_cube(
        cube_length,
        cube_width,
        cube_height,
        num_elements_x,
        num_elements_y,
        num_elements_z,
        filename="data/cube_mesh.msh",
    )
    print("\nScript finished.")