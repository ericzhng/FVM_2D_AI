import gmsh
import sys
import os


def create_and_mesh_rectangle(length, height, mesh_size, filename="data/rectangle_mesh.msh"):
    """
    Creates a rectangle, meshes it with quadrilateral elements,
    creates physical groups for its sides, and extracts mesh information.

    Args:
        length (float): The length of the rectangle along the x-axis.
        height (float): The height of the rectangle along the y-axis.
        mesh_size (float): The desired element size for the mesh.
        filename (str): The path to save the output .msh file.
    """

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Initialize Gmsh
    gmsh.initialize()

    # Add a new model
    gmsh.model.add("rectangle_mesh")

    # Define geometric points
    # (x, y, z, mesh_size_at_point)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.geo.addPoint(length, 0, 0, mesh_size)
    p3 = gmsh.model.geo.addPoint(length, height, 0, mesh_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

    # Define lines connecting the points (sides of the rectangle)
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left

    # Create a curve loop from the lines
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # Create a plane surface from the curve loop
    plane_surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Synchronize the CAD model with the Gmsh model
    gmsh.model.geo.synchronize()

    # Create physical groups for the lines (sides)
    # Physical groups are used to identify specific entities (points, curves, surfaces, volumes)
    # in the mesh for post-processing or boundary condition application.
    gmsh.model.addPhysicalGroup(1, [l1], name="Bottom_Side")
    gmsh.model.addPhysicalGroup(1, [l2], name="Right_Side")
    gmsh.model.addPhysicalGroup(1, [l3], name="Top_Side")
    gmsh.model.addPhysicalGroup(1, [l4], name="Left_Side")

    # Create a physical group for the surface
    gmsh.model.addPhysicalGroup(2, [plane_surface], name="Rectangle_Surface")

    # Set the 2D meshing algorithm to generate quadrilaterals
    # 1: MeshAdapt, 2: Automatic, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Octree
    # For structured quads, we often need to specify transfinite curves/surfaces,
    # but for general quadrilateral meshing, Gmsh's algorithms can do it.
    # Here, we'll use a standard algorithm and ensure recombination for quads.
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Octree is good for quads
    gmsh.option.setNumber(
        "Mesh.RecombineAll", 1
    )  # Try to recombine triangles into quadrangles
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # 1 for all-quad

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # --- Extract and print mesh information ---

    print("\n--- Mesh Node Information ---")
    # Get all nodes in the model.
    # Returns nodeTags (list of ints), coord (list of floats, [x1,y1,z1, x2,y2,z2,...]),
    # and parametricCoord (list of floats, not always used).
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    print(f"Total number of nodes: {len(node_tags)}")
    # print("Node ID | X-coordinate | Y-coordinate | Z-coordinate")
    # print("--------------------------------------------------")
    # for i, tag in enumerate(node_tags):
    #     x = node_coords[i * 3]
    #     y = node_coords[i * 3 + 1]
    #     z = node_coords[i * 3 + 2]
    #     print(f"{tag:<7} | {x:<12.6f} | {y:<12.6f} | {z:<12.6f}")

    print("\n--- Mesh Element Information ---")
    # Get all elements in the model.
    # Returns elementTypes (list of int), elementTags (list of int), and nodeTags (list of int)
    # elementTypes: Gmsh element type (e.g., 3 for 4-node quad, 2 for 3-node triangle)
    # elementTags: Unique ID for each element
    # nodeTags: Flattened list of node tags for each element.
    #           For a quad, it's [node1_tag, node2_tag, node3_tag, node4_tag, ...]
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()

    print(f"Total number of element types found: {len(element_types)}")
    for i, elem_type in enumerate(element_types):
        # Get information about the element type (name, dimension, number of nodes)
        elem_name, elem_dim, order, num_nodes_per_elem, _, _ = (
            gmsh.model.mesh.getElementProperties(elem_type)
        )

        print(
            f"\nElement Type: {elem_name} (Type ID: {elem_type}, Dimension: {elem_dim}, Nodes per element: {num_nodes_per_elem})"
        )
        print(f"Number of elements of this type: {len(element_tags[i])}")
        print("Element ID | Node IDs")
        print("--------------------------------")

        # Iterate through elements of the current type
        current_node_idx = 0
        for j, tag in enumerate(element_tags[i]):
            # Extract node tags for the current element
            nodes = element_node_tags[i][
                current_node_idx : current_node_idx + num_nodes_per_elem
            ]
            print(f"{tag:<10} | {' '.join(map(str, nodes))}")
            current_node_idx += num_nodes_per_elem

    # Optional: Save the mesh to a .msh file for visualization
    gmsh.write(filename)
    print(f"\nMesh saved to {filename}")

    # Finalize Gmsh
    gmsh.finalize()


# --- Main execution ---
if __name__ == "__main__":
    # Define rectangle dimensions and mesh size
    rect_length = 100.0
    rect_height = 100.0
    mesh_size_val = 40

    print(
        f"Creating a rectangle of size {rect_length}x{rect_height} with mesh size {mesh_size_val}"
    )
    create_and_mesh_rectangle(rect_length, rect_height, mesh_size_val)
    print("\nScript finished.")