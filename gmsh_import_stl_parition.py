import gmsh
import sys
import os


def remesh_stl_with_partition(
    stl_path,
    partition_start_point,
    partition_end_point,
    mesh_size,
    output_msh_path="remeshed_model_with_partition.msh",
):
    """
    Loads an STL file (assumed to be a 2D model/surface), embeds a partition line,
    remeshes it with quadrilateral elements, and saves the new mesh.

    Args:
        stl_path (str): Path to the input STL file.
        partition_start_point (tuple): (x, y, z) coordinates of the partition line start.
        partition_end_point (tuple): (x, y, z) coordinates of the partition line end.
        mesh_size (float): The desired element size for the remeshed model.
        output_msh_path (str): Path to save the remeshed .msh file.
    """

    gmsh.initialize()
    gmsh.model.add("remesh_stl_with_partition")

    print(f"Loading STL file: {stl_path}")

    # --- Corrected STL Import and Geometry Reconstruction ---
    # 1. Merge the STL file into the current Gmsh model.
    # This loads the mesh data from the STL.
    try:
        gmsh.merge(stl_path)
        print("Successfully merged STL mesh.")
    except Exception as e:
        print(f"Error merging STL file: {e}")
        gmsh.finalize()
        return

    # # Optional: Coalesce nodes and remove duplicate elements to improve geometry reconstruction robustness
    # # This helps if there are tiny gaps or duplicate nodes/elements in the STL
    # gmsh.model.mesh.removeDuplicateNodes()
    # gmsh.model.mesh.removeDuplicateElements()
    # print("Removed duplicate mesh nodes and elements (if any).")

    # 2. Create geometry (B-Rep entities like surfaces, curves, points) from the imported mesh.
    # This will likely create 'Discrete surface' entities if a clean OCC B-Rep cannot be formed.
    gmsh.model.mesh.createGeometry()
    gmsh.model.occ.synchronize()  # Synchronize OCC kernel after geometry creation
    print("Geometry reconstructed from STL mesh.")

    # IMPORTANT: Clear the existing mesh data. This is crucial to ensure Gmsh
    # re-meshes the reconstructed geometry rather than using the original STL mesh.
    gmsh.model.mesh.clear()
    print(
        "Cleared existing mesh data to prepare for remeshing the reconstructed geometry."
    )
    # --- End of Corrected STL Import ---

    # Find the surface(s) created from the STL.
    # These will likely be 'Discrete surface' entities.
    all_surfaces = gmsh.model.getEntities(2)
    if not all_surfaces:
        print(
            "Error: No 2D surfaces found after importing STL and creating geometry. Ensure it's a valid 2D STL."
        )
        gmsh.finalize()
        return

    # Extract just the tags of the surfaces
    surface_tags = [s[1] for s in all_surfaces]
    print(
        f"Found {len(surface_tags)} entities of dimension 2 after geometry reconstruction."
    )

    # --- Add Physical Group for the reconstructed surface(s) ---
    # This is crucial for telling Gmsh to mesh these surfaces.
    gmsh.model.addPhysicalGroup(2, surface_tags, name="Remeshed_Surface")
    print(
        f"Created physical group 'Remeshed_Surface' for the reconstructed 2D surfaces."
    )

    # Define the partition line using OCC points and lines
    p_start_tag = gmsh.model.occ.addPoint(*partition_start_point, mesh_size)
    p_end_tag = gmsh.model.occ.addPoint(*partition_end_point, mesh_size)
    partition_line_tag = gmsh.model.occ.addLine(p_start_tag, p_end_tag)
    gmsh.model.occ.synchronize()

    print(
        f"Partition line defined from {partition_start_point} to {partition_end_point}"
    )

    # --- New approach: Embed the line into the discrete surface(s) ---
    print("Embedding partition line into discrete surfaces...")
    for dim, tag in all_surfaces:
        # We only embed into surfaces (dimension 2)
        if dim == 2:
            try:
                # gmsh.model.mesh.embed(dim_entity_to_embed, tag_entity_to_embed,
                #                       dim_entity_to_embed_into, tag_entity_to_embed_into)
                gmsh.model.mesh.embed(1, partition_line_tag, 2, tag)
                print(
                    f"Successfully embedded line {partition_line_tag} into surface ({dim}, {tag})."
                )
            except Exception as e:
                print(
                    f"Warning: Could not embed line {partition_line_tag} into surface ({dim}, {tag}): {e}"
                )
                print(
                    "This might happen if the line does not intersect the surface or if the surface is not suitable for embedding."
                )

    # Synchronize after embedding operations
    gmsh.model.occ.synchronize()

    # --- Add Physical Group for the partition line ---
    # This is important for identifying the line in the mesh for boundary conditions.
    gmsh.model.addPhysicalGroup(1, [partition_line_tag], name="Partition_Line")
    print(f"Created physical group 'Partition_Line' for the embedded partition line.")

    # Set meshing options for quadrilateral elements
    gmsh.option.setNumber(
        "Mesh.Algorithm", 8
    )  # Frontal-Octree is generally good for quads
    gmsh.option.setNumber(
        "Mesh.RecombineAll", 1
    )  # Instructs Gmsh to try and recombine triangles into quadrangles
    gmsh.option.setNumber(
        "Mesh.SubdivisionAlgorithm", 1
    )  # Algorithm 1 for all-quad mesh generation

    # Generate the 2D mesh on all surfaces
    print("Generating 2D mesh...")
    gmsh.model.mesh.generate(2)
    print("Mesh generation complete.")

    # --- Extract and print mesh information (similar to previous script) ---
    print("\n--- Mesh Node Information ---")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()

    print(f"Total number of nodes: {len(node_tags)}")
    print("Node ID | X-coordinate | Y-coordinate | Z-coordinate")
    print("--------------------------------------------------")
    for i, tag in enumerate(node_tags):
        x = node_coords[i * 3]
        y = node_coords[i * 3 + 1]
        z = node_coords[i * 3 + 2]
        print(f"{tag:<7} | {x:<12.6f} | {y:<12.6f} | {z:<12.6f}")

    print("\n--- Mesh Element Information ---")
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()

    print(f"Total number of element types found: {len(element_types)}")
    for i, elem_type in enumerate(element_types):
        elem_name, elem_dim, order, num_nodes_per_elem, _, _ = (
            gmsh.model.mesh.getElementProperties(elem_type)
        )

        print(
            f"\nElement Type: {elem_name} (Type ID: {elem_type}, Dimension: {elem_dim}, Nodes per element: {num_nodes_per_elem})"
        )
        print(f"Number of elements of this type: {len(element_tags[i])}")
        print("Element ID | Node IDs")
        print("--------------------------------")

        current_node_idx = 0
        for j, tag in enumerate(element_tags[i]):
            nodes = element_node_tags[i][
                current_node_idx : current_node_idx + num_nodes_per_elem
            ]
            print(f"{tag:<10} | {' '.join(map(str, nodes))}")
            current_node_idx += num_nodes_per_elem

    # Save the remeshed model to a .msh file
    gmsh.write(output_msh_path)
    print(f"\nRemeshed model saved to {output_msh_path}")

    gmsh.finalize()


# --- Example Usage ---
if __name__ == "__main__":
    # --- Create a dummy STL file for demonstration ---
    # In a real scenario, you would use your actual 2D model's STL file.
    # This dummy STL creates a simple square surface (1x1 unit in XY plane).
    dummy_stl_filename = "dummy_square.stl"
    try:
        with open(dummy_stl_filename, "w") as f:
            f.write("solid dummy_square\n")
            # Triangle 1 (bottom-left to top-right diagonal)
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 0 0 0\n")
            f.write("      vertex 1 0 0\n")
            f.write("      vertex 0 1 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            # Triangle 2 (top-right to bottom-left diagonal)
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 1 0 0\n")
            f.write("      vertex 1 1 0\n")
            f.write("      vertex 0 1 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            f.write("endsolid dummy_square\n")
        print(f"Created dummy STL file: {dummy_stl_filename} for demonstration.")
        test_stl_path = dummy_stl_filename
    except Exception as e:
        print(f"Could not create dummy STL file: {e}")
        print(
            "Please ensure you have a 'dummy_square.stl' file or replace 'test_stl_path' with your actual STL path."
        )
        # Fallback if dummy creation fails, user must provide their own STL
        test_stl_path = "your_actual_2d_model.stl"

    # Define the start and end points of the partition line.
    # For the dummy square (0,0) to (1,1), a line across the middle would be:
    partition_start = (0.0, 0.5, 0.0)
    partition_end = (1.0, 0.5, 0.0)

    # Define the desired mesh size for the remeshed model
    remesh_size = 0.1

    print(
        f"\nAttempting to remesh '{test_stl_path}' with a partition line from {partition_start} to {partition_end}"
    )
    remesh_stl_with_partition(
        test_stl_path, partition_start, partition_end, remesh_size
    )
    print(
        "\nRemeshing process completed. Check 'remeshed_model_with_partition.msh' and console output."
    )

    # Optional: Clean up the dummy STL file
    # if os.path.exists(dummy_stl_filename):
    #     os.remove(dummy_stl_filename)
    #     print(f"Removed dummy STL file: {dummy_stl_filename}")
