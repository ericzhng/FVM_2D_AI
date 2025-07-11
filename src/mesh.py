import numpy as np
import gmsh

def read_mesh(mesh_file):
    """
    Reads a mesh file using gmsh and extracts node and element information.

    Args:
        mesh_file (str): Path to the mesh file.

    Returns:
        tuple: A tuple containing:
            - node_tags (np.ndarray): Array of node tags.
            - node_coords (np.ndarray): Array of node coordinates.
            - elem_tags_quad (np.ndarray): Array of quadrilateral element tags.
            - elem_conn (np.ndarray): Array of element connectivity.
    """
    gmsh.initialize()
    gmsh.open(mesh_file)

    print("\n--- Mesh Information ---")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    print(f"Total number of nodes: {len(node_tags)}")

    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
    print(f"Total number of element types found: {len(element_types)}")
    gmsh.finalize()

    # Extract quadrilateral element data (assuming they are the second type found)
    elem_tags_quad = element_tags[1]
    elem_conn = np.array(element_node_tags[1]).reshape(-1, 4)
    node_coord = np.array(node_coords).reshape(-1, 3)
    
    return node_tags, node_coord, elem_tags_quad, elem_conn

def get_neighbors(elem_conn):
    """
    Computes neighbors for all elements.

    Args:
        elem_conn (np.ndarray): Element connectivity array.

    Returns:
        list: A list of lists, where each inner list contains the neighbors of an element.
    """
    nelem = elem_conn.shape[0]

    face_to_elem = {}
    for i, elem_nodes in enumerate(elem_conn):
        for j in range(4):
            node1 = elem_nodes[j]
            node2 = elem_nodes[(j + 1) % 4]
            face = tuple(sorted((node1, node2)))
            if face not in face_to_elem:
                face_to_elem[face] = []
            face_to_elem[face].append(i)

    all_neighbors = [[] for _ in range(nelem)]
    for i in range(nelem):
        elem_nodes = elem_conn[i]
        neighbors = []
        for j in range(4):
            node1 = elem_nodes[j]
            node2 = elem_nodes[(j + 1) % 4]
            face = tuple(sorted((node1, node2)))
            for neighbor_idx in face_to_elem[face]:
                if neighbor_idx != i:
                    neighbors.append(neighbor_idx)
        all_neighbors[i] = list(set(neighbors))
    return all_neighbors

def get_face_normal_and_length(elem_idx, neighbor_idx, elem_conn, node_coord):
    """
    Computes the normal and length of the face between two elements.

    Args:
        elem_idx (int): Index of the first element.
        neighbor_idx (int): Index of the second element.
        elem_conn (np.ndarray): Element connectivity array.
        node_coord (np.ndarray): Node coordinates array.

    Returns:
        tuple: A tuple containing:
            - normal (np.ndarray): The normal vector of the face.
            - length (float): The length of the face.
    """
    elem1_nodes = set(elem_conn[elem_idx])
    elem2_nodes = set(elem_conn[neighbor_idx])

    common_nodes_tags = list(elem1_nodes.intersection(elem2_nodes))
    if len(common_nodes_tags) != 2:
        return np.array([0, 0]), 0.0

    p1 = node_coord[common_nodes_tags[0] - 1]
    p2 = node_coord[common_nodes_tags[1] - 1]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    length = np.sqrt(dx**2 + dy**2)

    if length < 1e-9:
        return np.array([0, 0]), 0

    # Normal vector, normalized
    normal = np.array([dy / length, -dx / length])

    # Ensure the normal points from elem_idx to neighbor_idx
    centroid1 = np.mean(node_coord[elem_conn[elem_idx] - 1], axis=0)
    face_midpoint = (p1 + p2) / 2.0

    vec_to_face = face_midpoint - centroid1

    if np.dot(normal, vec_to_face[:2]) < 0:
        normal = -normal

    return normal, length

def compute_cell_areas(elem_conn, node_coord):
    """
    Computes the areas of all cells.

    Args:
        elem_conn (np.ndarray): Element connectivity array.
        node_coord (np.ndarray): Node coordinates array.

    Returns:
        np.ndarray: An array containing the area of each cell.
    """
    areas = np.zeros(len(elem_conn))
    for i, elem_nodes in enumerate(elem_conn):
        nodes = node_coord[np.array(elem_nodes) - 1]
        x = nodes[:, 0]
        y = nodes[:, 1]
        areas[i] = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return areas

def compute_cell_centroids(elem_conn, node_coord):
    """
    Computes the centroids of all cells.

    Args:
        elem_conn (np.ndarray): Element connectivity array.
        node_coord (np.ndarray): Node coordinates array.

    Returns:
        np.ndarray: An array containing the centroid of each cell.
    """
    centroids = np.zeros((len(elem_conn), 2))
    for i, elem_nodes in enumerate(elem_conn):
        nodes = node_coord[np.array(elem_nodes) - 1]
        centroids[i] = np.mean(nodes[:, :2], axis=0)
    return centroids
