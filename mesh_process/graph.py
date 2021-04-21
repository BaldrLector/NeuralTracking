import os
import shutil
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh

from utils import utils, image_proc

from NeuralNRT._C import compute_mesh_from_depth_and_flow as compute_mesh_from_depth_and_flow_c
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import node_and_edge_clean_up as node_and_edge_clean_up_c
from NeuralNRT._C import compute_pixel_anchors_geodesic as compute_pixel_anchors_geodesic_c
from NeuralNRT._C import compute_pixel_anchors_euclidean as compute_pixel_anchors_euclidean_c
from NeuralNRT._C import compute_clusters as compute_clusters_c
from NeuralNRT._C import update_pixel_anchors as update_pixel_anchors_c

#########################################################################
# Options
#########################################################################
# Depth-to-mesh conversion
DEPTH_NORMALIZER = 1000.0
MAX_TRIANGLE_DISTANCE = 0.05

# Erosion of vertices in the boundaries
EROSION_NUM_ITERATIONS = 10
EROSION_MIN_NEIGHBORS = 4

# Node sampling and edges computation
NODE_COVERAGE = 0.05  # in meters
USE_ONLY_VALID_VERTICES = True
NUM_NEIGHBORS = 8
ENFORCE_TOTAL_NUM_NEIGHBORS = False
SAMPLE_RANDOM_SHUFFLE = False

# Pixel anchors
NEIGHBORHOOD_DEPTH = 2

MIN_CLUSTER_SIZE = 3
MIN_NUM_NEIGHBORS = 2

# Node clean-up
REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True


def build_graph(vertices, faces, ):
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    assert num_vertices > 0 and num_faces > 0

    # Erode mesh, to not sample unstable nodes on the mesh boundary.
    non_eroded_vertices = erode_mesh_c(
        vertices, faces, EROSION_NUM_ITERATIONS, EROSION_MIN_NEIGHBORS
    )

    #########################################################################
    # Sample graph nodes.
    #########################################################################
    valid_vertices = non_eroded_vertices

    # Sample graph nodes.
    node_coords = np.zeros((0), dtype=np.float32)
    node_indices = np.zeros((0), dtype=np.int32)

    num_nodes = sample_nodes_c(
        vertices, valid_vertices,
        node_coords, node_indices,
        NODE_COVERAGE,
        USE_ONLY_VALID_VERTICES,
        SAMPLE_RANDOM_SHUFFLE
    )

    node_coords = node_coords[:num_nodes, :]
    node_indices = node_indices[:num_nodes, :]

    #########################################################################
    # Compute graph edges.
    #########################################################################
    # Compute edges between nodes.
    graph_edges = -np.ones((num_nodes, NUM_NEIGHBORS), dtype=np.int32)
    graph_edges_weights = np.zeros(
        (num_nodes, NUM_NEIGHBORS), dtype=np.float32)
    graph_edges_distances = np.zeros(
        (num_nodes, NUM_NEIGHBORS), dtype=np.float32)
    node_to_vertex_distances = - \
        np.ones((num_nodes, num_vertices), dtype=np.float32)

    visible_vertices = np.ones_like(valid_vertices)

    compute_edges_geodesic_c(
        vertices, visible_vertices, faces, node_indices,
        NUM_NEIGHBORS, NODE_COVERAGE,
        graph_edges, graph_edges_weights, graph_edges_distances,
        node_to_vertex_distances,
        USE_ONLY_VALID_VERTICES,
        ENFORCE_TOTAL_NUM_NEIGHBORS
    )

    # Remove nodes
    valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
    node_id_black_list = []

    if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
        # Mark nodes with not enough neighbors
        node_and_edge_clean_up_c(graph_edges, valid_nodes_mask)

        # Get the list of invalid nodes
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()
    else:
        print("You're allowing nodes with not enough neighbors!")

    print("Node filtering: initial num nodes", num_nodes, "| invalid nodes", len(
        node_id_black_list), "({})".format(node_id_black_list))

    # Get only valid nodes and their corresponding info
    node_coords = node_coords[valid_nodes_mask.squeeze()]
    node_indices = node_indices[valid_nodes_mask.squeeze()]
    graph_edges = graph_edges[valid_nodes_mask.squeeze()]
    graph_edges_weights = graph_edges_weights[valid_nodes_mask.squeeze()]
    graph_edges_distances = graph_edges_distances[valid_nodes_mask.squeeze()]

    #########################################################################
    # Graph checks.
    #########################################################################
    num_nodes = node_coords.shape[0]

    # Update node ids only if we actually removed nodes
    if len(node_id_black_list) > 0:
        # 1. Mapping old indices to new indices
        count = 0
        node_id_mapping = {}
        for i, is_node_valid in enumerate(valid_nodes_mask):
            if not is_node_valid:
                node_id_mapping[i] = -1
            else:
                node_id_mapping[i] = count
                count += 1

        # 2. Update graph_edges using the id mapping
        for node_id, graph_edge in enumerate(graph_edges):
            # compute mask of valid neighbors
            valid_neighboring_nodes = np.invert(
                np.isin(graph_edge, node_id_black_list))

            # make a copy of the current neighbors' ids
            graph_edge_copy = np.copy(graph_edge)
            graph_edge_weights_copy = np.copy(graph_edges_weights[node_id])
            graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

            # set the neighbors' ids to -1
            graph_edges[node_id] = -np.ones_like(graph_edge_copy)
            graph_edges_weights[node_id] = np.zeros_like(
                graph_edge_weights_copy)
            graph_edges_distances[node_id] = np.zeros_like(
                graph_edge_distances_copy)

            count_valid_neighbors = 0
            for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                if is_valid_neighbor:
                    # current neighbor id
                    current_neighbor_id = graph_edge_copy[neighbor_idx]

                    # get mapped neighbor id
                    if current_neighbor_id == -1:
                        mapped_neighbor_id = -1
                    else:
                        mapped_neighbor_id = node_id_mapping[current_neighbor_id]

                    graph_edges[node_id,
                                count_valid_neighbors] = mapped_neighbor_id
                    graph_edges_weights[node_id,
                                        count_valid_neighbors] = graph_edge_weights_copy[neighbor_idx]
                    graph_edges_distances[node_id,
                                          count_valid_neighbors] = graph_edge_distances_copy[neighbor_idx]

                    count_valid_neighbors += 1

            # normalize edges' weights
            sum_weights = np.sum(graph_edges_weights[node_id])
            if sum_weights > 0:
                graph_edges_weights[node_id] /= sum_weights
            else:
                print("Hmmmmm", graph_edges_weights[node_id])
                raise Exception("Not good")

    #########################################################################
    # Compute clusters.
    #########################################################################
    graph_clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32)
    clusters_size_list = compute_clusters_c(graph_edges, graph_clusters)
    for i, cluster_size in enumerate(clusters_size_list):
        if cluster_size <= 2:
            print("Cluster is too small {}".format(clusters_size_list))
            print("It only has nodes:", np.where(graph_clusters == i)[0])

    return node_coords, node_indices, graph_edges, graph_edges_weights, graph_edges_distances, graph_clusters


def compute_pixel_achors_weigths_cpu(graphNodes, pointImage, nodeCoverage):
    pixel_anchors = np.zeros((0), dtype=np.int32)
    pixel_weights = np.zeros((0), dtype=np.float32)
    compute_pixel_anchors_euclidean_c(
        graphNodes, pointImage, nodeCoverage, pixel_anchors, pixel_weights)
    return pixel_anchors, pixel_weights
