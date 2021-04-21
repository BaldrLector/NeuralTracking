import sys
import os
import json
from numpy.core.fromnumeric import shape
from torch._C import dtype
from torch.utils.data import Dataset
import torch
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import math
from utils import image_proc
from timeit import default_timer as timer
import random
import scipy
import torchvision.transforms.functional as TF
from utils.utils import load_flow, load_graph_nodes, load_graph_edges, load_graph_edges_weights, load_graph_node_deformations, \
    load_graph_clusters, load_int_image, load_float_image
from utils import image_proc
from NeuralNRT._C import compute_pixel_anchors_geodesic as compute_pixel_anchors_geodesic_c
from NeuralNRT._C import compute_pixel_anchors_euclidean as compute_pixel_anchors_euclidean_c
from NeuralNRT._C import compute_mesh_from_depth as compute_mesh_from_depth_c
from NeuralNRT._C import compute_mesh_from_depth_and_color as compute_mesh_from_depth_and_color_c
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import compute_edges_euclidean as compute_edges_euclidean_c
from NeuralNRT._C import construct_regular_graph as construct_regular_graph_c

from utils import utils

import open3d as o3d
import numba
import cv2


class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size

    def __call__(self, img):
        if len(img.shape) == 2:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2]
        else:
            return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2, :]


class DeformDataset(Dataset):
    def __init__(
            self,
            dataset_base_dir, data_version,
            input_width, input_height, max_boundary_dist
    ):
        self.dataset_base_dir = dataset_base_dir
        self.data_version_json = os.path.join(
            self.dataset_base_dir, data_version + ".json")

        self.input_width = input_width
        self.input_height = input_height

        self.max_boundary_dist = max_boundary_dist

        self.cropper = None

        self._load()

    def _load(self):
        with open(self.data_version_json) as f:
            self.labels = json.loads(f.read())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.labels[index]

        src_color_image_path = os.path.join(
            self.dataset_base_dir, data["source_color"])
        src_depth_image_path = os.path.join(
            self.dataset_base_dir, data["source_depth"])
        tgt_color_image_path = os.path.join(
            self.dataset_base_dir, data["target_color"])
        tgt_depth_image_path = os.path.join(
            self.dataset_base_dir, data["target_depth"])
        graph_nodes_path = os.path.join(
            self.dataset_base_dir, data["graph_nodes"])
        graph_edges_path = os.path.join(
            self.dataset_base_dir, data["graph_edges"])
        graph_edges_weights_path = os.path.join(
            self.dataset_base_dir, data["graph_edges_weights"])
        graph_node_deformations_path = os.path.join(
            self.dataset_base_dir, data["graph_node_deformations"])
        graph_clusters_path = os.path.join(
            self.dataset_base_dir, data["graph_clusters"])
        pixel_anchors_path = os.path.join(
            self.dataset_base_dir, data["pixel_anchors"])
        pixel_weights_path = os.path.join(
            self.dataset_base_dir, data["pixel_weights"])
        optical_flow_image_path = os.path.join(
            self.dataset_base_dir, data["optical_flow"])
        scene_flow_image_path = os.path.join(
            self.dataset_base_dir, data["scene_flow"])

        # Load source, target image and flow.
        source, _, cropper = DeformDataset.load_image(
            src_color_image_path, src_depth_image_path, data[
                "intrinsics"], self.input_height, self.input_width
        )
        target, target_boundary_mask, _ = DeformDataset.load_image(
            tgt_color_image_path, tgt_depth_image_path, data[
                "intrinsics"], self.input_height, self.input_width, cropper=cropper,
            max_boundary_dist=self.max_boundary_dist, compute_boundary_mask=True
        )

        optical_flow_gt, optical_flow_mask, scene_flow_gt, scene_flow_mask = DeformDataset.load_flow(
            optical_flow_image_path, scene_flow_image_path, cropper
        )

        # Load/compute graph.
        graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters, pixel_anchors, pixel_weights = DeformDataset.load_graph_data(
            graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path,
            graph_clusters_path, pixel_anchors_path, pixel_weights_path, cropper
        )

        # Compute groundtruth transformation for graph nodes.
        num_nodes = graph_nodes.shape[0]

        # Check that flow mask is valid for at least one pixel.
        assert np.sum(
            optical_flow_mask) > 0, "Zero flow mask for sample: " + json.dumps(data)

        # Store intrinsics.
        fx = data["intrinsics"]["fx"]
        fy = data["intrinsics"]["fy"]
        cx = data["intrinsics"]["cx"]
        cy = data["intrinsics"]["cy"]

        fx, fy, cx, cy = image_proc.modify_intrinsics_due_to_cropping(
            fx, fy, cx, cy, self.input_height, self.input_width, original_h=480, original_w=640
        )

        intrinsics = np.zeros((4), dtype=np.float32)
        intrinsics[0] = fx
        intrinsics[1] = fy
        intrinsics[2] = cx
        intrinsics[3] = cy

        return {
            "source": source,
            "target": target,
            "target_boundary_mask": target_boundary_mask,
            "optical_flow_gt": optical_flow_gt,
            "optical_flow_mask": optical_flow_mask,
            "scene_flow_gt": scene_flow_gt,
            "scene_flow_mask": scene_flow_mask,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "graph_edges_weights": graph_edges_weights,
            "graph_node_deformations": graph_node_deformations,
            "graph_clusters": graph_clusters,
            "pixel_anchors": pixel_anchors,
            "pixel_weights": pixel_weights,
            "num_nodes": np.array(num_nodes, dtype=np.int64),
            "intrinsics": intrinsics,
            "index": np.array(index, dtype=np.int32)
        }

    def get_metadata(self, index):
        return self.labels[index]

    @staticmethod
    def backproject_depth(depth_image, fx, fy, cx, cy, normalizer=1000.0):
        return image_proc.backproject_depth(depth_image, fx, fy, cx, cy, normalizer=1000.0)

    @staticmethod
    def load_image(
            color_image_path, depth_image_path,
            intrinsics, input_height, input_width, cropper=None,
            max_boundary_dist=0.1, compute_boundary_mask=False
    ):
        # Load images.
        color_image = io.imread(color_image_path)  # (h, w, 3)
        depth_image = io.imread(depth_image_path)  # (h, w)

        # Backproject depth image.
        depth_image = image_proc.backproject_depth(
            depth_image, intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"])  # (3, h, w)
        depth_image = depth_image.astype(np.float32)
        depth_image = np.moveaxis(depth_image, 0, -1)  # (h, w, 3)

        image_size = color_image.shape[:2]

        # Crop, since we need it to be divisible by 64
        if cropper is None:
            cropper = StaticCenterCrop(image_size, (input_height, input_width))

        color_image = cropper(color_image)
        depth_image = cropper(depth_image)

        # Construct the final image.
        image = np.zeros((6, input_height, input_width), dtype=np.float32)

        image[:3, :, :] = np.moveaxis(
            color_image, -1, 0) / 255.0       # (3, h, w)
        assert np.max(image[:3, :, :]) <= 1.0, np.max(image[:3, :, :])
        image[3:, :, :] = np.moveaxis(
            depth_image, -1, 0)               # (3, h, w)

        if not compute_boundary_mask:
            return image, None, cropper
        else:
            assert max_boundary_dist
            boundary_mask = image_proc.compute_boundary_mask(
                depth_image, max_boundary_dist)
            return image, boundary_mask, cropper

    @staticmethod
    def load_flow(optical_flow_image_path, scene_flow_image_path, cropper):
        # Load flow images.
        optical_flow_image = load_flow(optical_flow_image_path)  # (2, h, w)
        scene_flow_image = load_flow(scene_flow_image_path)   # (3, h, w)

        # Temporarily move axis for cropping
        optical_flow_image = np.moveaxis(
            optical_flow_image, 0, -1)  # (h, w, 2)
        scene_flow_image = np.moveaxis(scene_flow_image, 0, -1)   # (h, w, 3)

        # Crop for dimensions to be divisible by 64
        optical_flow_image = cropper(optical_flow_image)
        scene_flow_image = cropper(scene_flow_image)

        # Compute flow mask.
        # (h, w, 2)
        optical_flow_mask = np.isfinite(optical_flow_image)
        optical_flow_mask = np.logical_and(
            optical_flow_mask[..., 0], optical_flow_mask[..., 1])  # (h, w)
        # (h, w, 1)
        optical_flow_mask = optical_flow_mask[..., np.newaxis]
        optical_flow_mask = np.repeat(
            optical_flow_mask, 2, axis=2)                              # (h, w, 2)

        # (h, w, 3)
        scene_flow_mask = np.isfinite(scene_flow_image)
        scene_flow_mask = np.logical_and(
            scene_flow_mask[..., 0], scene_flow_mask[..., 1], scene_flow_mask[..., 2])  # (h, w)
        # (h, w, 1)
        scene_flow_mask = scene_flow_mask[..., np.newaxis]
        # (h, w, 3)
        scene_flow_mask = np.repeat(scene_flow_mask, 3, axis=2)

        # set invalid pixels to zero in the flow image
        optical_flow_image[optical_flow_mask == False] = 0.0
        scene_flow_image[scene_flow_mask == False] = 0.0

        # put channels back in first axis
        optical_flow_image = np.moveaxis(
            optical_flow_image, -1, 0).astype(np.float32)  # (2, h, w)
        optical_flow_mask = np.moveaxis(
            optical_flow_mask, -1, 0).astype(np.int64)    # (2, h, w)

        scene_flow_image = np.moveaxis(
            scene_flow_image, -1, 0).astype(np.float32)  # (3, h, w)
        scene_flow_mask = np.moveaxis(
            scene_flow_mask, -1, 0).astype(np.int64)    # (3, h, w)

        return optical_flow_image, optical_flow_mask, scene_flow_image, scene_flow_mask

    @staticmethod
    def load_graph_data(
            graph_nodes_path, graph_edges_path, graph_edges_weights_path, graph_node_deformations_path, graph_clusters_path,
            pixel_anchors_path, pixel_weights_path, cropper
    ):
        # Load data.
        graph_nodes = load_graph_nodes(graph_nodes_path)
        graph_edges = load_graph_edges(graph_edges_path)
        graph_edges_weights = load_graph_edges_weights(
            graph_edges_weights_path)
        graph_node_deformations = load_graph_node_deformations(
            graph_node_deformations_path) if graph_node_deformations_path is not None else None
        graph_clusters = load_graph_clusters(graph_clusters_path)
        pixel_anchors = cropper(load_int_image(pixel_anchors_path))
        pixel_weights = cropper(load_float_image(pixel_weights_path))

        assert np.isfinite(graph_edges_weights).all(), graph_edges_weights
        assert np.isfinite(pixel_weights).all(),       pixel_weights

        if graph_node_deformations is not None:
            assert np.isfinite(
                graph_node_deformations).all(), graph_node_deformations
            assert graph_node_deformations.shape[1] == 3
            assert graph_node_deformations.dtype == np.float32

        return graph_nodes, graph_edges, graph_edges_weights, graph_node_deformations, graph_clusters, pixel_anchors, pixel_weights

    @staticmethod
    def collate_with_padding(batch):
        batch_size = len(batch)

        # Compute max number of nodes.
        item_keys = 0
        max_num_nodes = 0
        for sample_idx in range(batch_size):
            item_keys = batch[sample_idx].keys()
            num_nodes = batch[sample_idx]["num_nodes"]
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes

        # Convert merged parts into torch tensors.
        # We pad graph nodes, edges and deformation ground truth with zeros.
        batch_converted = {}

        for key in item_keys:
            if key == "graph_nodes" or key == "graph_edges" or \
                    key == "graph_edges_weights" or key == "graph_node_deformations" or \
                    key == "graph_clusters":

                batched_sample = torch.zeros(
                    (batch_size, max_num_nodes, batch[0][key].shape[1]), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx, :batch[sample_idx][key].shape[0], :] = torch.from_numpy(
                        batch[sample_idx][key])
                batch_converted[key] = batched_sample

            else:
                batched_sample = torch.zeros(
                    (batch_size, *batch[0][key].shape), dtype=torch.from_numpy(batch[0][key]).dtype)
                for sample_idx in range(batch_size):
                    batched_sample[sample_idx] = torch.from_numpy(
                        batch[sample_idx][key])
                batch_converted[key] = batched_sample

        return [
            batch_converted["source"],
            batch_converted["target"],
            batch_converted["target_boundary_mask"],
            batch_converted["optical_flow_gt"],
            batch_converted["optical_flow_mask"],
            batch_converted["scene_flow_gt"],
            batch_converted["scene_flow_mask"],
            batch_converted["graph_nodes"],
            batch_converted["graph_edges"],
            batch_converted["graph_edges_weights"],
            batch_converted["graph_node_deformations"],
            batch_converted["graph_clusters"],
            batch_converted["pixel_anchors"],
            batch_converted["pixel_weights"],
            batch_converted["num_nodes"],
            batch_converted["intrinsics"],
            batch_converted["index"]
        ]


def erode_mesh(vertexPositions, faceIndices, nIterations, minNeighbors):
    """[summary]
    Args:
        vertexPositions ([type]): [N,3]
        faceIndices ([type]): [N,3]
        nIterations ([type]): int
        minNeighbors ([type]): int

    Returns:
        [type]: [description]
    """
    nonErodedVertices = erode_mesh_c(
        vertexPositions, faceIndices, nIterations, minNeighbors)
    return nonErodedVertices


def sample_nodes(vertexPositions, nonErodedVertices, nodeCoverage, useOnlyValidIndices):
    nodePositions = np.zeros(shape=vertexPositions.shape, dtype=np.float32)
    nodeIndices = np.zeros(
        shape=[vertexPositions.shape[0], 1], dtype=np.int)
    nodeIndices[:, :] = -1

    nodes_size = sample_nodes_c(vertexPositions, nonErodedVertices,
                                nodePositions, nodeIndices, nodeCoverage, useOnlyValidIndices)

    return nodePositions, nodeIndices, nodes_size


def sample_node_py_v2(vertexPositions, nodeCoverage=0.05):
    nodeCoverage2 = nodeCoverage * nodeCoverage
    nVertices = vertexPositions.shape[0]

    shuffledVertices = [i for i in range(nVertices)]
    np.random.shuffle(shuffledVertices)

    nodePositionsVec = []
    nodeIndices = []

    for vertexIdx in shuffledVertices:
        point = vertexPositions[vertexIdx]
        bIsNode = True
        for node in nodePositionsVec:
            if np.sum((point-node) ** 2) <= nodeCoverage2:
                bIsNode = False
                break

        if bIsNode:
            nodePositionsVec.append(vertexPositions[vertexIdx])
            nodeIndices.append(vertexIdx)

    return np.array(nodePositionsVec, dtype=np.float32), np.array(nodeIndices, np.int)


def sample_nodes_v3(vertexPositions, nodeCoverage=0.05):
    # down-sampling vertices at frist, then sample nodes
    org_pcd = o3d.geometry.PointCloud()
    org_pcd.points = o3d.utility.Vector3dVector(vertexPositions)
    output, cubic_id, original_indices = org_pcd.voxel_down_sample_and_trace(
        voxel_size=nodeCoverage*0.8, min_bound=vertexPositions.min(0), max_bound=vertexPositions.max(0))
    sampled_vertices = np.asarray(output.points)
    return sampled_vertices


def sample_nodes_py(vertexPositions, radius=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertexPositions)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(vertexPositions, dtype=np.uint8)*np.array([0, 0, 255]))
    # sample nodes python
    downpcd = pcd.voxel_down_sample(voxel_size=0.025*0.7)
    graph_nodes = downpcd.points
    graph_nodes = sample_nodes(graph_nodes, radius=radius)
    return np.array(graph_nodes)


def compute_edges_geodesic(vertexPositions, faceIndices, nodeIndices, nMaxNeighbors, maxInfluence):
    graphEdges = compute_edges_geodesic_c(
        vertexPositions, faceIndices, nodeIndices, nMaxNeighbors, maxInfluence)
    return graphEdges


def compute_edges_geodesic_py(vertexPositions, faceIndices, nodeIndices, nMaxNeighbors, maxInfluence):
    from queue import PriorityQueue

    nVertices = vertexPositions.shape[0]
    nFaces = faceIndices.shape[0]
    nNodes = nodeIndices.shape[0]

    vertexNeighbors = [[] for i in range(nVertices)]

    # Preprocess vertex neighbors.
    for faceIdx in range(nFaces):
        for j in range(3):
            v_idx = faceIndices[faceIdx, j]
            for k in range(3):
                n_idx = faceIndices[faceIdx, k]
                if(v_idx == n_idx):
                    continue
                vertexNeighbors[v_idx].append(n_idx)

    # Compute inverse vertex -> node relationship.
    mapVertexToNode = np.array([-1 for i in range(nVertices)])
    for nodeId in range(nNodes):
        vertexIdx = nodeIndices[nodeId]
        if vertexIdx > 0:
            mapVertexToNode[vertexIdx] = nodeId

    graphEdges = -np.ones(shape=[nNodes, nMaxNeighbors], dtype=np.int)
    for nodeId in range(nNodes):
        nextVerticesWithIds = PriorityQueue()
        visitedVertices = []

        # Add node vertex as the first vertex to be visited
        nodeVertexIdx = nodeIndices[nodeId]
        if nodeVertexIdx < 0:
            continue
        nextVerticesWithIds.put([0., nodeVertexIdx, ])

        # Traverse all neighbors in the monotonically increasing order.
        neighborNodeIds = []
        while not nextVerticesWithIds.empty():
            nextVertexDist, nextVertexIdx = nextVerticesWithIds.get()

            # We skip the vertex, if it was already visited before.
            if nextVertexIdx in visitedVertices:
                continue

            # We check if the vertex is a node.
            nextNodeId = mapVertexToNode[nextVertexIdx]
            if nextNodeId >= 0 and nextNodeId != nodeId:
                neighborNodeIds.append(nextNodeId)
                if len(neighborNodeIds) > nMaxNeighbors:
                    break

            # We visit the vertex, and check all his neighbors.
            # We add only vertices under a certain distance.
            visitedVertices.append(nextVertexIdx)
            nextVertexPos = vertexPositions[nextVertexIdx]
            nextNeighbors = vertexNeighbors[nextVertexIdx]
            for neighborIdx in nextNeighbors:
                neighborVertexPos = vertexPositions[neighborIdx]
                dist = nextVertexDist + \
                    np.linalg.norm(nextVertexPos - neighborVertexPos, ord=2)
                if dist <= maxInfluence:
                    nextVerticesWithIds.put([dist, neighborIdx])

        # If we don't get any geodesic neighbors, we take one nearest Euclidean neighbor,
        # to have a constrained optimization system at non-rigid tracking.
        if len(neighborNodeIds) == 0:
            nearestDistance2 = np.inf
            nearestNodeId = -1

            nodePos = vertexPositions[nodeVertexIdx]
            for i in range(nNodes):
                vertexIdx = nodeIndices[i]
                if i != nodeId and vertexIdx >= 0:
                    neighborPos = vertexPositions[vertexIdx]
                    distance2 = np.linalg.norm(neighborPos - nodePos, ord=2)
                    if distance2 < nearestDistance2:
                        nearestDistance2 = distance2
                        nearestNodeId = i

            if (nearestNodeId >= 0):
                neighborNodeIds.append(nearestNodeId)

        nNeighbors = min(nMaxNeighbors, len(neighborNodeIds))
        for i in range(nNeighbors):
            graphEdges[nodeId, i] = neighborNodeIds[i]
        for i in range(nNeighbors, nMaxNeighbors):
            graphEdges[nodeId, i] = -1

    return graphEdges


def compute_edges_euclidean(nodePositions, nMaxNeighbors=8):
    graphEdges = compute_edges_euclidean_c(nodePositions, nMaxNeighbors)
    return graphEdges


@numba.jit()
def compute_distance(src_points, target_points):
    num_src = src_points.shape[0]
    num_tgt = target_points.shape[0]
    distance = np.zeros(shape=[num_src, num_tgt])
    for i in range(num_src):
        for j in range(num_tgt):
            distance[i, j] = np.linalg.norm(
                src_points[i] - target_points[j], ord=2)
    return distance


def compute_edges_py(graph_nodes, nMaxNeighbors=8):
    distance = compute_distance(graph_nodes, graph_nodes)
    sorted_index = np.argsort(distance)
    graph_edges = sorted_index[:, 1:nMaxNeighbors]
    return graph_edges


def compute_pixel_anchors_geodesic(graphNodes, graphEdges, pointImage, neighborhoodDepth, nodeCoverage):
    nMaxNeighbors = graphEdges.shape[1]
    _, height, width = pointImage.shape
    pixelAnchors = np.zeros(shape=[height, width, nMaxNeighbors], dtype=np.int)
    pixelAnchors[:] = -1
    pixelWeights = np.zeros(
        shape=[height, width, nMaxNeighbors], dtype=np.float32)
    compute_pixel_anchors_geodesic_c(
        graphNodes, graphEdges, pointImage, neighborhoodDepth, nodeCoverage, pixelAnchors, pixelWeights)
    return pixelAnchors, pixelWeights


@numba.jit()
def compute_pixel_anchors_geodesic_py(pixelAnchors, pixelWeights, graphNodes, graphEdges, pointImage, neighborhoodDepth, nodeCoverage):
    numNodes, numNeighbors = graphNodes.shape
    GRAPH_K = 4
    _, height, width = pointImage.shape

    for y in range(height):
        for x in range(width):
            pixelPos = pointImage[:, y, x]
            if pixelPos[2] <= 0:
                continue

            # find nearest Euclidean graph node.
            dists = np.sqrt(((graphNodes-pixelPos) ** 2).sum(axis=1))
            nearestNodeId = np.argsort(dists)

            # Compute the geodesic neighbor candidates.
            neighbors = set([nearestNodeId, ])
            newNeighbors = set([nearestNodeId, ])
            for i in range(neighborhoodDepth):
                currentNeighbors = set()
                for neighborId in newNeighbors:
                    for k in range(numNeighbors):
                        currentNeighborId = graphEdges[neighborId, k]
                        if currentNeighborId >= 0:
                            currentNeighbors.add(currentNeighborId)
                newNeighbors.clear()
                newNeighbors = currentNeighbors - neighbors
                neighbors.union(newNeighbors)

            # Keep only the k nearest geodesic neighbors.
            nodes_distances = [np.linalg.norm(
                graphNodes[neighborId] - pixelPos, ord=2) for neighborId in neighbors]
            nearestNodes = np.argsort(nodes_distances)[:GRAPH_K]

            # Compute skinning weights.
            nearestGeodesicNodeIds, skinningWeights = [], []
            weightSum = 0
            for nodeId in nearestNodes:
                nodePose = graphNodes[nodeId]
                weight = np.exp(-(np.linalg.norm(pixelPos - nodePose, ord=2))
                                ** 2 / (2*nodeCoverage*nodeCoverage))
                weightSum += weight
                nearestGeodesicNodeIds.append(nodeId)
                skinningWeights.append(weight)

            nAnchors = len(nearestGeodesicNodeIds)
            if weightSum > 0:
                for i in range(nAnchors):
                    skinningWeights[i] = skinningWeights[i]/weightSum
            elif nAnchors > 0:
                for i in range(nAnchors):
                    skinningWeights[i] = 1 / nAnchors

            # Store the results
            for i in range(nAnchors):
                pixelAnchors[y, x] = np.array(nearestGeodesicNodeIds[i])
                pixelWeights[y, x] = np.array(skinningWeights[i])

    return pixelAnchors, pixelWeights


@numba.jit()
def compute_mesh_anchors_geodesic_py(Anchors, Weights, graphNodes, graphEdges,
                                     verts, neighborhoodDepth, nodeCoverage):
    numNodes, numNeighbors = graphEdges.shape
    GRAPH_K = 4
    nverts, _ = verts.shape
    for x in range(nverts):
        vertPos = verts[x]
        if vertPos[2] <= 0:
            continue

        # find nearest Euclidean graph node.
        dists = np.sqrt(((graphNodes-vertPos) ** 2).sum(axis=1))
        nearestNodeId = np.argsort(dists)[0]

        # Compute the geodesic neighbor candidates.
        neighbors = set([nearestNodeId, ])
        newNeighbors = set([nearestNodeId, ])
        for i in range(neighborhoodDepth):
            currentNeighbors = set()
            for neighborId in newNeighbors:
                for k in range(numNeighbors):
                    currentNeighborId = graphEdges[neighborId, k]
                    if currentNeighborId >= 0:
                        currentNeighbors.add(currentNeighborId)
            newNeighbors.clear()
            newNeighbors = currentNeighbors - neighbors
            neighbors = neighbors.union(newNeighbors)

        # Keep only the k nearest geodesic neighbors.
        dists = [np.linalg.norm(
            graphNodes[neighborId] - vertPos, ord=2) for neighborId in neighbors]
        neighbors = np.argsort(dists)[:GRAPH_K]

        # Compute skinning weights.
        nearestNodeIds, skinningWeights = [], []
        weightSum = 0
        for nodeId in neighbors:
            dist = dists[nodeId]
            if dist > nodeCoverage:
                continue
            weight = np.exp(-dist ** 2 / (2*nodeCoverage*nodeCoverage))
            weightSum += weight
            nearestNodeIds.append(nodeId)
            skinningWeights.append(weight)
        nAnchors = len(nearestNodeIds)

        if weightSum > 0:
            for i in range(nAnchors):
                skinningWeights[i] = skinningWeights[i]/weightSum
        elif nAnchors > 0:
            for i in range(nAnchors):
                skinningWeights[i] = 1 / nAnchors
        # Store the results
        for i in range(nAnchors):
            Anchors[x, i] = np.array(nearestNodeIds[i])
            Weights[x, i] = np.array(skinningWeights[i])

    return Anchors, Weights


@numba.jit()
def compute_mesh_anchors_euclidean_py(Anchors, Weights, graphNodes,  verts, nodeCoverage):
    GRAPH_K = 4
    nverts, _ = verts.shape
    for x in range(nverts):
        vertPos = verts[x]
        if vertPos[2] <= 0:
            continue
        # find nearest Euclidean graph node.
        dists = np.sqrt(((graphNodes-vertPos) ** 2).sum(axis=1))
        neighbors = np.argsort(dists)[:GRAPH_K]
        # Compute skinning weights.
        nearestNodeIds, skinningWeights = [], []
        weightSum = 0
        for nodeId in neighbors:
            dist = dists[nodeId]
            if dist > nodeCoverage:
                continue
            weight = np.exp(-dist ** 2 / (2*nodeCoverage*nodeCoverage))
            weightSum += weight
            nearestNodeIds.append(nodeId)
            skinningWeights.append(weight)
        nAnchors = len(nearestNodeIds)
        if weightSum > 0:
            for i in range(nAnchors):
                skinningWeights[i] = skinningWeights[i]/weightSum
        elif nAnchors > 0:
            for i in range(nAnchors):
                skinningWeights[i] = 1 / nAnchors
        # Store the results
        for i in range(nAnchors):
            Anchors[x, i] = np.array(nearestNodeIds[i])
            Weights[x, i] = np.array(skinningWeights[i])
    return Anchors, Weights


def compute_pixel_anchors_euclidean(graphNodes, pointImage, nodeCoverage):
    nMaxNeighbors = graphNodes.shape[0]
    _, height, width = pointImage.shape
    pixelAnchors = - \
        np.ones(shape=[height, width, nMaxNeighbors], dtype=np.int)
    pixelWeights = np.zeros(
        shape=[height, width, nMaxNeighbors], dtype=np.float32)
    compute_pixel_anchors_euclidean_c(
        graphNodes, pointImage, nodeCoverage, pixelAnchors, pixelWeights)
    return pixelAnchors, pixelWeights


@numba.jit()
def compute_pixel_anchors_euclidean_py(graphNodes, pointImage, nodeCoverage):
    GRAPH_K = 4
    _, height, width = pointImage.shape
    pixelAnchors = -np.ones(shape=[height, width, GRAPH_K], dtype=np.int)
    pixelWeights = np.zeros(
        shape=[height, width, GRAPH_K], dtype=np.float32)

    for y in range(height):
        for x in range(width):
            pixelPos = pointImage[:, y, x]
            if pixelPos[2] < 0:
                continue
            # find nearest Euclidean graph node.
            dists = np.sqrt(((graphNodes-pixelPos) ** 2).sum(axis=1))
            neighbors = np.argsort(dists)[:GRAPH_K]

            # Compute skinning weights.
            nearestEuclideanNodeIds, skinningWeights = [], []
            weightSum = 0
            for nodeId in neighbors:
                distance = dists[nodeId]
                if distance > nodeCoverage:
                    continue
                weight = np.exp(-distance ** 2 / (2*nodeCoverage*nodeCoverage))
                weightSum += weight
                nearestEuclideanNodeIds.append(nodeId)
                skinningWeights.append(weight)

            nAnchors = len(nearestEuclideanNodeIds)
            if weightSum > 0:
                for i in range(nAnchors):
                    skinningWeights[i] = skinningWeights[i]/weightSum
            elif nAnchors > 0:
                for i in range(nAnchors):
                    skinningWeights[i] = 1 / nAnchors

            # Store the results
            for i in range(nAnchors):
                pixelAnchors[y, x, i] = np.array(nearestEuclideanNodeIds[i])
                pixelWeights[y, x, i] = np.array(skinningWeights[i])

    return pixelAnchors, pixelWeights


@ numba.jit()
def compute_voxel_anchors(voxel_anchors, voxel_weigths, transfromed_graphNodes,
                          w2d_r, w2d_t, cell_size, nodeCoverage):
    X_SIZE, Y_SIZE, Z_SIZE = voxel_anchors.shape[:3]
    GRAPH_K = 4
    for ix in range(X_SIZE):
        for iy in range(Y_SIZE):
            for iz in range(Z_SIZE):
                voxelPos = (np.array([ix, iy, iz]) + 0.5) * cell_size

                voxel_depth_frame = np.dot(voxelPos, w2d_r) + w2d_t
                if (voxel_depth_frame[2] < 0):
                    continue

                # find nearest Euclidean graph node.
                dists = np.sqrt(
                    ((transfromed_graphNodes-voxelPos) ** 2).sum(axis=1))
                neighbors = np.argsort(dists)[:GRAPH_K]

                # Compute skinning weights.
                nearestEuclideanNodeIds, skinningWeights = [], []
                weightSum = 0
                for nodeId in neighbors:
                    distance = dists[nodeId]
                    if distance > nodeCoverage:
                        continue
                    weight = np.exp(-distance ** 2 /
                                    (2*nodeCoverage*nodeCoverage))
                    weightSum += weight
                    nearestEuclideanNodeIds.append(nodeId)
                    skinningWeights.append(weight)

                nAnchors = len(nearestEuclideanNodeIds)
                if weightSum > 0:
                    for i in range(nAnchors):
                        skinningWeights[i] = skinningWeights[i]/weightSum
                elif nAnchors > 0:
                    for i in range(nAnchors):
                        skinningWeights[i] = 1 / nAnchors

                # Store the results
                for i in range(nAnchors):
                    voxel_anchors[ix, iy, iz, i] = nearestEuclideanNodeIds[i]
                    voxel_weigths[ix, iy, iz, i] = skinningWeights[i]

    return voxel_anchors, voxel_weigths


def construct_regular_graph(pointImage, xNodes, yNodes, edgeThreshold, maxPointToNodeDistance,
                            maxDepth, ):
    _, height, width = pointImage.shape
    graphNodes = np.zeros(shape=[height*width, 3])
    graphEdges = np.zeros(shape=[height*width, 8])
    pixelAnchors = np.zeros(shape=[height, width, 4], dtype=np.int)
    pixelWeights = np.zeros(shape=[height, width, 4], dtype=np.float32)

    graphNodes, graphEdges, pixelAnchors, pixelWeights = construct_regular_graph_c(pointImage, xNodes, yNodes, edgeThreshold, maxPointToNodeDistance,
                                                                                   maxDepth, graphNodes, graphEdges, pixelAnchors, pixelWeights)
    return graphNodes, graphEdges, pixelAnchors, pixelWeights


def construct_regular_graph_py(pointImage, xNodes, yNodes, edgeThreshold,
                               maxPointToNodeDistance, maxDepth):
    _, height, width = pointImage.shape
    xStep = (width - 1) / (xNodes - 1)
    yStep = (height - 1) / (yNodes - 1)

    # Sample graph nodes.
    nNodes = xNodes * yNodes
    sampledNodeMapping = np.array([-1] * nNodes, dtype=np.int)
    nodePositions = []

    nodeId = 0
    for y in range(yNodes):
        for x in range(xNodes):
            nodeIdx = y * xNodes + x
            xPixel = round(x * xStep)
            yPixel = round(y * yStep)
            pixelPos = pointImage[:, yPixel, xPixel]
            if pixelPos[2] <= 0 or pixelPos[2] > maxDepth:
                continue
            sampledNodeMapping[nodeIdx] = nodeId
            nodePositions.append(pixelPos)
            nodeId += 1

    nSampledNodes = nodeId

    # build graph edges
    numNeighbors = 8
    edgeThreshold2 = edgeThreshold * edgeThreshold
    sampledNodeEdges = np.array(
        [-1]*(nSampledNodes*numNeighbors), dtype=np.int)
    connectedNodes = np.array([False]*nSampledNodes, dtype=np.bool)

    nConnectedNodes = 0
    for y in range(yNodes):
        for x in range(xNodes):
            nodeIdx = y * xNodes + x
            nodeId = sampledNodeMapping[nodeIdx]
            if nodeId >= 0:
                nodePosition = nodePositions[nodeId]
                neighborCount = 0
                for yDelta in range(-1, 2):
                    for xDelta in range(-1, 2):
                        xNeighbor = x + xDelta
                        yNeighbor = y + yDelta
                        if xNeighbor < 0 or xNeighbor >= xNodes or yNeighbor < 0 or yNeighbor >= yNodes:
                            continue
                        neighborIdx = yNeighbor * xNodes + xNeighbor
                        if neighborIdx == nodeIdx or neighborIdx < 0:
                            continue
                        neighborId = sampledNodeMapping[neighborIdx]
                        if neighborId >= 0:
                            neighborPosition = nodePositions[neighborId]
                            if np.linalg.norm(neighborPosition - nodePosition, ord=2) <= edgeThreshold2:
                                sampledNodeEdges[nodeId * numNeighbors +
                                                 neighborCount] = neighborId
                                neighborCount += 1

                for i in range(neighborCount, numNeighbors):
                    sampledNodeEdges[nodeId * numNeighbors + i] = -1

                if neighborCount > 0:
                    connectedNodes[nodeId] = True
                    nConnectedNodes += 1

    validNodeMapping = np.array([-1]*nSampledNodes, dtype=np.int)
    graphNodes = np.zeros(shape=[nConnectedNodes, 3], dtype=np.float32)
    graphEdges = np.zeros(shape=[nConnectedNodes, numNeighbors], dtype=np.int)

    validNodeId = 0
    for y in range(0, yNodes):
        for x in range(0, xNodes):
            nodeIdx = y * xNodes + x
            nodeId = sampledNodeMapping[nodeIdx]
            if nodeId >= 0 and connectedNodes[nodeId]:
                validNodeMapping[nodeId] = validNodeId
                nodePosition = nodePositions[nodeId]
                graphNodes[validNodeId] = nodePosition
                validNodeId += 1

    for y in range(0, yNodes):
        for x in range(0, xNodes):
            nodeIdx = y * xNodes + x
            nodeId = sampledNodeMapping[nodeIdx]
            if nodeId >= 0 and connectedNodes[nodeId]:
                validNodeId = validNodeMapping[nodeId]

                if validNodeId >= 0:
                    for i in range(numNeighbors):
                        sampledNeighborId = sampledNodeEdges[nodeId *
                                                             numNeighbors + i]
                        if sampledNeighborId >= 0:
                            graphEdges[validNodeId,
                                       i] = validNodeMapping[sampledNeighborId]
                        else:
                            graphEdges[validNodeId, i] = -1

    # compute graph edge wight
    mask = (graphEdges != -1)
    nodesEdgeDiff = graphNodes[graphEdges] - graphNodes[:, np.newaxis, :]
    nodesEdgeLength = np.sqrt((nodesEdgeDiff ** 2).sum(-1))
    graphWeights = np.exp(-nodesEdgeLength) / \
        (np.exp(-nodesEdgeLength) * mask).sum(-1, keepdims=True)
    graphWeights = graphWeights * mask

    # Compute pixel anchors and weights.
    pixelAnchors = -np.ones(shape=[height, width, 4], dtype=np.int)
    pixelWeights = np.zeros(shape=[height, width, 4], dtype=np.float32)
    for y in range(0, height):
        for x in range(0, width):
            xNode = float(x) / xStep
            yNode = float(y) / yStep
            x0 = int(np.floor(xNode))
            x1 = x0+1
            y0 = int(np.floor(yNode))
            y1 = y0+1

            if x0 < 0 or x1 >= xNodes or y0 < 0 or y1 >= yNodes:
                continue
            sampledNode00 = sampledNodeMapping[y0 * xNodes + x0]
            sampledNode01 = sampledNodeMapping[y1 * xNodes + x0]
            sampledNode10 = sampledNodeMapping[y0 * xNodes + x1]
            sampledNode11 = sampledNodeMapping[y1 * xNodes + x1]

            if sampledNode00 < 0 or sampledNode01 < 0 or sampledNode10 < 0 or sampledNode11 < 0:
                continue

            validNode00 = validNodeMapping[sampledNode00]
            validNode01 = validNodeMapping[sampledNode01]
            validNode10 = validNodeMapping[sampledNode10]
            validNode11 = validNodeMapping[sampledNode11]

            if validNode00 < 0 or validNode01 < 0 or validNode10 < 0 or validNode11 < 0:
                continue

            pixelPos = pointImage[:, y, x]
            if pixelPos[2] <= 0 or pixelPos[2] > maxDepth:
                continue
            if (np.linalg.norm(pixelPos - nodePositions[sampledNode00], ord=2) > maxPointToNodeDistance) or \
                    (np.linalg.norm(pixelPos - nodePositions[sampledNode01], ord=2) > maxPointToNodeDistance) or \
                    (np.linalg.norm(pixelPos - nodePositions[sampledNode10], ord=2) > maxPointToNodeDistance) or \
                    (np.linalg.norm(pixelPos - nodePositions[sampledNode11], ord=2) > maxPointToNodeDistance):
                continue
            dx = xNode - x0
            dy = yNode - y0
            w00 = (1 - dx) * (1 - dy)
            w01 = (1 - dx) * dy
            w10 = dx * (1 - dy)
            w11 = dx * dy

            pixelAnchors[y, x] = np.array(
                [validNode00, validNode01, validNode10, validNode11])
            pixelWeights[y, x] = np.array([w00, w01, w10, w11])

    return graphNodes, graphEdges, graphWeights, pixelAnchors, pixelWeights


def compute_mesh_from_depth(pointImage, maxTriangleEdgeDistance):
    c, h, w = pointImage.shape
    vertexPositions = np.zeros(shape=[h*w, 3], dtype=np.float32)
    faceIndices = np.zeros(shape=[h*w, 3], dtype=np.int)
    compute_mesh_from_depth_c(
        pointImage, maxTriangleEdgeDistance, vertexPositions, faceIndices)
    return vertexPositions, faceIndices


def compute_mesh_from_depth_and_color(pointImage, colorImage, maxTriangleEdgeDistance,
                                      vertexPositions, vertexColors, faceIndices):
    compute_mesh_from_depth_and_color_c(pointImage, colorImage, maxTriangleEdgeDistance,
                                        vertexPositions, vertexColors, faceIndices)


def prepare_graph(depth_image_path, intric_path, mask_image_path=None, max_depth=2.4):

    intric = np.loadtxt(intric_path)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    depth_image = cv2.imread(
        depth_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_image[depth_image > max_depth*1000] = 0.

    # Backproject depth image.
    point_image = image_proc.backproject_depth(
        depth_image, fx, fy, cx, cy)  # (3, h, w)
    point_image = point_image.astype(np.float32)
    _, height, width = point_image.shape

    # graph related
    graphNodes, graphEdges, \
        graphWeights, pixelAnchors, \
        pixelWeights = construct_regular_graph_py(point_image, xNodes=width//10, yNodes=height//10,
                                                  edgeThreshold=1000,
                                                  maxPointToNodeDistance=1000,
                                                  maxDepth=max_depth)
    # useless item
    graph_clusters = np.array(
        [0]*graphNodes.shape[0], dtype=np.int)[:, np.newaxis]

    return graphNodes, graphEdges, graphWeights, graph_clusters, pixelAnchors, pixelWeights


def prepare_graph_v2(vertice, faces, init_pose, depth_image_path, intric_path, max_depth=2.4):
    intric = np.loadtxt(intric_path)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    depth_image = cv2.imread(
        depth_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_image[depth_image > max_depth*1000] = 0.

    # Backproject depth image.
    point_image = image_proc.backproject_depth(
        depth_image, fx, fy, cx, cy)  # (3, h, w)
    point_image = point_image.astype(np.float32)
    _, height, width = point_image.shape

    # build graph nodes and edges
    transformed_vertices = np.dot(
        init_pose[:3, :3], vertice.T).T + init_pose[:3, 3]
    graphNodes, graphNodesIndices = sample_node_py_v2(
        transformed_vertices, nodeCoverage=0.05)
    graphEdges = compute_edges_geodesic_py(
        transformed_vertices, faces, graphNodesIndices, nMaxNeighbors=8, maxInfluence=0.5)

    mask = (graphEdges != -1)
    nodesEdgeDiff = graphNodes[graphEdges] - graphNodes[:, np.newaxis, :]
    nodesEdgeLength = np.sqrt((nodesEdgeDiff ** 2).sum(-1))
    graphWeights = np.exp(-nodesEdgeLength) / \
        (np.exp(-nodesEdgeLength) * mask).sum(-1, keepdims=True)
    graphWeights = graphWeights * mask

    # compute archors and weights
    pixelAnchors, pixelWeights = compute_pixel_anchors_euclidean_py(
        graphNodes, point_image, 0.2)

    # useless item
    graph_clusters = np.array(
        [0]*graphNodes.shape[0], dtype=np.int)[:, np.newaxis]

    return graphNodes, graphEdges, graphWeights, graph_clusters, pixelAnchors, pixelWeights


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                continue
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' %
                   (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()


if __name__ == '__main__':
    pass
