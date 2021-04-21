

from logging import debug
from operator import matmul
import open3d as o3d
from tqdm import tqdm
import numba
import numpy as np
import cv2
from PIL import Image
import os
import json
from skimage import measure
import trimesh
import time

from model import dataset
import utils.viz_utils as viz_utils
import utils.line_mesh as line_mesh_utils

import platform
system_type = platform.system()


@numba.jit()
def on_border(depth_frame, du, dv, win_size=11, dist_limit=0.1):
    height, width = depth_frame.shape[:2]
    win_size = 11
    half_size = win_size // 2
    left = max(0, du - half_size)
    right = min(width-1, du+half_size)
    top = max(0, dv-half_size)
    bottom = min(height - 1, dv + half_size)
    depth_center = depth_frame[dv, du]

    for ridx in range(top, bottom+1):
        for cidx in range(left, right+1):
            depth_around = depth_frame[ridx, cidx]
            if ((depth_center - depth_around) / 1000.0 > dist_limit):
                return True

    return False

# v1, this version build  [0, 1] cannonical space


def get_init_pose(depth_image, depth_intr, max_depth=6, min_depth=0):
    height, width = depth_image.shape[:2]
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    point_image = dataset.DeformDataset.backproject_depth(
        depth_image, fx, fy, cx, cy)
    point_image = np.moveaxis(point_image, 0, -1)
    points = point_image[np.logical_and(
        point_image[:, :, 2] > min_depth, point_image[:, :, 2] < max_depth)]
    m = points.min(axis=0)
    M = points.max(axis=0)

    scale = (M-m).max()
    init_pose = np.eye(4)
    init_pose[0, 0] = scale
    init_pose[1, 1] = scale
    init_pose[2, 2] = scale
    init_pose[0, 3] = m[0]
    init_pose[1, 3] = m[1]
    init_pose[2, 3] = m[2]

    translate_matrix = np.eye(4)
    translate_matrix[0, 3] = -0.25
    translate_matrix[1, 3] = 0
    translate_matrix[2, 3] = -0.25
    init_pose = np.dot(translate_matrix, init_pose)
    return init_pose


def get_init_tsdf_parmas(depth_image, depth_intr, vol_resolution, max_depth=6, min_depth=0, ):
    height, width = depth_image.shape[:2]
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    point_image = dataset.DeformDataset.backproject_depth(
        depth_image, fx, fy, cx, cy)
    point_image = np.moveaxis(point_image, 0, -1)
    points = point_image[np.logical_and(
        point_image[:, :, 2] > min_depth, point_image[:, :, 2] < max_depth)]
    m = points.min(axis=0)
    M = points.max(axis=0)

    # set tsdf volume boundary
    diff = (M-m).max()
    vol_bound_min = m - 0.25*diff
    vol_bound_max = M + 0.25*diff
    cell_size = ((vol_bound_max - vol_bound_min)/vol_resolution)
    init_pose = np.eye(4)
    tranc_dist = 3 * cell_size[0]

    return init_pose, cell_size, tranc_dist, vol_bound_min, vol_bound_max


@numba.jit()
def first_integrate_tsdf_volume(depth_frame, color_frame, depth_intr,
                                w2d_r, w2d_t, trunc_dist, tsdf_volume, color_volume,
                                cell_size,):
    height, width = depth_frame.shape[:2]
    X_SIZE, Y_SIZE, Z_SIZE = tsdf_volume.shape[:3]
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    for ix in range(X_SIZE):
        for iy in range(Y_SIZE):
            for iz in range(Z_SIZE):
                voxel = (np.array([ix, iy, iz]) + 0.5) * cell_size
                voxel_depth_frame = np.dot(voxel, w2d_r) + w2d_t
                if (voxel_depth_frame[2] < 0):
                    continue

                du = int(fx * (voxel_depth_frame[0]/voxel_depth_frame[2]) + cx)
                dv = int(fy * (voxel_depth_frame[1]/voxel_depth_frame[2]) + cy)

                if (du > 0 and du < width and dv > 0 and dv < height):
                    # is_border = on_border(depth_frame, du, dv)
                    # if is_border:
                    #     continue

                    depth = depth_frame[dv, du] / 1000.

                    psdf = depth - voxel_depth_frame[2]
                    if (depth > 0 and psdf > -trunc_dist):

                        tsdf = min(1., psdf / trunc_dist)
                        tsdf_prev, weight_prev = tsdf_volume[ix,
                                                             iy, iz][0], tsdf_volume[ix, iy, iz][1]
                        Wrk = 1
                        tsdf_new = (tsdf_prev * weight_prev +
                                    Wrk * tsdf) / (weight_prev + Wrk)
                        weight_new = min(weight_prev + Wrk, 255)
                        tsdf_volume[ix, iy, iz][0], tsdf_volume[ix,
                                                                iy, iz][1] = tsdf_new, weight_new
                        if(abs(psdf) < trunc_dist):
                            color_volume[ix, iy, iz] = color_frame[dv, du]


@numba.jit()
def integrate_tsdf_volume(depth_frame, depth_intr, w2d_r, w2d_t, cell_size, tsdf_volume, trunc_dist,
                          voxel_anchors, voxel_weigths, node_positions, node_rotations, node_translations,
                          mask=None):
    '''
    volume_anchor_indexs: [X_SIZE,Y_SIZE,Z_SIZE, K]
    volume_anchor_weights: [X_SIZE,Y_SIZE,Z_SIZE, K]
    '''
    height, width = depth_frame.shape[:2]
    X_SIZE, Y_SIZE, Z_SIZE = tsdf_volume.shape[:3]
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    for ix in range(X_SIZE):
        for iy in range(Y_SIZE):
            for iz in range(Z_SIZE):
                voxel = (np.array([ix, iy, iz]) + 0.5) * cell_size
                voxel_depth_frame = np.dot(voxel, w2d_r) + w2d_t

                # deforamtion
                anchors = voxel_anchors[ix, iy, iz]
                weight = voxel_weigths[ix, iy, iz]
                point_validity = np.all(anchors != -1)
                if not point_validity:
                    deoformed_pos = voxel_depth_frame
                else:
                    deoformed_pos = np.matmul(
                        node_rotations[anchors], (voxel_depth_frame - node_positions[anchors])[..., np.newaxis])
                    deoformed_pos = deoformed_pos.reshape(
                        4, 3) + node_positions[anchors] + node_translations[anchors]
                    deoformed_pos = deoformed_pos * weight.reshape(4, 1)
                    deoformed_pos = deoformed_pos.sum(0)

                if (deoformed_pos[2] <= 0):
                    continue
                du = int(fx * (deoformed_pos[0]/deoformed_pos[2]) + cx)
                dv = int(fy * (deoformed_pos[1]/deoformed_pos[2]) + cy)

                if (du > 0 and du < width and dv > 0 and dv < height):
                    depth = depth_frame[dv, du] / 1000.
                    psdf = depth - deoformed_pos[2]

                    if depth > 0:
                        if mask is not None:
                            mask[dv, du] = 1

                    if (depth > 0 and psdf > -trunc_dist):

                        tsdf = min(1., psdf / trunc_dist)
                        tsdf_prev, weight_prev = tsdf_volume[ix,
                                                             iy, iz][0], tsdf_volume[ix, iy, iz][1]
                        Wrk = 1
                        tsdf_new = (tsdf_prev * weight_prev +
                                    Wrk * tsdf) / (weight_prev + Wrk)
                        weight_new = min(weight_prev + Wrk, 255)
                        tsdf_volume[ix, iy, iz][0], tsdf_volume[ix,
                                                                iy, iz][1] = tsdf_new, weight_new


def warp_verts(verts, init_pose, anchors, weights, node_positions, node_rotations, node_translations):
    if init_pose is not None:
        transformed_verts = np.dot(
            init_pose[:3, :3], verts.T).T + init_pose[:3, 3]
    else:
        transformed_verts = verts
    num_nodes = node_translations.shape[0]
    node_translations = node_translations.reshape(num_nodes, 3, 1)
    deoformed_pos = transformed_verts[:, np.newaxis] - node_positions[anchors]
    deoformed_pos = np.matmul(
        node_rotations[anchors], deoformed_pos[..., np.newaxis])
    deoformed_pos = deoformed_pos[..., 0] + \
        node_positions[anchors] + node_translations[anchors][..., 0]
    deoformed_pos = deoformed_pos * weights[..., np.newaxis]
    deoformed_pos = deoformed_pos.sum(1)
    # if deformation fail
    point_validity = np.all(anchors != -1, axis=1)
    deoformed_pos[~point_validity] = transformed_verts[~point_validity]
    return deoformed_pos


def sample_nodes(vertices, faces=None, radius=0.025):
    nodes_list = []
    for v in tqdm(vertices):
        if len(nodes_list) == 0:
            nodes_list.append(v)
            continue
        is_node = True
        for nv in nodes_list:
            if np.linalg.norm(v - nv, ord=2) < radius:
                is_node = False
                break
        if is_node:
            nodes_list.append(v)
    return np.array(nodes_list)


def new_build_Graph():
    if system_type == 'linux':
        intric_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/colorIntrinsics.txt'
        color_image_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/frame-000150.color.png'
        depth_image_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/frame-000150.depth.png'
    else:
        intric_path = r'E:/VolumeDeformData/upperbody/data/colorIntrinsics.txt'
        color_image_path = r'E:/VolumeDeformData/upperbody/data/frame-000150.color.png'
        depth_image_path = r'E:/VolumeDeformData/upperbody/data/frame-000150.depth.png'

    # laod RGBD and instric param
    intric = np.loadtxt(intric_path)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    color_image = cv2.imread(
        color_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_image = cv2.imread(
        depth_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    mask = np.logical_and(depth_image > 200, depth_image < 1000).astype(np.int)
    # depth_image = depth_image * mask
    # cv2.imshow("mask", (mask*255).astype(np.uint8))
    # cv2.waitKey(0)
    # build tsdf
    vol_size = np.ones(3, dtype=np.float32)
    vol_resolution = np.ones(3, dtype=np.int)*256
    cell_size = vol_size / vol_resolution
    init_pose = np.eye(4)
    init_pose[0, 3] = -vol_size[0] / 2
    init_pose[1, 3] = -vol_size[1] / 2
    init_pose[2, 3] = 0.5
    w2d_r = init_pose[:3, :3]
    w2d_t = init_pose[:3, 3]

    tranc_dist = 0.01
    tsdf_vol = np.zeros(shape=[vol_resolution[0], vol_resolution[1],
                               vol_resolution[2], 2], dtype=np.float32)
    tsdf_vol[..., 0] = 32767
    color_vol = np.zeros(
        shape=[vol_resolution[0], vol_resolution[1], vol_resolution[2], 3], dtype=np.float32)

    first_integrate_tsdf_volume(depth_image, color_image, intric, intric,
                                w2d_r, w2d_t, tranc_dist, tsdf_vol, color_vol, cell_size)

    # marching cube
    verts, faces, norms, vals = measure.marching_cubes(
        tsdf_vol[:, :, :, 0], level=0)
    verts = verts * cell_size
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # mesh.show()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(verts, dtype=np.uint8)*np.array([0, 0, 255]))

    # sample nodes with graph_proc
    nonErodedVertices = dataset.erode_mesh(verts, faces, 1, 8)
    eroded_nodes = verts[nonErodedVertices[:, 0]]
    nodePositions, nodeIndices, nodes_size = dataset.sample_nodes(
        verts, nonErodedVertices, 0.025, False)
    nodePositions = nodePositions[:nodes_size]
    nodeIndices = nodeIndices[:nodes_size]
    nodes_pcd = o3d.geometry.PointCloud()
    nodes_pcd.points = o3d.utility.Vector3dVector(nodePositions)
    nodes_pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(nodePositions, dtype=np.uint8)*np.array([255, 0, 0]))
    o3d.visualization.draw_geometries([nodes_pcd])

    # # build graph
    graph_edges = dataset.compute_edges_geodesic(
        verts, faces, nodeIndices, nMaxNeighbors=8, maxInfluence=0.025 * 4)
    graph_nodes = nodePositions
    graph_clusters = np.zeros(shape=[graph_nodes.shape[0], 1])

    # visual graph
    rendered_graph_nodes = []
    for node in graph_nodes:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        mesh_sphere.translate(node)
        rendered_graph_nodes.append(mesh_sphere)
    rendered_graph_nodes = viz_utils.merge_meshes(rendered_graph_nodes)
    edges_pairs = []
    for node_id, edges in enumerate(graph_edges):
        for neighbor_id in edges:
            if neighbor_id == -1:
                break
            edges_pairs.append([node_id, neighbor_id])
    colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
    line_mesh = line_mesh_utils.LineMesh(
        graph_nodes, edges_pairs, colors, radius=0.003)
    line_mesh_geoms = line_mesh.cylinder_segments
    line_mesh_geoms = viz_utils.merge_meshes(line_mesh_geoms)
    rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
    o3d.visualization.draw_geometries(rendered_graph + [pcd])

    # compute anchor and weight

    # inference

    # integrate


if __name__ == '__main__':

    # type 2, regular graph
    if system_type == 'linux':
        intric_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/colorIntrinsics.txt'
        color_image_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/frame-000150.color.png'
        depth_image_path = r'/media/baldr/3C07145D2FEBB57B/VolumeDeformData/upperbody/data/frame-000150.depth.png'
    else:
        intric_path = r'D:\deepdeform_v1_1\train\seq069/intrinsics.txt'
        color_image_path = r'D:\deepdeform_v1_1\train\seq069\color/000000.jpg'
        depth_image_path = r'D:\deepdeform_v1_1\train\seq069\depth/000000.png'
        mask_image_path = r'D:\deepdeform_v1_1\train\seq069\mask\000000.png'

    # laod RGBD and instric param
    intric = np.loadtxt(intric_path)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    color_image = cv2.imread(
        color_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_image = cv2.imread(
        depth_image_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    mask = cv2.imread(mask_image_path)[..., 0] == 255
    depth_image = depth_image * np.logical_and(mask, depth_image < 2400)

    # Backproject depth image.
    point_image = dataset.image_proc.backproject_depth(
        depth_image, fx, fy, cx, cy)  # (3, h, w)
    point_image = point_image.astype(np.float32)
    point_image_hw3 = np.moveaxis(point_image, 0, -1)  # (h, w, 3)

    points = point_image_hw3[point_image_hw3[:, :, -1] != 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones_like(points, dtype=np.uint8)*np.array([255, 0, 0]))
    # o3d.visualization.draw_geometries([pcd])

    _, height, width = point_image.shape
    graphNodes, graphEdges,\
        graphWeights, pixelAnchors, \
        pixelWeights = dataset.construct_regular_graph_py(point_image, xNodes=width//8, yNodes=height//8,
                                                          edgeThreshold=1000,
                                                          maxPointToNodeDistance=1000,
                                                          maxDepth=2400)

    # visual graph
    rendered_graph_nodes = []
    for node in graphNodes:
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        mesh_sphere.translate(node)
        rendered_graph_nodes.append(mesh_sphere)
    rendered_graph_nodes = viz_utils.merge_meshes(rendered_graph_nodes)
    edges_pairs = []
    for node_id, edges in enumerate(graphEdges):
        for neighbor_id in edges:
            if neighbor_id == -1:
                break
            edges_pairs.append([node_id, neighbor_id])
    colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
    line_mesh = line_mesh_utils.LineMesh(
        graphNodes, edges_pairs, colors, radius=0.003)
    line_mesh_geoms = line_mesh.cylinder_segments
    line_mesh_geoms = viz_utils.merge_meshes(line_mesh_geoms)
    rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
    o3d.visualization.draw_geometries(rendered_graph + [pcd])
    pass
