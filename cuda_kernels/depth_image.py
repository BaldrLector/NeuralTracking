import numba
from numba import cuda, float32, int32
import numpy as np
import math
import cmath
from cuda_kernels.cuda_func import *
import cv2

GRAPH_K = 4


@cuda.jit()
def cuda_compute_pixel_anchors_euclidean_kernel(graphNodes, pointImage, nodeCoverage, pixelAnchors, pixelWeights):
    _, height, width = pointImage.shape
    x, y = cuda.grid(2)
    num_nodes = graphNodes.shape[0]
    if x >= height or y >= width:
        return
    pixelPos_x = pointImage[0, x, y]
    pixelPos_y = pointImage[1, x, y]
    pixelPos_z = pointImage[2, x, y]
    if pixelPos_z <= 0:
        return

    # find nearest Euclidean graph node
    dist_array = cuda.local.array(shape=1000, dtype=numba.types.float32)
    index_array = cuda.local.array(shape=1000, dtype=numba.types.int32)

    for node_idx in range(num_nodes):
        dist = sqrt_distance(pixelPos_x, pixelPos_y, pixelPos_z,
                             graphNodes[node_idx, 0], graphNodes[node_idx, 1], graphNodes[node_idx, 2])
        dist_array[node_idx] = dist
        index_array[node_idx] = node_idx

    for i in range(GRAPH_K):
        min_idx = i
        for j in range(i+1, num_nodes):
            if dist_array[min_idx] > dist_array[j]:
                min_idx = j

        if min_idx != i:
            temp = dist_array[min_idx]
            dist_array[min_idx] = dist_array[i]
            dist_array[i] = temp

            temp = index_array[min_idx]
            index_array[min_idx] = index_array[i]
            index_array[i] = temp

    nAnchors = 0
    weightSum = 0
    for i in range(GRAPH_K):
        distance = dist_array[i]
        index = index_array[i]
        if distance > nodeCoverage:
            continue
        weight = math.exp(-math.pow(distance, 2) /
                          (2*nodeCoverage*nodeCoverage))
        weightSum += weight
        nAnchors += 1

        pixelAnchors[x, y, i] = index
        pixelWeights[x, y, i] = weight

    if weightSum > 0:
        for i in range(GRAPH_K):
            pixelWeights[x, y, i] = pixelWeights[x, y, i]/weightSum
    elif nAnchors > 0:
        for i in range(GRAPH_K):
            pixelWeights[x, y, i] = 1 / nAnchors



def cuda_compute_pixel_anchors_euclidean(graphNodes, pointImage, nodeCoverage, kernel_size=32):
    threadsperblock = (kernel_size, kernel_size)
    _, height, width = pointImage.shape
    blockspergrid_x = math.ceil(height / threadsperblock[0])
    blockspergrid_y = math.ceil(width / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    pixelAnchors = -np.ones(shape=[height, width, GRAPH_K], dtype=np.int)
    pixelWeights = np.zeros(shape=[height, width, GRAPH_K], dtype=np.float32)
    cuda_compute_pixel_anchors_euclidean_kernel[blockspergrid, threadsperblock](
        graphNodes, pointImage, nodeCoverage, pixelAnchors, pixelWeights)
    cuda.synchronize()

    return pixelAnchors, pixelWeights


@cuda.jit()
def cuda_compute_normal_kernel(vertex_map, normal_map):
    x, y = cuda.grid(2)
    _, height, width = vertex_map.shape[:3]
    if x >= height or y >= width:
        return

    left_x, left_y, left_z = vertex_map[0, x, y -
                                        1], vertex_map[1, x, y-1], vertex_map[2, x, y-1]

    rigth_x, rigth_y, rigth_z = vertex_map[0, x, y +
                                           1], vertex_map[1, x, y+1], vertex_map[2, x, y+1]

    up_x, up_y, up_z = vertex_map[0, x-1,
                                  y], vertex_map[1, x-1, y], vertex_map[2, x-1, y]

    low_x, low_y, low_z = vertex_map[0, x+1,
                                     y], vertex_map[1, x+1, y], vertex_map[2, x+1, y]

    if left_z == 0 or rigth_z == 0 or up_z == 0 or low_z == 0:
        normal_map[x, y, 0] = 0.0
        normal_map[x, y, 1] = 0.0
        normal_map[x, y, 2] = 0.0
    else:
        hor_x, hor_y, hor_z = left_x-rigth_x, left_y-rigth_y, left_z-rigth_z
        ver_x, ver_y, ver_z = up_x-low_x, up_y-low_y, up_z-low_z
        cx, cy, cz = cross(hor_x, hor_y, hor_z, ver_x, ver_y, ver_z)
        ncx, ncy, ncz = normlize(cx, cy, cz)
        if ncz > 0:
            normal_map[x, y, 0] = -ncx
            normal_map[x, y, 1] = -ncy
            normal_map[x, y, 2] = -ncz
        else:
            normal_map[x, y, 0] = ncx
            normal_map[x, y, 1] = ncy
            normal_map[x, y, 2] = ncz


def cuda_compute_normal(vertex_map):
    threadsperblock = (16, 16)
    _, height, width = vertex_map.shape[:3]
    blockspergrid_x = math.ceil(height / threadsperblock[0])
    blockspergrid_y = math.ceil(width / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    normal_map = np.zeros(shape=[height, width, 3], dtype=np.float32)
    cuda_compute_normal_kernel[blockspergrid,
                               threadsperblock](vertex_map, normal_map)
    cuda.synchronize()
    return normal_map


def depth2color(depth_image):
    visual_depth_img = np.copy(depth_image)
    Md = visual_depth_img[visual_depth_img != 0].max()
    md = visual_depth_img[visual_depth_img != 0].min()
    visual_depth_img[visual_depth_img != 0] = (
        visual_depth_img[visual_depth_img != 0] - md)/(Md-md) * 255
    colored_depth = cv2.applyColorMap(
        visual_depth_img.astype(np.uint8), cv2.COLORMAP_JET)
    return colored_depth


def biliteral_smooth(depth_image):
    mask = (depth_image == 0)
    M = depth_image.max()
    m = depth_image.min()
    depth_image = (depth_image-m)/(M-m)
    smoothed = cv2.bilateralFilter(depth_image.astype(
        np.float32), 5, 5, 5, cv2.BORDER_ISOLATED)
    smoothed = smoothed * (M-m) + m
    smoothed = smoothed.astype(np.uint16)
    smoothed[mask] = 0
    return smoothed
