import numba
import numba.cuda as cuda
import numpy as np
import math
import cmath
from cuda_kernels.cuda_func import *

GRAPH_K = 4


@cuda.jit()
def cuda_integrate_tsdf_volume_kernel(depth_frame, depth_intr, R, T, trunc_dist, tsdf_volume, cell_size, offset, mask):
    x, y = cuda.grid(2)
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    height, width = depth_frame.shape[:2]
    X_SIZE, Y_SIZE, Z_SIZE = tsdf_volume.shape[:3]

    if x >= X_SIZE or y >= Y_SIZE:
        return
    for z in range(Z_SIZE):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        voxel_depth_frame_x = R[0, 0]*voxel_x + \
            R[0, 1]*voxel_y+R[0, 2]*voxel_z + T[0]
        voxel_depth_frame_y = R[1, 0]*voxel_x + \
            R[1, 1]*voxel_y+R[1, 2]*voxel_z + T[1]
        voxel_depth_frame_z = R[2, 0]*voxel_x + \
            R[2, 1]*voxel_y+R[2, 2]*voxel_z + T[2]

        if (voxel_depth_frame_z < 0):
            continue
        du = int(round(fx * (voxel_depth_frame_x/voxel_depth_frame_z) + cx))
        dv = int(round(fy * (voxel_depth_frame_y/voxel_depth_frame_z) + cy))
        if (du > 0 and du < width and dv > 0 and dv < height):

            depth = depth_frame[dv, du] / 1000.
            psdf = depth - voxel_depth_frame_z

            if (depth > 0 and psdf > -trunc_dist):
                mask[dv, du] = 1

                tsdf = min(1., psdf / trunc_dist)
                tsdf_prev, weight_prev = tsdf_volume[x,
                                                     y, z][0], tsdf_volume[x, y, z][1]
                Wrk = 1
                tsdf_new = (tsdf_prev * weight_prev +
                            Wrk * tsdf) / (weight_prev + Wrk)
                weight_new = min(weight_prev + Wrk, 255)
                tsdf_volume[x, y, z][0], tsdf_volume[x,
                                                     y, z][1] = tsdf_new, weight_new


def cuda_integrate_tsdf_volume(depth_frame, depth_intr, R, T, trunc_dist, tsdf_volume, cell_size, offset=np.zeros(shape=[3, ])):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(tsdf_volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(tsdf_volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    mask = np.zeros_like(depth_frame)
    cuda_integrate_tsdf_volume_kernel[blockspergrid, threadsperblock](depth_frame, depth_intr, R, T, trunc_dist,
                                                                      tsdf_volume, cell_size, offset, mask)
    cuda.synchronize()
    return tsdf_volume, mask


@cuda.jit()
def cuda_compute_voxel_anchors_kernel(voxel_anchors, voxel_weigths, tsdf_graphNodes,
                                      R, T, cell_size, nodeCoverage, offset):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = voxel_anchors.shape[:3]

    if x >= X_SIZE or y >= Y_SIZE:
        return

    for z in range(Z_SIZE):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        voxel_depth_frame_z = R[2, 0]*voxel_x + \
            R[2, 1]*voxel_y+R[2, 2]*voxel_z + T[2]
        if (voxel_depth_frame_z < 0):
            continue

        num_nodes = tsdf_graphNodes.shape[0]
        dist_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.float32)
        index_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.int32)
        for i in range(GRAPH_K):
            dist_array[i] = math.inf
            index_array[i] = -1

        # find nearest nodes ( without order )
        maxidx = 0
        for idx in range(num_nodes):
            new_dist =sqrt_distance(voxel_x, voxel_y, voxel_z,
                                 tsdf_graphNodes[idx, 0],
                                 tsdf_graphNodes[idx, 1],
                                 tsdf_graphNodes[idx, 2])
            if (new_dist < dist_array[maxidx]):
                dist_array[maxidx] = new_dist
                index_array[maxidx] = idx
                # update the maximum distance
                maxidx = 0
                maxdist = dist_array[0]
                for j in range(1,GRAPH_K):
                    if dist_array[j] > maxdist:
                        maxidx = j
                        maxdist = dist_array[j]

        nAnchors = 0
        weightSum = 0
        for i in range(GRAPH_K):
            distance = dist_array[i]
            index = index_array[i]
            if distance > 2 * nodeCoverage:
                continue
            weight = math.exp(-math.pow(distance, 2) /
                              (2*nodeCoverage*nodeCoverage))
            weightSum += weight
            nAnchors += 1

            voxel_anchors[x, y, z, i] = index
            voxel_weigths[x, y, z, i] = weight

        if weightSum > 0:
            for i in range(GRAPH_K):
                voxel_weigths[x, y, z, i] = voxel_weigths[x, y, z, i]/weightSum
        elif nAnchors > 0:
            for i in range(GRAPH_K):
                voxel_weigths[x, y, z, i] = 1 / nAnchors


def cuda_compute_voxel_anchors(reference_tsdf, tsdf_graphNodes,
                               w2d_r, w2d_t, cell_size, nodeCoverage, offset):
    vol_resolution = reference_tsdf.shape[:3]
    voxel_anchors = -np.ones(shape=list(vol_resolution)+[4], dtype=np.int)
    voxel_weigths = np.zeros(shape=list(
        vol_resolution)+[4], dtype=np.float32)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(voxel_anchors.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(voxel_anchors.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda_compute_voxel_anchors_kernel[blockspergrid, threadsperblock](voxel_anchors, voxel_weigths, tsdf_graphNodes,
                                                                      w2d_r, w2d_t, cell_size, nodeCoverage, offset)
    cuda.synchronize()
    return voxel_anchors, voxel_weigths


@cuda.jit()
def cuda_depth_warp_integrate_kernel(depth_frame, depth_intr, R, T, cell_size, tsdf_volume, trunc_dist,
                                     voxel_anchors, voxel_weigths, node_positions, node_rotations, node_translations, offset, norma_map,
                                     mask, ):
    x, y = cuda.grid(2)
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    height, width = depth_frame.shape[:2]
    X_SIZE, Y_SIZE, Z_SIZE = tsdf_volume.shape[:3]
    if x >= X_SIZE or y >= Y_SIZE:
        return

    for z in range(Z_SIZE):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        voxel_depth_frame_x = R[0, 0]*voxel_x + \
            R[0, 1]*voxel_y+R[0, 2]*voxel_z + T[0]
        voxel_depth_frame_y = R[1, 0]*voxel_x + \
            R[1, 1]*voxel_y+R[1, 2]*voxel_z + T[1]
        voxel_depth_frame_z = R[2, 0]*voxel_x + \
            R[2, 1]*voxel_y+R[2, 2]*voxel_z + T[2]

        point_validity = True
        invalid_cnt = 0
        for i in range(GRAPH_K):
            if voxel_anchors[x, y, z, i] == -1:
                invalid_cnt += 1
            if invalid_cnt > 1:
                point_validity = False
                break

        if not point_validity:
            # deoformed_pos_x = voxel_depth_frame_x
            # deoformed_pos_y = voxel_depth_frame_y
            # deoformed_pos_z = voxel_depth_frame_z
            continue
        else:
            deoformed_pos_x = 0.0
            deoformed_pos_y = 0.0
            deoformed_pos_z = 0.0
            for i in range(GRAPH_K):
                if voxel_anchors[x, y, z, i] != -1:
                    new_x, new_y, new_z = warp_point_with_nodes(node_positions[voxel_anchors[x, y, z, i]],
                                                                node_rotations[voxel_anchors[x, y, z, i]],
                                                                node_translations[voxel_anchors[x, y, z, i]],
                                                                voxel_depth_frame_x, voxel_depth_frame_y, voxel_depth_frame_z)
                    deoformed_pos_x += voxel_weigths[x, y, z, i] * new_x
                    deoformed_pos_y += voxel_weigths[x, y, z, i] * new_y
                    deoformed_pos_z += voxel_weigths[x, y, z, i] * new_z

        if (deoformed_pos_z <= 0):
            continue

        du = int(round(fx * (deoformed_pos_x/deoformed_pos_z) + cx))
        dv = int(round(fy * (deoformed_pos_y/deoformed_pos_z) + cy))
        if (du > 0 and du < width and dv > 0 and dv < height):
            depth = depth_frame[dv, du] / 1000.
            psdf = depth - deoformed_pos_z

            view_dir_x, view_dir_y, view_dir_z = normlize(
                deoformed_pos_x, deoformed_pos_y, deoformed_pos_z)
            view_dir_x, view_dir_y, view_dir_z = -view_dir_x, -view_dir_y, -view_dir_z
            dn_x, dn_y, dn_z = norma_map[dv, du,
                                         0], norma_map[dv, du, 1], norma_map[dv, du, 2]
            cosine = dot(dn_x, dn_y, dn_z, view_dir_x, view_dir_y, view_dir_z)

            if depth > 0:
                # mask[dv, du, 0] = dn_x
                # mask[dv, du, 1] = dn_y
                # mask[dv, du, 2] = dn_z
                # mask[dv, du, 3] = view_dir_x
                # mask[dv, du, 4] = view_dir_y
                # mask[dv, du, 5] = view_dir_z
                mask[dv, du] = cosine
            if (depth > 0 and psdf > -trunc_dist and cosine > 0.5):
                tsdf = min(1., psdf / trunc_dist)
                tsdf_prev, weight_prev = tsdf_volume[x,
                                                     y, z][0], tsdf_volume[x, y, z][1]
                Wrk = 1
                tsdf_new = (tsdf_prev * weight_prev +
                            Wrk * tsdf) / (weight_prev + Wrk)
                weight_new = min(weight_prev + Wrk, 255)
                tsdf_volume[x, y, z][0], tsdf_volume[x,
                                                     y, z][1] = tsdf_new, weight_new


def cuda_depth_warp_integrate(depth_frame, depth_intr, w2d_r, w2d_t, cell_size, tsdf_volume, trunc_dist,
                              voxel_anchors, voxel_weigths, node_positions, node_rotations, node_translations, offset, norma_map,
                              mask=None,):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(voxel_anchors.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(voxel_anchors.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if mask is None:
        mask = np.zeros_like(depth_frame, dtype=np.float32)
    else:
        mask = np.copy(mask)
    tsdf_volume = np.copy(tsdf_volume)

    cuda_depth_warp_integrate_kernel[blockspergrid, threadsperblock](depth_frame, depth_intr, w2d_r, w2d_t, cell_size, tsdf_volume, trunc_dist,
                                                                     voxel_anchors, voxel_weigths, node_positions, node_rotations, node_translations,
                                                                     offset, norma_map, mask)
    cuda.synchronize()
    return tsdf_volume, mask


@cuda.jit()
def cuda_warp_volume_kernel(volume, ref_volume, achors, weights, rotations, translations, positions, R, T, cell_size, offset):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = volume.shape[:3]
    if x == 0 or x >= X_SIZE-1 or y == 0 or y >= Y_SIZE-1:
        return
    for z in range(Z_SIZE-1):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        point_validity = True
        for i in range(GRAPH_K):
            if achors[x, y, z, i] == -1:
                point_validity = False

        if not point_validity:
            continue
            # volume[x, y, z, 0] = ref_volume[x, y, z, 0]
            # volume[x, y, z, 1] = ref_volume[x, y, z, 1]
            deoformed_pos_x = x
            deoformed_pos_y = y
            deoformed_pos_z = z
        else:
            deoformed_pos_x = 0.0
            deoformed_pos_y = 0.0
            deoformed_pos_z = 0.0
            for i in range(GRAPH_K):
                new_x, new_y, new_z = warp_point_with_nodes(positions[achors[x, y, z, i]],
                                                            rotations[achors[x, y, z, i]],
                                                            translations[achors[x, y, z, i]],
                                                            voxel_x, voxel_y, voxel_z)
                deoformed_pos_x += weights[x, y, z, i] * new_x
                deoformed_pos_y += weights[x, y, z, i] * new_y
                deoformed_pos_z += weights[x, y, z, i] * new_z
            deoformed_vol_x = (deoformed_pos_x -
                               offset[0]) / cell_size[0] - 0.5
            deoformed_vol_y = (deoformed_pos_y -
                               offset[1]) / cell_size[1] - 0.5
            deoformed_vol_z = (deoformed_pos_z -
                               offset[2]) / cell_size[2] - 0.5

        min_tsdf = tsdf_smallest_tsdf(
            ref_volume, deoformed_vol_x, deoformed_vol_y, deoformed_vol_z)
        sampled_sdf, sapmled_weigth = tsdf_bounded_sample(ref_volume, deoformed_vol_x,
                                                          deoformed_vol_y, deoformed_vol_z, min_tsdf)
        volume[x, y, z, 0] = sampled_sdf
        volume[x, y, z, 1] = sapmled_weigth


def cuda_warp_volume(ref_volume, achors, weights, rotations, translations, positions, R, T, cell_size, offset):
    volume = np.copy(ref_volume)
    volume[..., 0] = 32767
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cuda_warp_volume_kernel[blockspergrid, threadsperblock](
        volume, ref_volume, achors, weights, rotations, translations, positions, R, T, cell_size, offset)
    cuda.synchronize()
    return volume


@cuda.jit()
def cuda_compute_mesh_anchors_euclidean_kernel(Anchors, Weights, graphNodes,  verts, nodeCoverage):
    x = cuda.grid(1)
    if x >= verts.shape[0]:
        return
    vertex_x, vertex_y, vertex_z = verts[x, 0], verts[x, 1], verts[x, 2]
    if vertex_z <= 0:
        return
    num_nodes = graphNodes.shape[0]

    # find nearest Euclidean graph node.
    dist_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.float32)
    index_array = cuda.local.array(shape=GRAPH_K, dtype=numba.types.int32)
    for i in range(GRAPH_K):
        dist_array[i] = math.inf
        index_array[i] = -1

    # find nearest nodes ( without order )
    maxidx = 0
    for idx in range(num_nodes):
        new_dist = sqrt_distance(vertex_x, vertex_y, vertex_z,
                                 graphNodes[idx, 0],
                                 graphNodes[idx, 1],
                                 graphNodes[idx, 2])
        if (new_dist < dist_array[maxidx]):
            dist_array[maxidx] = new_dist
            index_array[maxidx] = idx
            # update the maximum distance
            maxidx = 0
            maxdist = dist_array[0]
            for j in range(1, GRAPH_K):
                if dist_array[j] > maxdist:
                    maxidx = j
                    maxdist = dist_array[j]


    nAnchors = 0
    weightSum = 0
    for i in range(GRAPH_K):
        distance = dist_array[i]
        index = index_array[i]
        if distance > 2*nodeCoverage:
            continue
        weight = math.exp(-math.pow(distance, 2) /
                          (2*nodeCoverage*nodeCoverage))
        weightSum += weight
        nAnchors += 1

        Anchors[x, i] = index
        Weights[x, i] = weight

    if weightSum > 0:
        for i in range(GRAPH_K):
            Weights[x, i] = Weights[x, i]/weightSum
    elif nAnchors > 0:
        for i in range(GRAPH_K):
            Weights[x, i] = 1 / nAnchors


def cuda_compute_mesh_anchors_euclidean(graphNodes,  verts, nodeCoverage):
    num_vertices = verts.shape[0]
    Anchors = -np.ones(shape=[verts.shape[0], 4], dtype=np.int)
    Weights = np.zeros(shape=[verts.shape[0], 4], dtype=np.float32)
    threadsperblock = 256
    blockspergrid = (num_vertices + (threadsperblock - 1)) // threadsperblock
    cuda_compute_mesh_anchors_euclidean_kernel[blockspergrid, threadsperblock](
        Anchors, Weights, graphNodes,  verts, nodeCoverage)
    cuda.synchronize()
    return Anchors, Weights


@cuda.jit()
def cuda_volume_warp_integrate_kernel(ref_volume, data_volume, voxel_anchors, voxel_weigths, node_positions, node_rotations,
                                      node_translations, cell_size, offset):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = ref_volume.shape[:3]
    if x == 0 or x >= X_SIZE-1 or y == 0 or y >= Y_SIZE-1:
        return
    for z in range(Z_SIZE-1):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        point_validity = True
        for i in range(GRAPH_K):
            if voxel_anchors[x, y, z, i] == -1:
                point_validity = False
        if not point_validity:
            deoformed_pos_x = voxel_x
            deoformed_pos_y = voxel_y
            deoformed_pos_z = voxel_z
        else:
            deoformed_pos_x = 0.0
            deoformed_pos_y = 0.0
            deoformed_pos_z = 0.0
            for i in range(GRAPH_K):
                new_x, new_y, new_z = warp_point_with_nodes(node_positions[voxel_anchors[x, y, z, i]],
                                                            node_rotations[voxel_anchors[x, y, z, i]],
                                                            node_translations[voxel_anchors[x, y, z, i]],
                                                            voxel_x, voxel_y, voxel_z)
                deoformed_pos_x += voxel_weigths[x, y, z, i] * new_x
                deoformed_pos_y += voxel_weigths[x, y, z, i] * new_y
                deoformed_pos_z += voxel_weigths[x, y, z, i] * new_z

        # Aquire neighboring voxels within some distance to vote th tsdf value and weigth
        deoformed_vol_x = (deoformed_pos_x - offset[0]) / cell_size[0] - 0.5
        deoformed_vol_y = (deoformed_pos_y - offset[1]) / cell_size[1] - 0.5
        deoformed_vol_z = (deoformed_pos_z - offset[2]) / cell_size[2] - 0.5
        sampled_sdf, sapmled_weigth = tsdf_bilinear_sample(
            data_volume, deoformed_vol_x, deoformed_vol_y, deoformed_vol_z)
        if sampled_sdf != 32767:
            tsdf_prev, weight_prev = ref_volume[x,
                                                y, z][0], ref_volume[x, y, z][1]
            Wrk = 1
            tsdf_new = (tsdf_prev * weight_prev +
                        Wrk * sampled_sdf) / (weight_prev + Wrk)
            weight_new = min(weight_prev + Wrk, 255)
            ref_volume[x, y, z][0], ref_volume[x,
                                               y, z][1] = tsdf_new, weight_new


def cuda_volume_warp_integrate(ref_volume, data_volume, voxel_anchors, voxel_weigths, node_positions, node_rotations,
                               node_translations, cell_size, offset):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ref_volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ref_volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    ref_volume = np.copy(ref_volume)
    cuda_volume_warp_integrate_kernel[blockspergrid, threadsperblock](
        ref_volume, data_volume, voxel_anchors, voxel_weigths, node_positions, node_rotations,
        node_translations, cell_size, offset)
    cuda.synchronize()
    return ref_volume


@cuda.jit()
def cuda_volume_blending_kernel(ref_volume, data_volume, depth_frame,
                                depth_intr, R, T, trunc_dist, cell_size, offset):
    x, y = cuda.grid(2)
    fx, fy, cx, cy = depth_intr[0, 0], depth_intr[1,
                                                  1], depth_intr[0, 2], depth_intr[1, 2]
    height, width = depth_frame.shape[:2]
    X_SIZE, Y_SIZE, Z_SIZE = ref_volume.shape[:3]

    if x >= X_SIZE or y >= Y_SIZE:
        return
    for z in range(Z_SIZE):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        voxel_depth_frame_x = R[0, 0]*voxel_x + \
            R[0, 1]*voxel_y+R[0, 2]*voxel_z + T[0]
        voxel_depth_frame_y = R[1, 0]*voxel_x + \
            R[1, 1]*voxel_y+R[1, 2]*voxel_z + T[1]
        voxel_depth_frame_z = R[2, 0]*voxel_x + \
            R[2, 1]*voxel_y+R[2, 2]*voxel_z + T[2]

        if (voxel_depth_frame_z < 0):
            continue

        du = int(round(fx * (voxel_depth_frame_x/voxel_depth_frame_z) + cx))
        dv = int(round(fy * (voxel_depth_frame_y/voxel_depth_frame_z) + cy))
        e_pxiel = 1.0

        if (du > 0 and du < width and dv > 0 and dv < height):
            depth = depth_frame[dv, du] / 1000.
            psdf = depth - voxel_depth_frame_z
            if (depth > 0 and psdf > -trunc_dist):
                e_pxiel = min(1., abs(psdf) / trunc_dist)

        tsdf_ref, weight_ref = ref_volume[x,
                                          y, z][0], ref_volume[x, y, z][1]
        tsdf_data, weight_data = data_volume[x,
                                             y, z][0], data_volume[x, y, z][1]

        tsdf_new = (tsdf_ref * weight_ref * (1-e_pxiel) +
                    tsdf_data * weight_data) / (weight_ref*(1-e_pxiel) + weight_data)
        weight_new = min(weight_ref*(1-e_pxiel) + weight_data, 255)
        ref_volume[x, y, z][0], ref_volume[x,
                                           y, z][1] = tsdf_new, weight_new


def cuda_volume_blending(ref_volume, data_volume, depth_frame,
                         depth_intr, R, T, trunc_dist, cell_size, offset):
    ref_volume = np.copy(ref_volume)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ref_volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ref_volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda_volume_blending_kernel[blockspergrid, threadsperblock](
        ref_volume, data_volume, depth_frame,
        depth_intr, R, T, trunc_dist, cell_size, offset)
    cuda.synchronize()
    return ref_volume


@cuda.jit()
def cuda_compute_voxel_gradient_kernel(volume, volume_gradient):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = volume.shape[:3]
    if x >= X_SIZE-1 or y >= Y_SIZE-1:
        return
    for z in range(Z_SIZE-1):
        if volume[x, y, z, 0] == 32767 or volume[x, y, z, 1] == 0:
            continue
        if volume[x+1, y, z, 0] != 32767 and volume[x+1, y, z, 1] != 0:
            volume_gradient[x, y, z] = volume[x +
                                              1, y, z, 0] - volume[x, y, z, 0]
        if volume[x, y+1, z, 0] != 32767 and volume[x, y+1, z, 1] != 0:
            volume_gradient[x, y, z] = volume[x, y +
                                              1, z, 0] - volume[x, y, z, 0]
        if volume[x, y, z+1, 0] != 32767 and volume[x, y, z+1, 1] != 0:
            volume_gradient[x, y, z] = volume[x,
                                              y, z+1, 0] - volume[x, y, z, 0]


def cuda_compute_voxel_gradient(ref_volume):
    threadsperblock = (32, 32)
    volume_gradient = np.zeros(
        shape=[*ref_volume.shape[:3], 3], dtype=np.float32)
    blockspergrid_x = math.ceil(ref_volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ref_volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda_compute_voxel_gradient_kernel[blockspergrid, threadsperblock](
        ref_volume, volume_gradient)
    cuda.synchronize()
    return volume_gradient


@cuda.jit()
def cuda_warp_gradient_kernel(ref_volume, volume_gradient, voxel_anchors, voxel_weigths, node_rotations):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = ref_volume.shape[:3]
    if x == 0 or x >= X_SIZE-1 or y == 0 or y >= Y_SIZE-1:
        return
    for z in range(Z_SIZE-1):
        point_validity = True
        for i in range(GRAPH_K):
            if voxel_anchors[x, y, z, i] == -1:
                point_validity = False

        if point_validity:
            deoformed_grad_x = 0.0
            deoformed_grad_y = 0.0
            deoformed_grad_z = 0.0
            for i in range(GRAPH_K):
                new_x, new_y, new_z = warp_normal_with_nodes(node_rotations[voxel_anchors[x, y, z, i]],
                                                             volume_gradient[x,
                                                                             y, z, 0],
                                                             volume_gradient[x,
                                                                             y, z, 1],
                                                             volume_gradient[x, y, z, 2])
                deoformed_grad_x += voxel_weigths[x, y, z, i] * new_x
                deoformed_grad_y += voxel_weigths[x, y, z, i] * new_y
                deoformed_grad_z += voxel_weigths[x, y, z, i] * new_z


def cuda_warp_gradient(ref_volume, volume_gradient, voxel_anchors, voxel_weigths, node_rotations,):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(ref_volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(ref_volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    volume_gradient = np.copy(volume_gradient)
    node_rotations = np.linalg.inv(node_rotations).transpose([0, 2, 1])
    node_rotations = np.ascontiguousarray(node_rotations)
    cuda_warp_gradient_kernel[blockspergrid, threadsperblock](
        ref_volume, volume_gradient, voxel_anchors, voxel_weigths, node_rotations)
    cuda.synchronize()
    return volume_gradient


@cuda.jit()
def cuda_warp_volume_with_gradient_kernel(volume, ref_volume, volme_gradient, data_volume, achors, weights,
                                          rotations, translations, positions, R, T, cell_size, offset):
    x, y = cuda.grid(2)
    X_SIZE, Y_SIZE, Z_SIZE = volume.shape[:3]
    if x == 0 or x >= X_SIZE-1 or y == 0 or y >= Y_SIZE-1:
        return
    for z in range(Z_SIZE-1):
        voxel_x = (x + 0.5) * cell_size[0] + offset[0]
        voxel_y = (y + 0.5) * cell_size[1] + offset[1]
        voxel_z = (z + 0.5) * cell_size[2] + offset[2]

        point_validity = True
        for i in range(GRAPH_K):
            if achors[x, y, z, i] == -1:
                point_validity = False

        if not point_validity:
            volume[x, y, z, 0] = ref_volume[x, y, z, 0]
            volume[x, y, z, 1] = ref_volume[x, y, z, 1]
        else:
            deoformed_pos_x = 0.0
            deoformed_pos_y = 0.0
            deoformed_pos_z = 0.0
            for i in range(GRAPH_K):
                new_x, new_y, new_z = warp_point_with_nodes(positions[achors[x, y, z, i]],
                                                            rotations[achors[x, y, z, i]],
                                                            translations[achors[x, y, z, i]],
                                                            voxel_x, voxel_y, voxel_z)
                deoformed_pos_x += weights[x, y, z, i] * new_x
                deoformed_pos_y += weights[x, y, z, i] * new_y
                deoformed_pos_z += weights[x, y, z, i] * new_z
            deoformed_vol_x = (deoformed_pos_x -
                               offset[0]) / cell_size[0] - 0.5
            deoformed_vol_y = (deoformed_pos_y -
                               offset[1]) / cell_size[1] - 0.5
            deoformed_vol_z = (deoformed_pos_z -
                               offset[2]) / cell_size[2] - 0.5


def cuda_warp_volume_with_gradient(ref_volume, volme_gradient, data_volume, achors, weights,
                                   rotations, translations, positions, R, T, cell_size, offset):
    volume = np.copy(ref_volume)
    volume[..., 0] = 32767
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(volume.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(volume.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    cuda_warp_volume_kernel[blockspergrid, threadsperblock](
        volume, ref_volume, volme_gradient, data_volume, achors, weights,
        rotations, translations, positions, R, T, cell_size, offset)
    cuda.synchronize()
    return volume


if __name__ == '__main__':
    pass
