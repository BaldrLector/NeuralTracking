import os

import torch
import numpy as np
import cv2
import trimesh
import open3d as o3d
import skimage.measure as measure
from cuda_kernels.tsdf_volume import *
from cuda_kernels.depth_image import *
from utils import image_proc
from model.model import DeformNet
from model import dataset
import utils.utils as utils
import utils.viz_utils as viz_utils
import utils.nnutils as nnutils
import utils.line_mesh as line_mesh_utils
import options as opt
import platform
system_type = platform.system()

T_opengl_cv = np.array(
    [[1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]]
)


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


def MarchingCube(tsdf, cell_size, vol_bound_min):
    verts, faces, norms, val = measure.marching_cubes(
        tsdf[:, :, :, 0], level=0)
    verts = verts * cell_size + vol_bound_min
    return verts, faces, norms, val


def inference_deepdeform_dir(root, output_root=None, DEBUG=True, with_rendering=True, min_depth=500, max_depth=1600):
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)
    #####################################################################################################
    # Load model
    #####################################################################################################
    saved_model = opt.saved_model
    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist."
    pretrained_dict = torch.load(saved_model)
    # Construct model
    model = DeformNet().cuda()
    if "chairs_things" in saved_model:
        model.flow_net.load_state_dict(pretrained_dict)
    else:
        if opt.model_module_to_load == "full_model":
            # Load completely model
            model.load_state_dict(pretrained_dict)
        elif opt.model_module_to_load == "only_flow_net":
            # Load only optical flow part
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if "flow_net" in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            print(opt.model_module_to_load,
                  "is not a valid argument (A: 'full_model', B: 'only_flow_net')")
            exit()
    model.eval()

    intrinsics_file = os.path.join(root, 'intrinsics.txt')
    intric = np.loadtxt(intrinsics_file)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    intrinsics = {"fx": fx, "fy": fy,
                  "cx": cx, "cy": cy}

    # Some params for coloring the predicted correspondence confidences
    weight_thr = 0.3
    weight_scale = 1
    # We will overwrite the default value in options.py / settings.py
    opt.use_mask = True
    image_height = opt.image_height
    image_width = opt.image_width
    max_boundary_dist = opt.max_boundary_dist

    color_images = [i for i in os.listdir(
        os.path.join(root, 'color')) if i.endswith('.jpg')]
    depth_images = [i for i in os.listdir(
        os.path.join(root, 'depth')) if i.endswith('.png')]
    color_images = sorted(color_images, key=lambda x: int(x.split('.')[0]))
    depth_images = sorted(depth_images, key=lambda x: int(x.split('.')[0]))
    color_images = [os.path.join(root, 'color', i) for i in color_images]
    depth_images = [os.path.join(root, 'depth', i) for i in depth_images]
    assert len(color_images) == len(depth_images)
    start, end, step = 30, -1, 2

    #####################################################################################################
    # build init reference tsdf volume
    #####################################################################################################
    current_color_file = color_images[start]
    current_depth_file = depth_images[start]
    current_color_image = cv2.imread(
        current_color_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    current_depth_image = cv2.imread(
        current_depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    current_depth_image = cv2.bilateralFilter(
        current_depth_image.astype(np.float32), 5, 1, 1)
    current_depth_image = current_depth_image.astype(np.uint16)
    current_depth_image = current_depth_image * \
        (current_depth_image < max_depth)*(current_depth_image > min_depth)

    vol_resolution = np.ones(3, dtype=np.int)*256
    init_pose, cell_size, tranc_dist, vol_bound_min, vol_bound_max = get_init_tsdf_parmas(
        current_depth_image, intric, vol_resolution)
    inv_pose = np.linalg.inv(init_pose)
    w2d_r = np.ascontiguousarray(init_pose[:3, :3])
    w2d_t = np.ascontiguousarray(init_pose[:3, 3])

    reference_tsdf = np.zeros(shape=[vol_resolution[0], vol_resolution[1],
                                     vol_resolution[2], 2], dtype=np.float32)
    reference_tsdf[..., 0] = 32767
    # integrate depth image to volume
    reference_tsdf = cuda_integrate_tsdf_volume(current_depth_image, intric, w2d_r, w2d_t, tranc_dist,
                                                reference_tsdf, cell_size, vol_bound_min)
    # marching cube
    ref_verts, ref_faces, ref_norms, _ = MarchingCube(
        reference_tsdf, cell_size, vol_bound_min)
    ref_mesh = trimesh.Trimesh(ref_verts, ref_faces, vertex_normals=ref_norms)
    print("Reference mesh")
    ref_mesh.show()

    #####################################################################################################
    # build graph
    #####################################################################################################
    # mesh simplification
    from mesh_process.simplification import simplification
    simpled_verts, simpled_faces = simplification(ref_verts, ref_faces, 5000)
    simpled_mesh = trimesh.Trimesh(simpled_verts, simpled_faces)
    simpled_mesh.remove_degenerate_faces()
    simpled_verts, simpled_faces = simpled_mesh.vertices, simpled_mesh.faces
    print('down-sampled mesh')
    simpled_mesh.show()
    # compute edges
    transformed_vertices = np.dot(
        init_pose[:3, :3], simpled_verts.T).T + init_pose[:3, 3]  # to reference space
    graph_nodes, graphNodesIndices = dataset.sample_node_py_v2(
        transformed_vertices, nodeCoverage=0.05)
    graph_edges = dataset.compute_edges_geodesic_py(
        transformed_vertices, simpled_faces, graphNodesIndices, nMaxNeighbors=8, maxInfluence=0.5)
    reference_graph_nodes = np.dot(
        inv_pose[:3, :3], graph_nodes.T).T + inv_pose[:3, 3]  # [0,1]^3 reference space to the depth space
    edges_lengths = np.sqrt(
        ((graph_nodes[:, None] - graph_nodes[graph_edges])**2).sum(axis=-1))
    max_edge_length = edges_lengths[(graph_edges != -1)].max()
    mask = (graph_edges != -1)
    nodesEdgeDiff = graph_nodes[graph_edges] - graph_nodes[:, np.newaxis, :]
    nodesEdgeLength = np.sqrt((nodesEdgeDiff ** 2).sum(-1))
    graph_edges_weights = np.exp(-nodesEdgeLength) / \
        (np.exp(-nodesEdgeLength) * mask).sum(-1, keepdims=True)
    graph_edges_weights = graph_edges_weights * mask
    graph_clusters = np.array(
        [0]*graph_nodes.shape[0], dtype=np.int)[:, np.newaxis]

    for idx, (tgt_color_file, tgt_depth_file) in \
            enumerate(zip(color_images[start+step:end:step], depth_images[start+step:end:step])):
        print(idx)
        # check TSDF volume and graph
        if DEBUG:
            reference_mesh = o3d.geometry.TriangleMesh()
            reference_mesh.vertices = o3d.utility.Vector3dVector(ref_verts)
            reference_mesh.triangles = o3d.utility.Vector3iVector(ref_faces)
            reference_mesh.compute_vertex_normals()

            bbox_nodes = np.array([[vol_bound_min[0], vol_bound_min[1], vol_bound_min[2]],
                                   [vol_bound_max[0], vol_bound_min[1],
                                       vol_bound_min[2]],
                                   [vol_bound_min[0], vol_bound_max[1],
                                       vol_bound_min[2]],
                                   [vol_bound_min[0], vol_bound_min[1],
                                       vol_bound_max[2]],

                                   [vol_bound_max[0], vol_bound_max[1],
                                       vol_bound_min[2]],
                                   [vol_bound_max[0], vol_bound_min[1],
                                    vol_bound_max[2]],
                                   [vol_bound_min[0], vol_bound_max[1],
                                    vol_bound_max[2]],
                                   [vol_bound_max[0], vol_bound_max[1], vol_bound_max[2]]])
            edges_pairs = [[0, 1], [0, 2],
                           [0, 3], [1, 4],
                           [1, 5], [2, 4],
                           [2, 6], [3, 6],
                           [3, 5], [4, 7],
                           [5, 7], [6, 7],
                           ]
            # bbox
            rendered_bbox_nodes = []
            for node in bbox_nodes:
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                mesh_sphere.translate(node)
                rendered_bbox_nodes.append(mesh_sphere)
            rendered_bbox_nodes = viz_utils.merge_meshes(rendered_bbox_nodes)
            colors = [[0.2, 1.0, 0.2] for i in range(len(edges_pairs))]
            line_mesh = line_mesh_utils.LineMesh(
                bbox_nodes, edges_pairs, colors, radius=0.003)
            line_mesh_geoms = line_mesh.cylinder_segments
            line_mesh_geoms = viz_utils.merge_meshes(line_mesh_geoms)
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[
                -vol_bound_min[0], -vol_bound_min[1], -vol_bound_min[2]])
            # graph
            rendered_graph_nodes = []
            for node in reference_graph_nodes:
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    radius=0.01)
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color([0.0, 0.0, 1.0])
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
            graph_line_mesh = line_mesh_utils.LineMesh(
                reference_graph_nodes, edges_pairs, colors, radius=0.003)
            graph_line_mesh = graph_line_mesh.cylinder_segments
            graph_line_mesh = viz_utils.merge_meshes(graph_line_mesh)

            o3d.visualization.draw_geometries(
                [reference_mesh, rendered_bbox_nodes, line_mesh_geoms, axis_pcd,
                    rendered_graph_nodes, graph_line_mesh])

        point_image = cv2.imread(
            current_depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        point_image[point_image > max_depth] = 0.
        point_image[point_image < min_depth] = 0.
        point_image = dataset.image_proc.backproject_depth(
            point_image, fx, fy, cx, cy)
        point_image = point_image.astype(np.float32)
        pixel_anchors, pixel_weights = cuda_compute_pixel_anchors_euclidean(
            reference_graph_nodes, point_image, max_edge_length)

        #####################################################################################################
        #  prepare input
        #####################################################################################################

        source, _, cropper = dataset.DeformDataset.load_image(
            current_color_file, current_depth_file, intrinsics, image_height, image_width
        )
        pixel_anchors = cropper(pixel_anchors)
        pixel_weights = cropper(pixel_weights)

        # Target color and depth (and boundary mask)
        target, target_boundary_mask, _ = dataset.DeformDataset.load_image(
            tgt_color_file, tgt_depth_file, intrinsics, image_height, image_width, cropper=cropper,
            max_boundary_dist=max_boundary_dist, compute_boundary_mask=True
        )
        num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)

        # Update intrinsics to reflect the crops
        corped_fx, corped_fy, corped_cx, corped_cy = image_proc.modify_intrinsics_due_to_cropping(
            intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
            image_height, image_width, original_h=cropper.h, original_w=cropper.w
        )
        corped_intrinsics = np.zeros((4), dtype=np.float32)
        corped_intrinsics[0] = corped_fx
        corped_intrinsics[1] = corped_fy
        corped_intrinsics[2] = corped_cx
        corped_intrinsics[3] = corped_cy

        #####################################################################################################
        # Predict deformation
        #####################################################################################################

        # Move to device and unsqueeze in the batch dimension (to have batch size 1)
        source_cuda = torch.from_numpy(source).cuda().unsqueeze(0)
        target_cuda = torch.from_numpy(target).cuda().unsqueeze(0)
        target_boundary_mask_cuda = torch.from_numpy(
            target_boundary_mask).cuda().unsqueeze(0)
        graph_nodes_cuda = torch.from_numpy(graph_nodes).cuda().unsqueeze(0)
        graph_edges_cuda = torch.from_numpy(graph_edges).cuda().unsqueeze(0)
        graph_edges_weights_cuda = torch.from_numpy(
            graph_edges_weights).cuda().unsqueeze(0)
        graph_clusters_cuda = torch.from_numpy(
            graph_clusters).cuda().unsqueeze(0)
        pixel_anchors_cuda = torch.from_numpy(
            pixel_anchors).cuda().unsqueeze(0)
        pixel_weights_cuda = torch.from_numpy(
            pixel_weights).cuda().unsqueeze(0)
        intrinsics_cuda = torch.from_numpy(
            corped_intrinsics).cuda().unsqueeze(0)
        num_nodes_cuda = torch.from_numpy(num_nodes).cuda().unsqueeze(0)

        with torch.no_grad():
            model_data = model(
                source_cuda, target_cuda,
                graph_nodes_cuda, graph_edges_cuda, graph_edges_weights_cuda, graph_clusters_cuda,
                pixel_anchors_cuda, pixel_weights_cuda,
                num_nodes_cuda, intrinsics_cuda,
                evaluate=True, split="test"
            )

        # Get some of the results
        rotations_pred = model_data["node_rotations"].view(
            num_nodes, 3, 3).cpu().numpy()
        translations_pred = model_data["node_translations"].view(
            num_nodes, 3).cpu().numpy()

        mask_pred = model_data["mask_pred"]
        assert mask_pred is not None, "Make sure use_mask=True in options.py"
        mask_pred = mask_pred.view(-1, opt.image_height,
                                   opt.image_width).cpu().numpy()

        # Compute mask gt for mask baseline
        _, source_points, valid_source_points, target_matches, \
            valid_target_matches, valid_correspondences, _, \
            _ = model_data["correspondence_info"]

        target_matches = target_matches.view(-1,
                                             opt.image_height, opt.image_width).cpu().numpy()
        valid_source_points = valid_source_points.view(
            -1, opt.image_height, opt.image_width).cpu().numpy()
        valid_target_matches = valid_target_matches.view(
            -1, opt.image_height, opt.image_width).cpu().numpy()
        valid_correspondences = valid_correspondences.view(
            -1, opt.image_height, opt.image_width).cpu().numpy()

        # Delete tensors to free up memory
        del source_cuda
        del target_cuda
        del target_boundary_mask_cuda
        del graph_nodes_cuda
        del graph_edges_cuda
        del graph_edges_weights_cuda
        del graph_clusters_cuda
        del pixel_anchors_cuda
        del pixel_weights_cuda
        del intrinsics_cuda

        if True:
            #####################################################################################################
            # Prepare visulization data
            #####################################################################################################

            #####################################################################################################
            # Source
            #####################################################################################################
            source_flat = np.moveaxis(source, 0, -1).reshape(-1, 6)
            source_points = viz_utils.transform_pointcloud_to_opengl_coords(
                source_flat[..., 3:])
            source_colors = source_flat[..., :3]

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(source_points)
            source_pcd.colors = o3d.utility.Vector3dVector(source_colors)

            # keep only object using the mask
            valid_source_mask = np.moveaxis(
                valid_source_points, 0, -1).reshape(-1).astype(np.bool)
            valid_source_points = source_points[valid_source_mask, :]
            valid_source_colors = source_colors[valid_source_mask, :]
            # source object PointCloud
            source_object_pcd = o3d.geometry.PointCloud()
            source_object_pcd.points = o3d.utility.Vector3dVector(
                valid_source_points)
            source_object_pcd.colors = o3d.utility.Vector3dVector(
                valid_source_colors)

            #####################################################################################################
            # Source warped
            #####################################################################################################
            warped_deform_pred_3d_np = image_proc.warp_deform_3d(
                source, pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred
            )
            source_warped = np.copy(source)
            source_warped[3:, :, :] = warped_deform_pred_3d_np

            # (source) warped RGB-D image
            source_warped = np.moveaxis(source_warped, 0, -1).reshape(-1, 6)
            warped_points = viz_utils.transform_pointcloud_to_opengl_coords(
                source_warped[..., 3:])
            warped_colors = source_warped[..., :3]
            # Filter points at (0, 0, 0)
            warped_points = warped_points[valid_source_mask]
            warped_colors = warped_colors[valid_source_mask]
            # warped PointCloud
            warped_pcd = o3d.geometry.PointCloud()
            warped_pcd.points = o3d.utility.Vector3dVector(warped_points)
            warped_pcd.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries([source_object_pcd, warped_pcd])

            ####################################
            # TARGET #
            ####################################
            # target RGB-D image
            target_flat = np.moveaxis(target, 0, -1).reshape(-1, 6)
            target_points = viz_utils.transform_pointcloud_to_opengl_coords(
                target_flat[..., 3:])
            target_colors = target_flat[..., :3]
            # target PointCloud
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            target_pcd.colors = o3d.utility.Vector3dVector(target_colors)
            o3d.visualization.draw_geometries(
                [target_pcd,  warped_pcd])

        # build data volume
        data_tsdf = np.zeros(shape=[vol_resolution[0], vol_resolution[1],
                                    vol_resolution[2], 2], dtype=np.float32)
        data_tsdf[..., 0] = 32767
        tgt_depth_image = cv2.imread(
            tgt_depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        tgt_depth_image = cv2.bilateralFilter(
            tgt_depth_image.astype(np.float32), 5, 1, 1)
        tgt_depth_image = tgt_depth_image.astype(np.uint16)
        tgt_depth_image = tgt_depth_image * \
            (tgt_depth_image < max_depth)*(tgt_depth_image > min_depth)
        data_tsdf = cuda_integrate_tsdf_volume(tgt_depth_image, intric, w2d_r, w2d_t, tranc_dist,
                                               data_tsdf, cell_size, vol_bound_min)

        # warp reference volume
        # transform the graph embeding to depth space to reference space
        mesh_anchors, mesh_weigths = cuda_compute_mesh_anchors_euclidean(
            reference_graph_nodes, ref_verts, max_edge_length/2)

        if True:
            # deformation at depth space then inv_pose to reference space
            deoformed_refverts = warp_verts(
                ref_verts, init_pose, mesh_anchors, mesh_weigths, graph_nodes, rotations_pred, translations_pred)
            deoformed_refverts = np.dot(
                inv_pose[:3, :3], deoformed_refverts.T).T + inv_pose[:3, 3]  # [0,1]^3 reference space to the depth space

            # chek the data volume and reference volume
            data_verts, data_faces, _, _ = measure.marching_cubes(
                data_tsdf[:, :, :, 0], level=0)
            data_verts = data_verts * cell_size + vol_bound_min
            data_mesh = o3d.geometry.TriangleMesh()
            data_mesh.vertices = o3d.utility.Vector3dVector(data_verts)
            data_mesh.triangles = o3d.utility.Vector3iVector(data_faces)
            data_mesh.compute_vertex_normals()
            data_mesh.paint_uniform_color([0, 0, 1])
            deformed_refmesh = o3d.geometry.TriangleMesh()
            deformed_refmesh.vertices = o3d.utility.Vector3dVector(
                deoformed_refverts)
            deformed_refmesh.triangles = o3d.utility.Vector3iVector(ref_faces)
            deformed_refmesh.compute_vertex_normals()
            deformed_refmesh.paint_uniform_color([1, 0, 0])
            reference_mesh.paint_uniform_color([1, 0.706, 0])
            o3d.visualization.draw_geometries(
                [deformed_refmesh, rendered_bbox_nodes, line_mesh_geoms, axis_pcd, data_mesh])

        # aquire voxel nodes anchors and weight
        voxel_anchors, voxel_weigths = cuda_compute_voxel_anchors(reference_tsdf, reference_graph_nodes,  w2d_r, w2d_t,
                                                                  cell_size, max_edge_length/2, vol_bound_min)
        tgt_depth_image = cv2.imread(
            tgt_depth_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        tgt_depth_image = tgt_depth_image * \
            (tgt_depth_image < max_depth)*(tgt_depth_image > min_depth)
        tgt_color_iamge = cv2.imread(tgt_color_file)
        tgt_color_iamge = tgt_color_iamge * \
            (tgt_depth_image[..., np.newaxis] > 0)
        tgt_color_iamge = tgt_color_iamge/255

        # v1: fuse the depth image into the reference volume
        reference_tsdf, mask = cuda_depth_warp_integrate(tgt_depth_image, intric, w2d_r, w2d_t, cell_size, reference_tsdf, tranc_dist,
                                                         voxel_anchors, voxel_weigths, graph_nodes, rotations_pred, translations_pred,
                                                         vol_bound_min, mask=tgt_color_iamge)

        # v2: fuse reference volume into data volume
        # reference_tsdf = cuda_volume_warp_integrate(reference_tsdf, data_tsdf, voxel_anchors, voxel_weigths, graph_nodes, rotations_pred,
        #                                             translations_pred, cell_size, vol_bound_min)
        ref_verts, ref_faces, ref_norms, _ = MarchingCube(
            reference_tsdf, cell_size, vol_bound_min)
        fused_mesh = trimesh.Trimesh(
            ref_verts, ref_faces, vertex_normals=ref_norms)
        fused_mesh.show()
        # Image.fromarray((mask[..., ::-1]*255).astype(np.uint8)).save(
        #     os.path.join(output_root, '%d_warpmask_cuda.jpg' % (idx)))
        # return

        simpled_verts, simpled_faces = simplification(
            ref_verts, ref_faces, 5000)
        simpled_mesh = trimesh.Trimesh(simpled_verts, simpled_faces)
        simpled_mesh.remove_degenerate_faces()
        simpled_verts, simpled_faces = simpled_mesh.vertices, simpled_mesh.faces

        # compute edges
        transformed_vertices = np.dot(
            init_pose[:3, :3], simpled_verts.T).T + init_pose[:3, 3]  # to reference space
        graph_nodes, graphNodesIndices = dataset.sample_node_py_v2(
            transformed_vertices, nodeCoverage=0.05)
        graph_edges = dataset.compute_edges_geodesic_py(
            transformed_vertices, simpled_faces, graphNodesIndices, nMaxNeighbors=8, maxInfluence=0.5)
        reference_graph_nodes = np.dot(
            inv_pose[:3, :3], graph_nodes.T).T + inv_pose[:3, 3]  # [0,1]^3 reference space to the depth space
        edges_lengths = np.sqrt(
            ((graph_nodes[:, None] - graph_nodes[graph_edges])**2).sum(axis=-1))
        max_edge_length = edges_lengths[(graph_edges != -1)].max()
        mask = (graph_edges != -1)
        nodesEdgeDiff = graph_nodes[graph_edges] - \
            graph_nodes[:, np.newaxis, :]
        nodesEdgeLength = np.sqrt((nodesEdgeDiff ** 2).sum(-1))
        graph_edges_weights = np.exp(-nodesEdgeLength) / \
            (np.exp(-nodesEdgeLength) * mask).sum(-1, keepdims=True)
        graph_edges_weights = graph_edges_weights * mask
        graph_clusters = np.array(
            [0]*graph_nodes.shape[0], dtype=np.int)[:, np.newaxis]


if __name__ == '__main__':
    pass
    if system_type.lower() == 'linux':
        root = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070'
        output_root = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/neural-tracking'
    else:
        root = r'D:\deepdeform_v1_1\test\seq001'
        output_root = r'D:\deepdeform_v1_1\test\seq001/neural-tracking'
    inference_deepdeform_dir(root, output_root)
