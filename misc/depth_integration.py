
'''
TODO: integration depth image with predicted non-rigid deformation
1. build TSDF, the cannonical model
2. predicted deformation
3. warp to cannoical model
3. marching cube to achieve mesh
'''

import os
import cv2
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
import tqdm
import trimesh
from mayavi import mlab

from numba import njit, prange
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0


class TSDF(object):
    def __init__(self, vol_bnds, voxel_size, use_gpu=True):

        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (
            3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 3 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256
        self._vol_dim = np.ceil(
            (self._vol_bnds[:, 1]-self._vol_bnds[:, 0])/self._voxel_size).copy(order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + \
            self._vol_dim*self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(
            order='C').astype(np.float32)
        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0]*self._vol_dim[1]*self._vol_dim[2])
        )

        self._tsdf_vol_cpu = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        # Copy voxel volumes to GPU
        if self.gpu_mode:
            self._tsdf_vol_gpu = cuda.mem_alloc(self._tsdf_vol_cpu.nbytes)
            cuda.memcpy_htod(self._tsdf_vol_gpu, self._tsdf_vol_cpu)
            self._weight_vol_gpu = cuda.mem_alloc(self._weight_vol_cpu.nbytes)
            cuda.memcpy_htod(self._weight_vol_gpu, self._weight_vol_cpu)
            self._color_vol_gpu = cuda.mem_alloc(self._color_vol_cpu.nbytes)
            cuda.memcpy_htod(self._color_vol_gpu, self._color_vol_cpu)

            # Cuda kernel function (C++)
            self._cuda_src_mod = SourceModule("""
        __global__ void integrate(float * tsdf_vol,
                                  float * weight_vol,
                                  float * color_vol,
                                  float * vol_dim,
                                  float * vol_origin,
                                  float * cam_intr,
                                  float * cam_pose,
                                  float * other_params,
                                  float * color_im,
                                  float * depth_im) {
          // Get voxel index
          int gpu_loop_idx = (int) other_params[0];
          int max_threads_per_block = blockDim.x;
          int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
          int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
          int vol_dim_x = (int) vol_dim[0];
          int vol_dim_y = (int) vol_dim[1];
          int vol_dim_z = (int) vol_dim[2];
          if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
              return;
          // Get voxel grid coordinates (note: be careful when casting)
          float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
          float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
          float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
          // Voxel grid coordinates to world coordinates
          float voxel_size = other_params[1];
          float pt_x = vol_origin[0]+voxel_x*voxel_size;
          float pt_y = vol_origin[1]+voxel_y*voxel_size;
          float pt_z = vol_origin[2]+voxel_z*voxel_size;
          // World coordinates to camera coordinates
          float tmp_pt_x = pt_x-cam_pose[0*4+3];
          float tmp_pt_y = pt_y-cam_pose[1*4+3];
          float tmp_pt_z = pt_z-cam_pose[2*4+3];
          float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
          float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
          float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
          // Camera coordinates to image pixels
          int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
          int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
          // Skip if outside view frustum
          int im_h = (int) other_params[2];
          int im_w = (int) other_params[3];
          if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
              return;
          // Skip invalid depth
          float depth_value = depth_im[pixel_y*im_w+pixel_x];
          if (depth_value == 0)
              return;
          // Integrate TSDF
          float trunc_margin = other_params[4];
          float depth_diff = depth_value-cam_pt_z;
          if (depth_diff < -trunc_margin)
              return;
          float dist = fmin(1.0f,depth_diff/trunc_margin);
          float w_old = weight_vol[voxel_idx];
          float obs_weight = other_params[5];
          float w_new = w_old + obs_weight;
          weight_vol[voxel_idx] = w_new;
          tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
          // Integrate color
          float old_color = color_vol[voxel_idx];
          float old_b = floorf(old_color/(256*256));
          float old_g = floorf((old_color-old_b*256*256)/256);
          float old_r = old_color-old_b*256*256-old_g*256;
          float new_color = color_im[pixel_y*im_w+pixel_x];
          float new_b = floorf(new_color/(256*256));
          float new_g = floorf((new_color-new_b*256*256)/256);
          float new_r = new_color-new_b*256*256-new_g*256;
          new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
          new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
          new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
          color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
        }""")

            self._cuda_integrate = self._cuda_src_mod.get_function("integrate")

            # Determine block/grid size on GPU
            gpu_dev = cuda.Device(0)
            self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
            n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) /
                                   float(self._max_gpu_threads_per_block)))
            grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X,
                             int(np.floor(np.cbrt(n_blocks))))
            grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, int(
                np.floor(np.sqrt(n_blocks/grid_dim_x))))
            grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, int(
                np.ceil(float(n_blocks)/float(grid_dim_x*grid_dim_y))))
            self._max_gpu_grid_dim = np.array(
                [grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
            self._n_gpu_loops = int(np.ceil(float(np.prod(
                self._vol_dim))/float(np.prod(self._max_gpu_grid_dim)*self._max_gpu_threads_per_block)))

        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    @staticmethod
    def get_view_frustum(depth_im, cam_intr, cam_pose):
        im_h = depth_im.shape[0]
        im_w = depth_im.shape[1]
        max_depth = np.max(depth_im)
        view_frust_pts = np.array([
            (np.array([0, 0, 0, im_w, im_w])-cam_intr[0, 2])*np.array([0,
                                                                       max_depth, max_depth, max_depth, max_depth])/cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h])-cam_intr[1, 2])*np.array([0,
                                                                       max_depth, max_depth, max_depth, max_depth])/cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth])
        ])
        view_frust_pts = TSDF.rigid_transform(view_frust_pts.T, cam_pose).T
        return view_frust_pts

    @staticmethod
    def rigid_transform(xyz, transform):
        """Applies a rigid transform to an (N, 3) pointcloud.
        """
        xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
        xyz_t_h = np.dot(transform, xyz_h.T).T
        return xyz_t_h[:, :3]

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates.
        """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates.
        """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(
                np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(
                np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
        """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] +
                               obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def init(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2]*self._color_const + color_im[..., 1]*256 + color_im[..., 0])

        # GPU mode: integrate voxel volume (calls CUDA kernel)
        if self.gpu_mode:
            for gpu_loop_idx in range(self._n_gpu_loops):
                self._cuda_integrate(self._tsdf_vol_gpu,
                                     self._weight_vol_gpu,
                                     self._color_vol_gpu,
                                     cuda.InOut(
                                         self._vol_dim.astype(np.float32)),
                                     cuda.InOut(
                                         self._vol_origin.astype(np.float32)),
                                     cuda.InOut(
                                         cam_intr.reshape(-1).astype(np.float32)),
                                     cuda.InOut(
                                         cam_pose.reshape(-1).astype(np.float32)),
                                     cuda.InOut(np.asarray([
                                         gpu_loop_idx,
                                         self._voxel_size,
                                         im_h,
                                         im_w,
                                         self._trunc_margin,
                                         obs_weight
                                     ], np.float32)),
                                     cuda.InOut(
                                         color_im.reshape(-1).astype(np.float32)),
                                     cuda.InOut(
                                         depth_im.reshape(-1).astype(np.float32)),
                                     block=(
                                         self._max_gpu_threads_per_block, 1, 1),
                                     grid=(
                                         int(self._max_gpu_grid_dim[0]),
                                         int(self._max_gpu_grid_dim[1]),
                                         int(self._max_gpu_grid_dim[2]),
                                     )
                                     )
        else:  # CPU mode: integrate voxel volume (vectorized implementation)
            # Convert voxel grid coordinates to pixel coordinates
            cam_pts = self.vox2world(
                self._vol_origin, self.vox_coords, self._voxel_size)
            cam_pts = TSDF.rigid_transform(cam_pts, np.linalg.inv(cam_pose))
            pix_z = cam_pts[:, 2]
            pix = self.cam2pix(cam_pts, cam_intr)
            pix_x, pix_y = pix[:, 0], pix[:, 1]

            # Eliminate pixels outside view frustum
            valid_pix = np.logical_and(pix_x >= 0,
                                       np.logical_and(pix_x < im_w,
                                                      np.logical_and(pix_y >= 0,
                                                                     np.logical_and(pix_y < im_h,
                                                                                    pix_z > 0))))
            depth_val = np.zeros(pix_x.shape)
            depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

            # Integrate TSDF
            depth_diff = depth_val - pix_z
            valid_pts = np.logical_and(
                depth_val > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x,
                                           valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = self.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x,
                               valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x,
                                            valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color-old_b*self._color_const)/256)
            old_r = old_color - old_b*self._color_const - old_g*256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b*self._color_const) / 256)
            new_r = new_color - new_b*self._color_const - new_g*256
            new_b = np.minimum(255., np.round(
                (w_old*old_b + obs_weight*new_b) / w_new))
            new_g = np.minimum(255., np.round(
                (w_old*old_g + obs_weight*new_g) / w_new))
            new_r = np.minimum(255., np.round(
                (w_old*old_r + obs_weight*new_r) / w_new))
            self._color_vol_cpu[valid_vox_x, valid_vox_y,
                                valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

    def non_rigid_integrate(self, color_im, depth_im, cam_intr, cam_pose, nodes_pos, nodes_deform, nodes_trans, obs_weight=1.):
        for xid in range(self._vol_dim[0]):
            for yid in range(self._vol_dim[1]):
                for zid in range(self._vol_dim[2]):
                    vox_center = np.array(
                        [xid, yid, zid]) * self._voxel_size + self._vol_origin
                    # nodes distance
                    dist = np.abs(nodes_pos - vox_center).sum(-1)
                    sorted_idx = np.argsort(dist)
                    sorted_value = dist[sorted_idx]
                   
                    # compute weight
                    

                    # compute index

    def get_volume(self):
        if self.gpu_mode:
            cuda.memcpy_dtoh(self._tsdf_vol_cpu, self._tsdf_vol_gpu)
            cuda.memcpy_dtoh(self._color_vol_cpu, self._color_vol_gpu)
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
        """
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes_lewiner(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
        colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
        """
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(
            tsdf_vol, level=0.5)
        verts_ind = np.round(verts).astype(int)
        # voxel grid coordinates to world coordinates
        # verts = verts*self._voxel_size+self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals/self._color_const)
        colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
        colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


def get_non_rigid_tracking():
    import os
    import open3d as o3d
    import torch
    from utils import image_proc
    from model.model import DeformNet
    from model import dataset
    import utils.utils as utils
    import utils.viz_utils as viz_utils
    import utils.nnutils as nnutils
    import utils.line_mesh as line_mesh_utils
    import options as opt

    split = "train"
    seq_id = 70
    src_id = 0  # source frame
    tgt_id = 100  # target frame
    srt_tgt_str = "5c8446e47ef76a0addc6d0d1_adult0_000000_000100_geodesic_0.05"
    intrinsics = {
        "fx": 575.541,
        "fy": 577.583,
        "cx": 322.523,
        "cy": 238.559
    }
    weight_thr = 0.3
    weight_scale = 1
    opt.use_mask = True
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

    #####################################################################################################
    # Load example dataset
    #####################################################################################################
    # example_dir = os.path.join("example_data" , f"{split}/seq{str(seq_id).zfill(3)}")
    example_dir = os.path.join(
        "/media/baldr/新加卷/deepdeform_v1_1", f"{split}/seq{str(seq_id).zfill(3)}")

    image_height = opt.image_height
    image_width = opt.image_width
    max_boundary_dist = opt.max_boundary_dist

    src_id_str = str(src_id).zfill(6)
    tgt_id_str = str(tgt_id).zfill(6)

    src_color_image_path = os.path.join(
        example_dir, "color",                   src_id_str + ".jpg")
    src_depth_image_path = os.path.join(
        example_dir, "depth",                   src_id_str + ".png")
    tgt_color_image_path = os.path.join(
        example_dir, "color",                   tgt_id_str + ".jpg")
    tgt_depth_image_path = os.path.join(
        example_dir, "depth",                   tgt_id_str + ".png")
    graph_nodes_path = os.path.join(
        example_dir, "graph_nodes",             srt_tgt_str + ".bin")
    graph_edges_path = os.path.join(
        example_dir, "graph_edges",             srt_tgt_str + ".bin")
    graph_edges_weights_path = os.path.join(
        example_dir, "graph_edges_weights",     srt_tgt_str + ".bin")
    graph_clusters_path = os.path.join(
        example_dir, "graph_clusters",          srt_tgt_str + ".bin")
    pixel_anchors_path = os.path.join(
        example_dir, "pixel_anchors",           srt_tgt_str + ".bin")
    pixel_weights_path = os.path.join(
        example_dir, "pixel_weights",           srt_tgt_str + ".bin")

    # Source color and depth
    source, _, cropper = dataset.DeformDataset.load_image(
        src_color_image_path, src_depth_image_path, intrinsics, image_height, image_width
    )

    # Target color and depth (and boundary mask)
    target, target_boundary_mask, _ = dataset.DeformDataset.load_image(
        tgt_color_image_path, tgt_depth_image_path, intrinsics, image_height, image_width, cropper=cropper,
        max_boundary_dist=max_boundary_dist, compute_boundary_mask=True
    )

    # Graph
    graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = dataset.DeformDataset.load_graph_data(
        graph_nodes_path, graph_edges_path, graph_edges_weights_path, None,
        graph_clusters_path, pixel_anchors_path, pixel_weights_path, cropper
    )

    num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)

    # Update intrinsics to reflect the crops
    fx, fy, cx, cy = image_proc.modify_intrinsics_due_to_cropping(
        intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'],
        image_height, image_width, original_h=cropper.h, original_w=cropper.w
    )

    intrinsics = np.zeros((4), dtype=np.float32)
    intrinsics[0] = fx
    intrinsics[1] = fy
    intrinsics[2] = cx
    intrinsics[3] = cy

    source_cuda = torch.from_numpy(source).cuda().unsqueeze(0)
    target_cuda = torch.from_numpy(target).cuda().unsqueeze(0)
    target_boundary_mask_cuda = torch.from_numpy(
        target_boundary_mask).cuda().unsqueeze(0)
    graph_nodes_cuda = torch.from_numpy(graph_nodes).cuda().unsqueeze(0)
    graph_edges_cuda = torch.from_numpy(graph_edges).cuda().unsqueeze(0)
    graph_edges_weights_cuda = torch.from_numpy(
        graph_edges_weights).cuda().unsqueeze(0)
    graph_clusters_cuda = torch.from_numpy(graph_clusters).cuda().unsqueeze(0)
    pixel_anchors_cuda = torch.from_numpy(pixel_anchors).cuda().unsqueeze(0)
    pixel_weights_cuda = torch.from_numpy(pixel_weights).cuda().unsqueeze(0)
    intrinsics_cuda = torch.from_numpy(intrinsics).cuda().unsqueeze(0)

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
    del model

    valid_source_mask = np.moveaxis(
        valid_source_points, 0, -1).reshape(-1).astype(np.bool)
    warped_deform_pred_3d_np = image_proc.warp_deform_3d(
        source, pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred
    )
    source_warped = np.copy(source)
    source_warped[3:, :, :] = warped_deform_pred_3d_np
    source_warped = np.moveaxis(source_warped, 0, -1).reshape(-1, 6)
    warped_points = viz_utils.transform_pointcloud_to_opengl_coords(
        source_warped[..., 3:])
    warped_colors = source_warped[..., :3]
    warped_points = warped_points[valid_source_mask]
    warped_colors = warped_colors[valid_source_mask]

    # target RGB-D image
    target_flat = np.moveaxis(target, 0, -1).reshape(-1, 6)
    target_points = viz_utils.transform_pointcloud_to_opengl_coords(
        target_flat[..., 3:])
    target_colors = target_flat[..., :3]

    return warped_points, warped_colors, target_points, target_colors, \
        pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred


if __name__ == "__main__":
    color_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/color/000000.jpg'
    depth_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/depth/000000.png'
    mask_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/mask/000000.png'

    color_image = cv2.imread(
        color_image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth_image = cv2.imread(
        depth_image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    mask_image = cv2.imread(mask_image)
    color_image = color_image*(mask_image == 255)
    depth_image = depth_image*(mask_image[..., 0] == 255) / 1000
    depth_image[depth_image > 2.5] = 0
    cam_intr = np.loadtxt(
        "/media/baldr/新加卷/deepdeform_v1_1/train/seq070/intrinsics.txt", delimiter=' ')[:3, :3]
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:] = np.inf
    cam_pose = np.eye(4)
    view_frust_pts = TSDF.get_view_frustum(depth_image, cam_intr, cam_pose)
    vol_bnds[:, 0] = np.minimum(
        vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:, 1] = np.maximum(
        vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    vol_resolution = 128
    voxel_size = np.min((vol_bnds[:, 1] - vol_bnds[:, 0])/vol_resolution)
    tsdf_vol = TSDF(vol_bnds, voxel_size=voxel_size)
    tsdf_vol.init(color_image, depth_image,
                  cam_intr, cam_pose, obs_weight=1.)
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_colors=colors[:, ::-1])
    mesh.show()

    warped_points, warped_colors, target_points, target_colors, \
        pixel_anchors, pixel_weights, graph_nodes, rotations_pred, translations_pred = get_non_rigid_tracking()

    # add warped meshes
    target_color_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/color/000100.jpg'
    target_depth_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/depth/000100.png'
    target_mask_image = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/mask/000100.png'
    target_color_image = cv2.imread(
        target_color_image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    target_depth_image = cv2.imread(
        target_depth_image, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    target_mask_image = cv2.imread(target_mask_image)
    target_color_image = target_color_image*(target_mask_image == 255)
    target_depth_image = target_depth_image * \
        (target_mask_image[..., 0] == 255) / 1000
    target_depth_image[target_depth_image > 2.5] = 0

    tsdf_vol.non_rigid_integrate(color_image, depth_image,
                                 cam_intr, cam_pose, graph_nodes, rotations_pred, translations_pred, obs_weight=1.)
    
    # import open3d as o3d
    # volume = o3d.pipelines.integration.ScalableTSDFVolume(
    #     voxel_length=4.0 / 512.0,
    #     sdf_trunc=0.1,
    #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    # color = o3d.io.read_image(color_image)
    # depth = o3d.io.read_image(depth_image)
    # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #     color, depth, depth_trunc=2.5, convert_rgb_to_intensity=False)
    # volume.integrate(
    #     rgbd,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
    #     np.linalg.inv(np.eye(4)))
    # mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh],)
