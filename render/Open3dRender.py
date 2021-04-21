from pytorch3d.renderer.cameras import OpenGLOrthographicCameras, SfMOrthographicCameras
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (DirectionalLights, MeshRasterizer, MeshRenderer,
                                OpenGLPerspectiveCameras, RasterizationSettings,
                                 look_at_view_transform, Textures, TexturesAtlas, TexturesUV,SoftPhongShader)
from pytorch3d.structures import Meshes,Volumes
from pytorch3d.transforms import Transform3d
from pytorch3d.io.obj_io import load_obj

import pickle
import numpy as np
from PIL import Image
import torch,cv2
import torch.nn as nn
import matplotlib.pyplot as plt
import trimesh

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm

class Render(nn.Module):
    def __init__(self,size, device='cuda:0', device_ids=['0', ]):
        super(Render, self).__init__()
        self.device = device
        R, T = look_at_view_transform(10, 0, 0)
        self.size=size
        self.cameras = OpenGLOrthographicCameras(R=R, T=T, device=self.device)
        self.raster_settings = RasterizationSettings(image_size=size,
                                                     blur_radius=0.0, faces_per_pixel=1,
                                                     bin_size=0, cull_backfaces=True)
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings)

        self.lights = DirectionalLights(
            ambient_color=((1, 1, 1),),
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
            direction=((1, 1, 1),), device=self.device)
        self.shader = SoftPhongShader(
            cameras=self.cameras, lights=self.lights, device=self.device)

        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer, shader=self.shader)

    def rendering_vertices(self, vertices, colors, triangles):
        textures = Textures(verts_rgb=colors)
        meshes = Meshes(verts=vertices, faces=triangles.long(),
                        textures=textures)
        v = meshes.verts_padded().detach()
        prj_v = self.cameras.transform_points(v)
        z_view = self.cameras.get_world_to_view_transform().transform_points(v)
        prj_v[:, :, :2] = -prj_v[:, :, :2]
        prj_v[:, :, 2] = z_view[:, :, 2]
        fragments = self.renderer.rasterizer(meshes)
        zbuf = fragments.zbuf
        rendered_img = self.renderer.shader(fragments, meshes)
        rendered_img = rendered_img[..., :3] * (zbuf != -1)

        rendered_img = rendered_img.permute(0, 3, 1, 2).contiguous()
        zbuf = zbuf.permute(0, 3, 1, 2).contiguous()

        rendered_img = torch.flip(rendered_img, [2])
        zbuf = torch.flip(zbuf, [2])
        return rendered_img, zbuf, prj_v

def visualize_reconstructed_mesh(render,color_img,vertices,faces,intric,normals=None):
    height,width  = color_img.shape[:2]
    if normals is None:
        normals = compute_normal(vertices, faces)
    colors = 0.5 * normals + 0.5

    max_size = max(height,width)
    pad_dim_eq1 = height < width
    if pad_dim_eq1:
        pad_size = (max_size -height)//2
        color_img = np.pad(color_img, [[pad_size, pad_size], [0, 0], [0, 0]], mode='constant',
                           constant_values=[[0, 0], [0, 0], [0, 0]])
    else:
        pad_size = (max_size -width)//2
        color_img = np.pad(color_img,[[0,0],[pad_size,pad_size],[0,0]], mode='constant',constant_values=[[0,0],[0,0],[0,0]])

    transformed_vertices = np.copy(vertices)
    fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    du = transformed_vertices[:,
                              0]/transformed_vertices[:, 2] * fx + cx
    dv = transformed_vertices[:,
                              1]/transformed_vertices[:, 2] * fy + cy + pad_size
    dv= dv / (max_size/2)-1
    du =du / (max_size/2)-1
    transformed_vertices[:, 0] = du
    transformed_vertices[:, 1] = dv
    vertices = torch.from_numpy(transformed_vertices).float().cuda().unsqueeze(0)
    colors =  torch.from_numpy(colors).float().cuda().unsqueeze(0)
    triangles = torch.from_numpy(faces.astype(np.int32)).long().cuda().unsqueeze(0)

    with torch.no_grad():
        rendered_img, zbuf, _ = render.rendering_vertices(vertices,colors,triangles)
        rendered_img = rendered_img.permute(0,2,3,1).contiguous()
        zbuf = zbuf.permute(0,2,3,1).contiguous()
        rendered_img = rendered_img.squeeze().cpu().numpy()
        zbuf = zbuf.squeeze().cpu().numpy()
        color_img[zbuf!=-1] = rendered_img[zbuf!=-1][...,::-1] * 255
    if pad_dim_eq1:
        color_img = color_img[pad_size:-pad_size]
    else:
        color_img = color_img[:,pad_size:-pad_size]
    del vertices
    del colors
    del triangles
    torch.cuda.empty_cache()
    return color_img

if __name__ == '__main__':
    color_img = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/color/000000.jpg'
    color_img = cv2.imread(color_img)
    height,width  = color_img.shape[:2]
    obj_path = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/neural-tracking/0-refmesh.obj'
    intrinsics_file = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/intrinsics.txt'
    intric = np.loadtxt(intrinsics_file)
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces
    render= Render(max(height,width)).cuda()
    color_img = visualize_reconstructed_mesh(render, color_img, vertices, faces, intric)
    cv2.imshow('render', color_img)
    cv2.waitKey(0)

    # normals = compute_normal(vertices, faces)
    # colors = 0.5 * normals + 0.5
    # transformed_vertices = np.copy(vertices)
    # fx, fy, cx, cy = intric[0, 0], intric[1, 1], intric[0, 2], intric[1, 2]
    # du = transformed_vertices[:,
    #                           0]/transformed_vertices[:, 2] * fx + cx
    # dv = transformed_vertices[:,
    #                           1]/transformed_vertices[:, 2] * fy + cy + 80
    # dv= dv / (640/2)-1
    # du =du / (640/2)-1
    # transformed_vertices[:, 0] = du
    # transformed_vertices[:, 1] = dv
    #
    # vertices = torch.from_numpy(transformed_vertices).float().cuda().unsqueeze(0)
    # colors =  torch.from_numpy(colors).float().cuda().unsqueeze(0)
    # triangles = torch.from_numpy(faces.astype(np.int32)).long().cuda().unsqueeze(0)
    #
    # with torch.no_grad():
    #     rendered_img, zbuf, _ = render(vertices,colors,triangles)
    #     rendered_img = rendered_img.permute(0,2,3,1).contiguous()
    #     zbuf = zbuf.permute(0,2,3,1).contiguous()
    #     rendered_img = rendered_img.squeeze().cpu().numpy()
    #     zbuf = zbuf.squeeze().cpu().numpy()
    #     color_img[zbuf!=-1] = rendered_img[zbuf!=-1][...,::-1] * 255
    #     color_img = color_img[80:-80]
    #     cv2.imshow('render',color_img)
    #     cv2.waitKey(0)
