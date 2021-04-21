# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from termcolor import colored
import argparse
import cv2
import trimesh
import math
import numpy as np
import sys
import os
from PIL import Image
from render.camera import Camera
from render.color_render import ColorRender
from render.cam_render import CamRender


width = 640
height = 480


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


def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def cycle_rendering_test():
    renderer = ColorRender(width=width, height=height)
    cam = Camera(width=width, height=height)
    # cam.ortho_ratio = width / height
    cam.near = -1
    cam.far = 10

    obj_path = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/neural-tracking/4-refmesh.obj'
    obj_files = [obj_path]

    for i, obj_path in enumerate(obj_files):
        print(obj_path)
        if not os.path.exists(obj_path):
            continue
        mesh = trimesh.load(obj_path)
        vertices = mesh.vertices
        faces = mesh.faces
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        vertices = np.dot(vertices, rot.T)

        bbox_max = vertices.max(0)
        bbox_min = vertices.min(0)

        # notice that original scale is discarded to render with the same size
        vertices -= 0.5 * (bbox_max + bbox_min)[None, :]
        vertices /= bbox_max[1] - bbox_min[1]

        normals = compute_normal(vertices, faces)

        renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)

        self_rot = make_rotate(i, math.radians(-180), 0)
        vertices = np.matmul(vertices, self_rot.T)
        cnt = 0
        for j in range(0, 361, 4):
            cam.center = np.array([0, 0, 0])
            cam.eye = np.array([
                2.0 * math.sin(math.radians(0)), 0, 2.0 *
                math.cos(math.radians(0))
            ]) + cam.center

            self_rot = make_rotate(i, math.radians(-4), 0)
            vertices = np.matmul(vertices, self_rot.T)
            normals = compute_normal(vertices, faces)

            renderer.set_mesh(vertices, faces, 0.5 * normals + 0.5, faces)
            renderer.set_camera(cam)
            renderer.display()

            img = renderer.get_color(0)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            mask = img[..., -1, None]
            img = img[..., :3]
            img = img*mask
            cv2.imshow('render', img)
            cv2.waitKey(1)


def reproject_rendering_test():
    color_img = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/color/000000.jpg'
    color_img = cv2.imread(color_img)
    height,width  = color_img.shape[:2]
    obj_path = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/neural-tracking/0-refmesh.obj'
    intrinsics_file = r'/media/baldr/新加卷/deepdeform_v1_1/train/seq070/intrinsics.txt'
    intric = np.loadtxt(intrinsics_file)
    mesh = trimesh.load(obj_path)
    vertices = mesh.vertices
    faces = mesh.faces
    normals = compute_normal(vertices, faces)

    max_size = max(height, width)
    pad_size = (max_size - height) // 2
    color_img = np.pad(color_img, [[pad_size, pad_size], [0, 0], [0, 0]], mode='constant',
                       constant_values=[[0, 0], [0, 0], [0, 0]])

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

    cv = (dv+1)*(max_size/2)
    cu = (du+1)*(max_size/2)
    cv = np.round(cv).astype(int)
    cu = np.round(cu).astype(int)
    color_img[cv, cu] = 255
    # cv2.imshow('render', color_img)
    # cv2.waitKey(0)

    # rot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # transformed_vertices = np.dot(transformed_vertices, rot.T)
    renderer = ColorRender(width=max_size, height=max_size)
    cam = Camera(width=1.0, height=1.0)
    cam.ortho_ratio = 1.0
    cam.near = -10
    cam.far = 100
    cam.set_projection_matrix(np.eye(4)[:3, :4])

    renderer.set_mesh(transformed_vertices, faces, 0.5 * normals + 0.5, faces)

    renderer.set_camera(cam)
    renderer.display()

    img = renderer.get_color(0)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    # img = np.flip(img, 0)

    mask = img[..., -1, None]
    img = img[..., :3]
    img = img*mask * 255
    mask = mask.astype(bool)
    color_img[mask[..., 0]] = img[mask[..., 0]]
    cv2.imshow('render', color_img)
    cv2.waitKey(0)


# cycle_rendering_test()
reproject_rendering_test()
