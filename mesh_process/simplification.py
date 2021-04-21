import numpy as np
import open3d as o3d


def simplification(verts, faces, target_number_of_triangles, maximum_error=None, boundary_weight=1.0):
    tri_mesh = o3d.geometry.TriangleMesh()
    tri_mesh.vertices = o3d.utility.Vector3dVector(verts)
    tri_mesh.triangles = o3d.utility.Vector3iVector(faces)
    simplified_mesh = tri_mesh.simplify_quadric_decimation(
        target_number_of_triangles)
    return np.array(simplified_mesh.vertices), np.array(simplified_mesh.triangles)
