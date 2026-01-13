import trimesh
import numpy as np

def extract_points(path: str, num_points: int):
    mesh = trimesh.load(path, file_type="off", process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded OFF is not a Trimesh object")

    # Ensure triangulation WITHOUT changing shape
    if mesh.faces.shape[1] != 3:
        mesh = mesh.triangulate()

    # Uniform surface sampling
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)

    # Center
    points = points - points.mean(axis=0)

    # Normalize to unit cube (better for voxel grids)
    scale = np.abs(points).max()
    points = points / scale

    return points.astype(np.float32)


def save_pcd(points):
    pcd = trimesh.points.PointCloud(points)
    pcd.export('hello.ply')

