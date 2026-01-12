import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import trimesh
    import numpy as np
    return np, trimesh


@app.cell
def _(np, trimesh):
    def extract_points(path: str, num_points):
        mesh = trimesh.load(path, file_type="off", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded OFF is not a Trimesh object")

        if mesh.faces.shape[1] != 3:
        # Convert polygon faces → triangles
            mesh = mesh.subdivide().convex_hull
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        points -= points.mean(axis=0)
        points /= np.max(np.linalg.norm(points, axis=1))

        return points

    def save_pcd(points):
        pcd = trimesh.points.PointCloud(points)
        pcd.export('hello.ply')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
