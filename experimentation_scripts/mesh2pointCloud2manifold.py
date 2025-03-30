import numpy as np
import open3d as o3d

input_path = "raw_50.ply"
output_path = "manifold_50.ply"
mesh = o3d.io.read_triangle_mesh(input_path)

# Ensure normals are computed (if missing)
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Convert mesh vertices and normals into a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.normals = mesh.vertex_normals

# visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

distances = pcd.compute_nearest_neighbor_distance()
avg_distance = np.mean(distances)
radius = 3 * avg_distance

bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius * 2, radius * 5]))

# visualize the mesh
o3d.visualization.draw_geometries([bpa_mesh])