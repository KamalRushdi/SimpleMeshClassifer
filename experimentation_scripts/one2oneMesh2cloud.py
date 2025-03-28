import open3d as o3d
import numpy as np

# Load the mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("raw_2.ply")

# Extract vertices from the mesh
vertices = np.asarray(mesh.vertices)  # Shape: (N, 3)

# Create a point cloud from the vertices
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)

# Optional: Add vertex colors if available
if mesh.has_vertex_colors():
    vertex_colors = np.asarray(mesh.vertex_colors)  # Shape: (N, 3)
    point_cloud.colors = o3d.utility.Vector3dVector(vertex_colors)

# Optional: Add vertex normals if available
if mesh.has_vertex_normals():
    vertex_normals = np.asarray(mesh.vertex_normals)  # Shape: (N, 3)
    point_cloud.normals = o3d.utility.Vector3dVector(vertex_normals)

# Save the point cloud to a .ply file
o3d.io.write_point_cloud("one2one_cloud_2.ply", point_cloud)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])