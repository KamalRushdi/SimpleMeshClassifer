import open3d as o3d
import numpy as np

# Load the mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("raw_2.ply")

# Sample points from the mesh to create a point cloud
# num_points is the number of points you want in the point cloud
num_points = 100000
point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

# Save the point cloud to a .ply file
o3d.io.write_point_cloud("cloud_2.ply", point_cloud)

# Optional: Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])