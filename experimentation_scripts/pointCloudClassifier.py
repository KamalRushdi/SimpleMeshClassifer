# import open3d as o3d
# import numpy as np
# from plyfile import PlyData
#
# # Load the PLY file
# ply_data = PlyData.read("raw_48.ply")
#
# # Extract vertices and labels
# vertices = np.vstack([
#     ply_data['vertex']['x'],
#     ply_data['vertex']['y'],
#     ply_data['vertex']['z']
# ]).T  # Shape: (N, 3)
#
# vertex_labels = np.array(ply_data['vertex']['label'])  # Shape: (N,)
#
# # Load the mesh as an Open3D object
# mesh = o3d.io.read_triangle_mesh("raw_48.ply")
#
# # Sample points from the mesh
# num_points = 10000  # Number of points to sample
# point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
#
# # Extract point cloud coordinates
# points = np.asarray(point_cloud.points)
#
# from sklearn.neighbors import NearestNeighbors
#
# # Find nearest neighbors between point cloud and mesh vertices
# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
# distances, indices = nbrs.kneighbors(points)
#
# # Assign labels to the point cloud
# point_labels = vertex_labels[indices.flatten()]
#
# # Find nearest neighbors between mesh vertices and point cloud
# nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
# distances, indices = nbrs.kneighbors(vertices)
#
# # Assign labels to the mesh vertices
# predicted_vertex_labels = point_labels[indices.flatten()]
#
# # save the new vertex labels
# np.savetxt("predicted_vertex_labels.txt", predicted_vertex_labels, fmt='%d')
#
# # Compare predicted labels with original labels
# correct_labels = (predicted_vertex_labels == vertex_labels).sum()
# total_vertices = len(vertex_labels)
# accuracy = correct_labels / total_vertices
#
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"Number of vertices with wrong labels: {total_vertices - correct_labels}")

import open3d as o3d
import numpy as np
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

def load_and_prepare_data(ply_path, num_points=10000):
    # Load the PLY file
    ply_data = PlyData.read(ply_path)

    # Extract vertices and labels
    vertices = np.vstack([
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ]).T  # Shape: (N, 3)

    vertex_labels = np.array(ply_data['vertex']['label'])  # Shape: (N,)

    # Load the mesh as an Open3D object
    mesh = o3d.io.read_triangle_mesh(ply_path)

    # Sample points from the mesh
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

    # Extract point cloud coordinates
    points = np.asarray(point_cloud.points)

    # Find nearest neighbors between point cloud and mesh vertices
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(vertices)
    distances, indices = nbrs.kneighbors(points)

    # Assign labels to the point cloud
    point_labels = vertex_labels[indices.flatten()]

    return points, point_labels




