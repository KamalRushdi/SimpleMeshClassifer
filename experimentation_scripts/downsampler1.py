import open3d as o3d
import numpy as np
from plyfile import PlyData

# Load the PLY file
ply_data = PlyData.read("raw_3.ply")

# Extract vertices and labels
vertices = np.vstack([
    ply_data['vertex']['x'],
    ply_data['vertex']['y'],
    ply_data['vertex']['z']
]).T  # Shape: (N, 3)

labels = np.array(ply_data['vertex']['label'])  # Shape: (N,)
# print(vertices[1704])
# print(labels[1704])
# print(vertices[1705])
# print(labels[1705])

# Load the mesh as an Open3D object
mesh = o3d.io.read_triangle_mesh("raw_3.ply")
# Print the original number of vertices
print(f"Original number of vertices: {len(mesh.vertices)}")

# Simplify using vertex clustering
voxel_size = 0.00001  # Adjust for desired level of simplification (almost not simplifying)
simplified_mesh = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Quadric # Use Quadric for better detail preservation
)

# Map labels to the simplified mesh
# Step 1: Create a KDTree for the original mesh vertices
from scipy.spatial import KDTree
original_vertices = np.asarray(mesh.vertices)
tree = KDTree(original_vertices)

# Step 2: For each vertex in the simplified mesh, find the nearest original vertex
simplified_vertices = np.asarray(simplified_mesh.vertices)
_, indices = tree.query(simplified_vertices)

# Step 3: Assign labels based on the nearest original vertex
simplified_labels = labels[indices]
# write it to a text file to see if the mapping works
# np.savetxt("simplified_labels.txt", simplified_labels, fmt='%d')

# Now `simplified_labels` contains the labels for the simplified mesh

# Print the simplified number of vertices
print(f"Simplified number of vertices: {len(simplified_mesh.vertices)}")

# Save the simplified mesh
o3d.io.write_triangle_mesh("simplified_mesh.ply", simplified_mesh)