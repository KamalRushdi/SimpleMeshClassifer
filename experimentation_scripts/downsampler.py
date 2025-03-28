import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree

# Load the PLY file
ply_data = PlyData.read("raw_3.ply")

# Extract vertices and labels
vertices = np.vstack([
    ply_data['vertex']['x'],
    ply_data['vertex']['y'],
    ply_data['vertex']['z']
]).T  # Shape: (N, 3)

labels = np.array(ply_data['vertex']['label'])  # Shape: (N,)

# Load the mesh as an Open3D object
mesh = o3d.io.read_triangle_mesh("raw_3.ply")

# Print the original number of vertices
print(f"Original number of vertices: {len(mesh.vertices)}")

# Simplify using vertex clustering
voxel_size = 0.00001  # Adjust for desired level of simplification
simplified_mesh = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Quadric  # Use Quadric for better detail preservation
)

# Map labels to the simplified mesh
# Step 1: Create a KDTree for the original mesh vertices
original_vertices = np.asarray(mesh.vertices)
tree = KDTree(original_vertices)

# Step 2: For each vertex in the simplified mesh, find the nearest original vertex
simplified_vertices = np.asarray(simplified_mesh.vertices)
_, indices = tree.query(simplified_vertices)

# Step 3: Assign labels based on the nearest original vertex
simplified_labels = labels[indices]
# save the simplified labels in a text file
np.savetxt("vertex_labels.txt", simplified_labels, fmt='%d')

# Print the simplified number of vertices
print(f"Simplified number of vertices: {len(simplified_mesh.vertices)}")

# Save the simplified mesh with labels
# Create a structured array for the vertices and labels
vertex_data = np.zeros(len(simplified_vertices), dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'f4')
])

# Fill the structured array
vertex_data['x'] = simplified_vertices[:, 0]
vertex_data['y'] = simplified_vertices[:, 1]
vertex_data['z'] = simplified_vertices[:, 2]
vertex_data['label'] = simplified_labels

# Create a PlyElement for the vertices
vertices_element = PlyElement.describe(vertex_data, 'vertex')

# Create a PlyElement for the faces (if the mesh has faces)
if len(simplified_mesh.triangles) > 0:
    faces = np.asarray(simplified_mesh.triangles)
    face_data = np.zeros(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    face_data['vertex_indices'] = faces
    faces_element = PlyElement.describe(face_data, 'face')
else:
    faces_element = None

# Save the PLY file
if faces_element:
    PlyData([vertices_element, faces_element], text=False).write("simplified_mesh_with_labels.ply")
else:
    PlyData([vertices_element], text=False).write("simplified_mesh_with_labels.ply")

print("Simplified mesh with labels saved to 'simplified_mesh_with_labels.ply'")