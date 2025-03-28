import open3d as o3d
import numpy as np

# Load the mesh
mesh = o3d.io.read_triangle_mesh("simplified_mesh_with_labels.ply")

# Voxelize the mesh
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.02)

# Get the voxel grid as a binary 3D array
voxels = np.zeros((64, 64, 64), dtype=np.float32)  # Example: 64x64x64 grid
for voxel in voxel_grid.get_voxels():
    voxels[voxel.grid_index] = 1  # Mark occupied voxels

from scipy.spatial import KDTree

# Assume you have vertex labels as a NumPy array
# read the vertex labels from a text file
vertex_labels = np.loadtxt("vertex_labels.txt", dtype=np.int32)

# Create a KDTree for the original mesh vertices
vertices = np.asarray(mesh.vertices)
tree = KDTree(vertices)

# Initialize the voxel label grid
voxel_labels = np.zeros((64, 64, 64), dtype=np.int32)

# Iterate over each voxel and assign a label
for voxel in voxel_grid.get_voxels():
    # Get the center of the voxel
    voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

    # Find the nearest vertex to the voxel center
    _, nearest_vertex_index = tree.query(voxel_center)

    # Assign the label of the nearest vertex to the voxel
    voxel_labels[voxel.grid_index] = vertex_labels[nearest_vertex_index]

# save the voxel labels in a text file
np.savetxt("voxel_labels.txt", voxel_labels.flatten(), fmt='%d')
