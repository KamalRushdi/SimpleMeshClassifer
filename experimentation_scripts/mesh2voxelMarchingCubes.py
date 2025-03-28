import open3d as o3d
import numpy as np
import mcubes  # For Marching Cubes

# Load the mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("raw_2.ply")

# Debug: Check input mesh
print("Number of vertices in the input mesh:", len(mesh.vertices))
print("Number of triangles in the input mesh:", len(mesh.triangles))

# Voxelize the mesh
voxel_size = 0.005  # Adjust this value to control the voxel size
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

# Debug: Check voxel grid
print("Number of voxels in the voxel grid:", len(voxel_grid.get_voxels()))

# Get the bounds of the voxel grid and calculate grid_shape
min_bound = voxel_grid.get_min_bound()
max_bound = voxel_grid.get_max_bound()
grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

# Initialize a 3D boolean array for the voxel grid
voxel_array = np.zeros(grid_shape, dtype=bool)

# Fill the voxel array with True where voxels exist
voxels = voxel_grid.get_voxels()
for voxel in voxels:
    voxel_index = voxel.grid_index
    if all(0 <= idx < dim for idx, dim in zip(voxel_index, grid_shape)):
        voxel_array[voxel_index] = True

# Debug: Check voxel array
print("Voxel array shape:", voxel_array.shape)
print("Number of True values in voxel array:", np.sum(voxel_array))

# Use Marching Cubes to create a surface mesh
vertices, triangles = mcubes.marching_cubes(voxel_array, 0)  # 0 is the isosurface level

# Debug: Check Marching Cubes output
print("Number of vertices from Marching Cubes:", len(vertices))
print("Number of triangles from Marching Cubes:", len(triangles))

# Scale and translate the vertices to match the original mesh
vertices = vertices * voxel_size + min_bound

# Create an Open3D mesh
voxel_mesh = o3d.geometry.TriangleMesh()
voxel_mesh.vertices = o3d.utility.Vector3dVector(vertices)
voxel_mesh.triangles = o3d.utility.Vector3iVector(triangles)

# Save the voxelized mesh to a .ply file
o3d.io.write_triangle_mesh("voxelized_marching_cubes_2.ply", voxel_mesh)

# Optional: Visualize the voxelized mesh
o3d.visualization.draw_geometries([voxel_mesh])