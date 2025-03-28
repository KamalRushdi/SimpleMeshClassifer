import open3d as o3d
import numpy as np

# Load the mesh from a .ply file
mesh = o3d.io.read_triangle_mesh("raw_2.ply")

# Voxelize the mesh
# voxel_size determines the resolution of the voxel grid
voxel_size = 0.005 # Adjust this value to control the voxel size
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

# Visualize the voxel grid
# o3d.visualization.draw_geometries([voxel_grid])

# Optional: Save the voxel grid to a file (e.g., .ply or .obj)
# Note: Open3D does not directly support saving voxel grids, but you can convert it to a point cloud or mesh for saving.

# Get the voxel centers
voxels = voxel_grid.get_voxels()  # Get all voxels in the grid
voxel_centers = np.asarray([voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels])

# Create a unit cube mesh
unit_cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
unit_cube.scale(voxel_size, center=unit_cube.get_center())  # Scale the cube to match the voxel size

# Create a combined mesh for all voxels
voxel_mesh = o3d.geometry.TriangleMesh()

for center in voxel_centers:
    cube = o3d.geometry.TriangleMesh(unit_cube)  # Create a copy of the unit cube
    cube.translate(center)  # Move the cube to the voxel center
    voxel_mesh += cube  # Add the cube to the combined mesh

# Save the voxelized mesh to a .ply file
o3d.io.write_triangle_mesh("voxelized_2.ply", voxel_mesh)


