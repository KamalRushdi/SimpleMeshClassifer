import trimesh

def flip_manually(mesh_path, flip_z = False, flip_x = False):
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Flip the mesh manually
    flipped_mesh = mesh.copy()

    if flip_z:
        flipped_mesh.vertices[:, 2] = -flipped_mesh.vertices[:, 2]

    if flip_x:
        flipped_mesh.vertices[:, 0] = -flipped_mesh.vertices[:, 0]

    # Save the flipped mesh
    flipped_mesh.export(mesh_path)

mesh_path = "../experimentation_scripts/data_samples/train/Melis Kargaci.ply"

# flip_manually(mesh_path, flip_x=True, flip_z=False)