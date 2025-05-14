import open3d as o3d

def fill_mesh_large_holes(input_path, output_path, point_count=30000, depth=12):
    print("[INFO] Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh.compute_vertex_normals()

    print("[INFO] Sampling point cloud (uniform)...")
    pcd = mesh.sample_points_uniformly(number_of_points=point_count)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(30)

    print("[INFO] Poisson surface reconstruction...")
    mesh_filled, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    print("[INFO] Cropping with margin...")
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.4, bbox.get_center())
    mesh_filled = mesh_filled.crop(bbox)

    print(f"[INFO] Saving to: {output_path}")
    o3d.io.write_triangle_mesh(output_path, mesh_filled)

    return mesh_filled


fill_mesh_large_holes(
    input_path="data_samples/train/2021 - 21 Pelinsu Özkan (Aktif).ply",
    output_path="mesh_filled.ply",
    point_count=5000,  # More = better, but slower
    depth=14             # Controls detail level
)