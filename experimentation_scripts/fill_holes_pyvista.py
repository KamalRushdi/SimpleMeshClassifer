import pyvista as pv

def fill_mesh_holes_pyvista(input_path, output_path, hole_size=1000.0):
    # Load mesh
    print("[INFO] Loading mesh...")
    mesh = pv.read(input_path)
    mesh = mesh.clean(tolerance=1e-6)
    mesh = mesh.triangulate()
    # Fill holes
    print("[INFO] Filling holes...")
    filled = mesh.fill_holes(hole_size)

    # Optional: clean up
    filled = filled.clean()

    # Save the result
    print(f"[INFO] Saving to: {output_path}")
    filled.save(output_path)

    return filled

# Run it
fill_mesh_holes_pyvista(
    input_path="data_samples/train/2021 - 21 Pelinsu Özkan (Aktif).ply",
    output_path="mesh_filled_pyvista.ply",
    hole_size=10000  # Increase if holes aren't being filled
)

