import numpy as np
import trimesh
from scipy.spatial import cKDTree
import os

def chamfer_distance(A, B):
    tree_A = cKDTree(A)
    tree_B = cKDTree(B)
    dist_A_to_B, _ = tree_A.query(B)
    dist_B_to_A, _ = tree_B.query(A)
    return np.mean(dist_A_to_B**2) + np.mean(dist_B_to_A**2)

def normalize_and_align_to_template(sample_mesh, template_mesh=None, improvement_threshold=.05):
    def preprocess(vertices):
        centered = vertices - np.mean(vertices, axis=0)
        scale = np.max(np.linalg.norm(centered, axis=1))
        return centered / scale

    def compute_pca_rotation(vertices):
        _, _, Vt = np.linalg.svd(vertices, full_matrices=False)
        R = Vt.T
        # Ensure right-handed coordinate system for mirroring
        # if np.linalg.det(R) < 0: # probably not doing shit
        #     R[:, -1] *= -1
        #     #print("🪞 Flipped axes detected — correcting.")
        return R

    # Step 1: preprocess and PCA-align sample
    sample_vertices = preprocess(sample_mesh.vertices)
    R_sample = compute_pca_rotation(sample_vertices)
    sample_aligned = sample_vertices @ R_sample

    if template_mesh is not None:
        # template_vertices = preprocess(template_mesh.vertices) # preprocess template again
        template_vertices = template_mesh.vertices # dont preprocess template again
        R_template = compute_pca_rotation(template_vertices)
        template_aligned = template_vertices @ R_template

        # Step 2: calculate original chamfer (no flip)
        original_error = chamfer_distance(template_aligned, sample_aligned)
        flips = [
            # we go x, z ,y now most important being x
            # [1, 1, 1], # no flip
            [-1, 1, 1], # flip x  (largest threshold should have casuse it results in biggest difference)
            [1, 1, -1], # flip z
            # [1, -1, 1]  # flip y (symmetry flip swap hands sort of (mirror)) no need no way to tell from even a template
        ]

        for flip in flips:
            flipped = sample_aligned * flip
            error = chamfer_distance(template_aligned, flipped)
            improvement = (original_error - error) / original_error
            if improvement > improvement_threshold:
                sample_aligned = flipped
                #print(f"✅ Flip applied: {flip} | Chamfer improvement: {improvement * 100:.2f}%")
                original_error = chamfer_distance(template_aligned, sample_aligned)
            #else:
                #print(
                #    f"⚠️ No significant improvement from flipping ({improvement * 100:.2f}%) on flip {flip}, keeping original")

        y_flip = np.array([1, -1,1])



    aligned_mesh = trimesh.Trimesh(vertices=sample_aligned, faces=sample_mesh.faces)
    return aligned_mesh

#
# template = trimesh.load("2021 - 32 Ömer Albayrak (Aktif).ply")
# canonical_template = normalize_and_align_to_template(template)
# canonical_template.export("template_aligned.ply")

canonical_template = trimesh.load("template_aligned.ply")

src_dir = os.path.expanduser("~/Desktop/new_data/kifoz")
dst_dir = os.path.expanduser("~/Desktop/new_data/aligned_kifoz")

# read obj files in src and convert to ply
for filename in os.listdir(src_dir):
    if filename.lower().endswith('.obj'):
        obj_path = os.path.join(src_dir, filename)
        ply_name = os.path.splitext(filename)[0] + '.ply'
        ply_path = os.path.join(dst_dir, ply_name)

        # Load and convert
        try:
            mesh = trimesh.load(obj_path, file_type='obj')
            aligned_mesh = normalize_and_align_to_template(mesh, template_mesh=canonical_template)
            aligned_mesh.export(ply_path, file_type='ply')
            print(f"Converted: {filename} → {ply_name}")
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")


#
# for filename in os.listdir(src_dir):
#     if filename.lower().endswith('.ply'):
#         obj_path = os.path.join(src_dir, filename)
#         ply_name = os.path.splitext(filename)[0] + '.ply'
#         ply_path = os.path.join(dst_dir, ply_name)
#
#         # Load and convert
#         try:
#             mesh = trimesh.load(obj_path, file_type='ply')
#             aligned_mesh = normalize_and_align_to_template(mesh, template_mesh=canonical_template)
#             aligned_mesh.export(ply_path, file_type='ply')
#             print(f"Converted: {filename} → {ply_name}")
#         except Exception as e:
#             print(f"Failed to convert {filename}: {e}")