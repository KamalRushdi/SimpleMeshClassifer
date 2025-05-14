import os
import trimesh
import numpy as np
from trimesh import util, bounds, transformations
from trimesh.registration import icp

def my_mesh_other(
    mesh,
    other,
    samples: int = 500,
    scale: bool = False,
    icp_first: int = 50,
    **kwargs,
):
    """
    Align a mesh with another mesh or a PointCloud using
    the principal axes of inertia as a starting point which
    is refined by iterative closest point.

    Parameters
    ------------
    mesh : trimesh.Trimesh object
      Mesh to align with other
    other : trimesh.Trimesh or (n, 3) float
      Mesh or points in space
    samples : int
      Number of samples from mesh surface to align
    scale : bool
      Allow scaling in transform
    icp_first : int
      How many ICP iterations for the 9 possible
      combinations of sign flippage
    icp_final : int
      How many ICP iterations for the closest
      candidate from the wider search
    kwargs : dict
      Passed through to `icp`, which passes through to `procrustes`

    Returns
    -----------
    mesh_to_other : (4, 4) float
      Transform to align mesh to the other object
    cost : float
      Average squared distance per point
    """

    def key_points(m, count):
        """
        Return a combination of mesh vertices and surface samples
        with vertices chosen by likelihood to be important
        to registration.
        """
        if len(m.vertices) < (count / 2):
            return np.vstack((m.vertices, m.sample(count - len(m.vertices))))
        else:
            return m.sample(count)

    if not util.is_instance_named(mesh, "Trimesh"):
        raise ValueError("mesh must be Trimesh object!")

    inverse = True
    search = mesh
    # if both are meshes use the smaller one for searching
    if util.is_instance_named(other, "Trimesh"):
        if len(mesh.vertices) > len(other.vertices):
            # do the expensive tree construction on the
            # smaller mesh and query the others points
            search = other
            inverse = False
            points = key_points(m=mesh, count=samples)
            points_mesh = mesh
        else:
            points_mesh = other
            points = key_points(m=other, count=samples)

        if points_mesh.is_volume:
            points_PIT = points_mesh.principal_inertia_transform
        else:
            points_PIT = points_mesh.bounding_box_oriented.principal_inertia_transform

    elif util.is_shape(other, (-1, 3)):
        # case where other is just points
        points = other
        points_PIT = bounds.oriented_bounds(points)[0]
    else:
        raise ValueError("other must be mesh or (n, 3) points!")

    # get the transform that aligns the search mesh principal
    # axes of inertia with the XYZ axis at the origin
    if search.is_volume:
        search_PIT = search.principal_inertia_transform
    else:
        search_PIT = search.bounding_box_oriented.principal_inertia_transform

    # transform that moves the principal axes of inertia
    # of the search mesh to be aligned with the best- guess
    # principal axes of the points
    search_to_points = np.dot(np.linalg.inv(points_PIT), search_PIT)

    # permutations of cube rotations
    # the principal inertia transform has arbitrary sign
    # along the 3 major axis so try all combinations of
    # 180 degree rotations with a quick first ICP pass
    cubes = np.array(
        [
            np.eye(4) * np.append(diag, 1)
            for diag in [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
            ]
        ]
    )

    # loop through permutations and run iterative closest point
    costs = np.ones(len(cubes)) * np.inf
    transforms = [None] * len(cubes)
    centroid = search.centroid

    for i, flip in enumerate(cubes):
        # transform from points to search mesh
        # flipped around the centroid of search
        a_to_b = np.dot(
            transformations.transform_around(flip, centroid),
            np.linalg.inv(search_to_points),
        )

        # run first pass ICP
        matrix, _junk, cost = icp(
            a=points,
            b=search,
            initial=a_to_b,
            max_iterations=int(icp_first),
            scale=scale,
            **kwargs,
        )

        # save transform and costs from ICP
        transforms[i] = matrix
        costs[i] = cost

    # convert to per- point distance average
    cost /= len(points)

    # we picked the smaller mesh to construct the tree
    # on so we may have calculated a transform backwards
    # to save computation, so just invert matrix here
    if inverse:
        mesh_to_other = np.linalg.inv(matrix)
    else:
        mesh_to_other = matrix

    return mesh_to_other, cost


def align_trimesh(input_path, template_path, output_path):
    def preprocess(vertices):
        centered = vertices - np.mean(vertices, axis=0)
        scale = np.max(np.linalg.norm(centered, axis=1))
        return centered / scale

    template_mesh = trimesh.load(template_path)
    sample_mesh = trimesh.load(input_path)
    # Preprocess the input mesh and transform the preprocessed mesh to a trimesh object
    sample_vertices = preprocess(sample_mesh.vertices)
    sample_vertices = trimesh.Trimesh(vertices=sample_vertices, faces=sample_mesh.faces)

    mesh_transform, cost =  my_mesh_other(sample_vertices, template_mesh)

    # Apply the transformation to the sample mesh

    aligned_mesh = sample_vertices.copy()
    aligned_mesh.apply_transform(mesh_transform)

    # Save the aligned mesh under the same name as the input mesh in the output directory
    aligned_mesh.export(output_path)


if __name__ == "__main__":
    # Example usage
    template_path = os.path.expanduser("template_aligned.ply")
    input_dir = os.path.expanduser("~/Desktop/ostieod_cut_plys/train")
    output_dir = os.path.expanduser("~/Desktop/ostieod_cut_plys_aligned_trimesh")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".ply"):
            input_path = os.path.join(input_dir, filename)
            # Call the align_trimesh function for each file
            output_path = os.path.join(output_dir, filename)
            align_trimesh(input_path, template_path, output_path)