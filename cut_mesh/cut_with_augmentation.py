import pyvista as pv
import pymeshfix
import os
import trimesh
import numpy as np
import pandas as pd

#
# def fill_mesh_holes_pyvista(input_mesh, hole_size=1000.0):
#     input_mesh_vertexes = input_mesh.vertices
#     input_mesh_faces = input_mesh.faces
#     clean_vertexes, clean_faces = pymeshfix.clean_from_arrays(input_mesh_vertexes, input_mesh_faces)
#
#     # Convert clean_faces to PyVista format (flattened with leading 3s)
#     n_faces = clean_faces.shape[0]
#     faces_pv = np.hstack([np.full((n_faces, 1), 3), clean_faces]).astype(np.int32)
#     faces_pv = faces_pv.flatten()
#
#     # Create a PyVista mesh
#     mesh = pv.PolyData(clean_vertexes, faces_pv)
#
#     mesh = mesh.clean(tolerance=1e-6)
#     mesh = mesh.triangulate()
#     # Fill holes
#     filled = mesh.fill_holes(hole_size)
#
#     # Optional: clean up
#     filled = filled.clean()
#     return filled
# #
# # if __name__ == "__main__":
# #     # try the new fill_mesh_holes_pyvista
# #     mesh = trimesh.load("../experimentation_scripts/data_samples/train/2021 - 21 Pelinsu Özkan (Aktif).ply")
# #     filled = fill_mesh_holes_pyvista(mesh, hole_size=1000.0)
# #     filled.save("filled.ply")
# #
# #     mesh = trimesh.load("../experimentation_scripts/data_samples/train/2021 - 32 Ömer Albayrak (Aktif).ply")
# #     filled = fill_mesh_holes_pyvista(mesh, hole_size=1000.0)
# #     filled.save("filled2.ply")

def processMesh(input_mesh : trimesh.Trimesh):
    # fill the mesh using pyvista
    # mesh = fill_mesh_holes_pyvista(input_mesh, hole_size=10000.0)
    # convert the mesh to trimesh
    mesh = input_mesh
    center_z = (mesh.bounds[0][2] + mesh.bounds[1][2]) / 2
    # cut the mesh from with a plane on center_z
    plane_normal = [0, 0, 1]
    plane_origin = [0, 0, center_z]

    # Slice the mesh with the plane
    sliced_meshes = mesh.slice_plane(plane_origin, plane_normal)

    # sliced_meshes.export(f"{sliced_mesh_path}/{"".join(base_name.split(".")[:-1])}.ply")
    return sliced_meshes

N_SLICES = 21
def sliceMesh(mesh):
    # cut the mesh across the Z axis alonge the Y axis
    jump = abs(mesh.bounds[0][0] - mesh.bounds[1][0]) / N_SLICES
    initial = mesh.bounds[0][0]
    plane_normal = [1, 0, 0]
    plane_origin = [initial, 0, 0]
    plane_negative_normal = [-1, 0, 0]

    # new_center_x = 0
    # Slice the mesh with the plane
    slices = []
    for i in range (N_SLICES):
        sliced_meshes = mesh.slice_plane(plane_origin, plane_normal)
        plane_origin[0] = plane_origin[0] + jump
        sliced_meshes = sliced_meshes.slice_plane(plane_origin, plane_negative_normal)
        slices.append(sliced_meshes)
        # sliced_meshes.export(f"{i}.ply")
    # remove the last N_SLICES - 17
    for i in range(N_SLICES - 17):
        slices.pop()
    return slices

from scipy.spatial.transform import Rotation as R
def rotate_normal_around_axis(normal, origin, axis, angle_degrees):
    axis = axis / np.linalg.norm(axis)  # ensure unit axis
    angle_rad = np.deg2rad(angle_degrees)
    rot = R.from_rotvec(angle_rad * axis)
    new_normal = rot.apply(normal)
    return new_normal

N_PATCHES = 10
def radial_slice(mesh : trimesh.Trimesh):
    # cut the mesh radially
    # print(len(mesh.vertices))
    initial_z = mesh.bounds[0][2]
    # print(f"Center X: {initial_z}")
    plane_normal = [0, 0, 1]
    plane_negative_normal = [0, 0, -1]
    # TODO TRY to take abs for (mesh.bounds[0][1] + mesh.bounds[1][1])
    plane_origin = np.array([0, (mesh.bounds[0][1] + mesh.bounds[1][1])/ 2, initial_z])
    axis = np.array([1, 0, 0])      # rotate around X
    # last_normal = rotate_normal_around_axis(np.array(plane_normal), np.array(plane_origin), axis, 170)
    # last_slice =  mesh.slice_plane(plane_origin, last_normal)
    # last_slice.export("last_slice.ply")


    patches_list = []

    for i in range(N_PATCHES):
        sliced_patches = mesh.slice_plane(plane_origin, plane_normal)
        plane_normal = rotate_normal_around_axis(np.array(plane_normal), np.array(plane_origin), axis, 180 / N_PATCHES)
        plane_negative_normal = rotate_normal_around_axis(np.array(plane_negative_normal), np.array(plane_origin), axis, 180 / N_PATCHES)
        sliced_patches = sliced_patches.slice_plane(plane_origin, plane_negative_normal)
        # sliced_patches.export(f"radial_slice_{i}.ply")
        patches_list.append(sliced_patches)

    return patches_list

from skspatial.objects import Plane
def get_patch_normal_scikit_spatial(patch: trimesh.Trimesh):
    """
    Compute the normal of a mesh patch using scikit-spatial.

    Parameters:
    -----------
    patch : trimesh.Trimesh
        A mesh patch for which to estimate the normal

    Returns:
    --------
    numpy.ndarray
        The estimated normal vector for the patch
    """
    if patch is None:
        print("Warning: Patch is None")
        return np.array([0.0, 0.0, 1.0])  # Default normal

    # Check if the patch has enough vertices (need at least 3 points for a proper plane)
    if len(patch.vertices) < 3:
        print(f"Warning: Patch has only {len(patch.vertices)} vertices, need at least 3 for reliable PCA")
        if len(patch.vertices) == 0:
            return np.array([0.0, 0.0, 1.0])  # Default normal if no vertices
        elif len(patch.vertices) == 1 or len(patch.vertices) == 2:
            # Single point - use vertex normal if available, otherwise default
            avg_normal = np.mean(patch.vertex_normals, axis=0)
            return avg_normal / np.linalg.norm(avg_normal)

    # Get vertices as points for PCA
    points = patch.vertices
    plane = Plane.best_fit(points)

    # Get the normal of the plane
    normal = plane.normal

    # Ensure the normal has unit length
    normal = normal / np.linalg.norm(normal)
    # Optional: Make sure the normal points "outward"
    # We can use the average vertex normal as a reference for consistent orientation
    if len(patch.vertex_normals) > 0:
        avg_vertex_normal = np.mean(patch.vertex_normals, axis=0)
        avg_vertex_normal = avg_vertex_normal / np.linalg.norm(avg_vertex_normal)

        # If the computed normal points in the opposite direction of the average vertex normal,
        # flip it to maintain consistent orientation
        if np.dot(normal, avg_vertex_normal) < 0:
            normal = -normal

    return normal

def get_features(input_mesh, export_output=False, export_path=None):
    # print(meshPath)
    back_slice = processMesh(input_mesh)
    vertical_slices = sliceMesh(back_slice)

    # # collection of geometries
    geometries_original = []
    geometries_features = []

    all_normals = []
    for slice in vertical_slices:
        if slice is None:
            print("Warning vertical slice is None")
            continue
        patches = radial_slice(slice)
        for patch in patches:
            # make the original geometry
            if export_output:
                geometry_original = trimesh.Trimesh(vertices=patch.vertices, faces=patch.faces, process=False)
                geometry_original.vertex_normals = patch.vertex_normals
                geometries_original.append(geometry_original)

            patch_normal = get_patch_normal_scikit_spatial(patch)
            if export_output:
                # make the features geometry
                geometry_features = trimesh.Trimesh(vertices=patch.vertices, faces=patch.faces, process=False)
                like_array = np.full_like(geometry_features.vertex_normals, patch_normal)
                geometry_features.vertex_normals = like_array
                # geometry_features.vertex_normals = np.tile(patch_normal, (geometry_features.vertices.shape[0], 1))
                geometries_features.append(geometry_features)

            all_normals.append(patch_normal)

    if export_output:
        # export the original geometries as one unit
        merged = trimesh.util.concatenate(geometries_original)
        merged.export(export_path + "_original.ply")
        # export the features geometries as one unit
        merged_features = trimesh.util.concatenate(geometries_features)
        merged_features.export(export_path + "_features.ply")

    return np.array(all_normals)
# visualize the mesh
# mesh = trimesh.load_mesh("../experimentation_scripts/data_samples/train/filled_meshes/2021 - 21 Pelinsu Özkan (Aktif).ply")
# normals = get_features(mesh, export_output=True, export_path="filled")

def rotate_pointcloud_3d(pc, max_degrees=20):
    """ Apply a random 3D rotation """
    max_angle = np.deg2rad(max_degrees)  # Converts 30 degrees to radians
    angles = np.random.uniform(-max_angle, max_angle, size=3)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]),  np.cos(angles[0])]
    ])
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]),  np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    return pc @ R.T

def get_label(ply_path, labels_dict, class_label_mapper):
    # Extract the label from the file name
    sample_name = os.path.basename(ply_path)[0:-4]
    class_label = labels_dict.get(sample_name, None)
    class_label = class_label_mapper.get(class_label, None)
    return class_label

labels_df = pd.read_csv("../labels.csv")

label_mapper = {"K1": 0, "K2": 1, "K3": 2, "K4": 3, "K5": 4, "K3B": 2, "K6" : 5}
# Filter the DataFrame to keep only rows with valid class labels
filtered_df = labels_df[labels_df['class'].isin(label_mapper)]
#print(filtered_df.shape)
labels_dict = dict(zip(filtered_df['name'], filtered_df['class']))
print(labels_dict)
#print(get_label('2021 - 21 Pelinsu Özkan (Aktif)', labels_dict, label_mapper))
# meshPathes = [path for path in os.listdir("../experimentation_scripts/data_samples/train") if path.endswith(".ply")]
# print(meshPathes)
import glob
data_path = glob.glob("../experimentation_scripts/data_samples/train/filled_meshes/*.ply")

#
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MeshesDataset(Dataset):
    def __init__(self, file_paths, labels_dict, label_mapper, training=False):
        self.file_paths = file_paths
        self.training = training  # Flag to control augmentation
        self.labels_dict = labels_dict
        self.label_mapper = label_mapper
        self.cache = {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        ply_path = self.file_paths[idx]
        # simple cache
        if ply_path in self.cache:
            # load the mesh
            mesh = self.cache[ply_path]
        else:
            mesh = trimesh.load(ply_path)
            self.cache[ply_path] = mesh
        # rotate the mesh here
        if self.training:
            # mesh = rotate_mesh(mesh)
            pass
        label = get_label(ply_path, labels_dict, label_mapper)
        normals = get_features(mesh)

        # augmented_mesh = rotate_mesh(ply_path)
        # get the label from the file name
        # and extract the features
        # augmetnation
        return (
            torch.tensor(normals.flatten(), dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            self.file_paths[idx]
        )
#
from sklearn.model_selection import train_test_split
train_paths, test_paths = train_test_split(data_path, test_size=0.2, random_state=42, stratify=[
    label_mapper[labels_dict[os.path.basename(p)[:-4]]] for p in data_path
])


# dataset = MeshesDataset(data_path, labels_dict, training=True)
# #print(len(dataset))

train_dataset = MeshesDataset(train_paths, labels_dict, label_mapper, training=True)
#print(len(train_dataset[0][0]))
test_dataset = MeshesDataset(test_paths, labels_dict, label_mapper, training=False)
# print(len(dataset[0][0]))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleANN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# add imbalance weights
# Model setup
input_dim = train_dataset[0][0].shape[0]
print(input_dim)
unique_values = set(label_mapper.values())
num_classes = len(unique_values)
print(num_classes)
model = SimpleANN(input_dim, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
# add weight decay and augmentation
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, targets, _ in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == targets).sum().item()
        total_train += targets.size(0)

    train_accuracy = correct_train / total_train

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == targets).sum().item()
            total_test += targets.size(0)

    test_accuracy = correct_test / total_test
    avg_train_loss = total_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1}: "
          f"Train Loss = {avg_train_loss:.4f}, "
          f"Train Acc = {train_accuracy:.4f}, "
          f"Test Loss = {avg_test_loss:.4f}, "
          f"Test Acc = {test_accuracy:.4f}")