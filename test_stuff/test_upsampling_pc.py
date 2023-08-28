import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize

def generate_sphere_surface_points(radius, num_points):
    
    sphere_mesh = trimesh.creation.icosphere(radius=radius)

    
    return trimesh.sample.sample_surface_even(sphere_mesh, count=num_points)[0]



sphere_point_cloud = generate_sphere_surface_points(1.0, 1000)
before = pcd_ize(sphere_point_cloud, vis=False, color=[0,0,0])

# Load the point cloud from a file
cloud = PyntCloud.from_instance('OPEN3D', pcd_ize(sphere_point_cloud))
points = cloud.points[['x', 'y', 'z']].values  # Extract XYZ coordinates

# Fit nearest neighbors for efficient interpolation
nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(points)
distances, indices = nbrs.kneighbors(points)

def average_interpolate(neighbors):
    interpolated_point = np.mean(neighbors, axis=0)
    return interpolated_point

upsampled_points = []

for neighbor_indices in indices:
    neighbors = points[neighbor_indices]
    interpolated_point = average_interpolate(neighbors)
    upsampled_points.append(interpolated_point)

upsampled_points = np.vstack((points, upsampled_points))

upsampled_points = np.array(upsampled_points)
print(sphere_point_cloud.shape, upsampled_points.shape)
after = pcd_ize(upsampled_points, vis=False, color=[0,0,0])

open3d.visualization.draw_geometries([before, after.translate((2.0,0,0))])
