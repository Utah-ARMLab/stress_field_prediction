import numpy as np
import mcubes
import trimesh
import open3d
from matplotlib import pyplot
from pyntcloud import PyntCloud
import pandas as pd
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize

def plot_voxel(voxel, img_path=None, voxel_res=(32,32,32)):
    fig = pyplot.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(voxel) != 0:
        ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('voxel')

    if voxel_res is not None:
        ax.set_xlim3d(0, voxel_res[0])
        ax.set_ylim3d(0, voxel_res[1])
        ax.set_zlim3d(0, voxel_res[2])
    
    pyplot.show()
    if img_path is not None:
        pyplot.savefig(img_path)

def generate_spherical_volumetric_point_cloud(radius, num_points):
    # Generate random points within a cube centered at the origin
    points = np.random.rand(num_points, 3) - 0.5
    
    # Normalize the points to be within the unit sphere
    norms = np.linalg.norm(points, axis=1)
    points_inside_sphere = points[norms <= 0.5]
    
    # Scale the points to the desired sphere radius
    scaled_points = points_inside_sphere * 2 * radius
    
    return scaled_points

def voxel_vis(voxel_grid):
    import matplotlib.pyplot as plt
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of occupied voxels
    x, y, z = np.where(voxel_grid)

    # Plot the occupied voxels as points
    ax.scatter(x, y, z, c='b', marker='o', s=10)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()

sphere_point_cloud = generate_spherical_volumetric_point_cloud(1.0, 100000)
# # pcd_ize(sphere_point_cloud, vis=True)
# voxel_grid = open3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ize(sphere_point_cloud),
#                                                             voxel_size=0.1)
# # open3d.visualization.draw_geometries([voxel_grid])

# # Get the voxel data
# # print(voxel_grid.voxel_size)
# voxel_data = voxel_grid.get_voxels()
# print(voxel_data[1].grid_index)

# voxels = []
# for i in range(len(voxel_data)):
#     grid_idx = voxel_data[i].grid_index
#     voxels.append(grid_idx)

# voxels = np.array(voxels)
# print(max(voxels[:,0]), min(voxels[:,0]))
# print(max(voxels[:,1]), min(voxels[:,1]))
# print(max(voxels[:,2]), min(voxels[:,2]))

# voxel_array = np.zeros((max(voxels[:,0])+1, max(voxels[:,1])+1, max(voxels[:,2])+1))
# for i in range(len(voxel_data)):
#     voxel_array[voxel_data[i].grid_index] = 1

# # Choose voxel size and calculate grid size
# voxel_size = 0.1  # Edge length of each voxel cube
# max_point = 2*np.max(np.abs(sphere_point_cloud))
# grid_size = int(np.ceil(2 * max_point / voxel_size)) + 1  # Calculate grid size and add 1

# # Initialize voxel grid
# voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

# # Iterate over points and fill the voxel grid
# for point in sphere_point_cloud:
#     voxel_coordinates = np.floor((point + max_point) / voxel_size).astype(int)
#     if (0 <= voxel_coordinates).all() and (voxel_coordinates < grid_size).all():
#         voxel_grid[voxel_coordinates[0], voxel_coordinates[1], voxel_coordinates[2]] = True


cloud = PyntCloud.from_instance('OPEN3D', pcd_ize(sphere_point_cloud))
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)
voxelgrid = cloud.structures[voxelgrid_id]

# Determine the dimensions of the voxel grid
n_x = 64
n_y = 64
n_z = 64

print(voxelgrid.voxel_x)

voxel_array = np.zeros((n_x, n_y, n_z))

for (x,y,z) in zip(voxelgrid.voxel_x, voxelgrid.voxel_y, voxelgrid.voxel_z):
    voxel_array[x,y,z] = 1

# # Convert the voxel data to a 3D NumPy array
# voxel_array = np.zeros((n_x, n_y, n_z), dtype=bool)
# for x, y, z in voxelgrid.voxel_data:
#     voxel_array[x, y, z] = True
# print(voxel_array.shape)

# # Sphere parameters
# sphere_radius = 10
# sphere_center = np.array([15, 15, 15])

# # Create a 3D grid
# grid_size = 30
# grid = np.zeros((grid_size, grid_size, grid_size))

# # Iterate over voxels and determine if they are part of the sphere
# for x in range(grid_size):
#     for y in range(grid_size):
#         for z in range(grid_size):
#             voxel_position = np.array([x, y, z])
#             distance = np.linalg.norm(voxel_position - sphere_center)
            
#             if distance <= sphere_radius:
#                 grid[x, y, z] = 1  # Mark as part of the sphere

# voxel_vis(voxel_grid)

u = voxel_array

# binary_mask = u <= 0
# u[binary_mask] = 0
# u[~binary_mask] = 1
# u = mcubes.smooth(u)

# plot_voxel(u)


# # Print the shape of the voxel array
# print(voxel_array.shape)

# smoothed_sphere = mcubes.smooth(voxel_array)

# Extract the 0-isosurface
vertices, triangles = mcubes.marching_cubes(u, 0)
print(vertices.shape)

mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
point_cloud = trimesh.points.PointCloud(sphere_point_cloud)
# mesh = trimesh.smoothing.filter_laplacian(mesh, lamb=1, iterations=1)

# mesh.show()

scene = trimesh.Scene([mesh,point_cloud])
scene.show()