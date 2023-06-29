import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the object mesh
# obj_name = "banana"
# fname_object = f'/home/baothach/stress_field_prediction/examples/{obj_name}/{obj_name}.obj'
fname_object = "/home/baothach/sim_data/Custom/Custom_mesh/multi_cylinder_10kPa/cylinder_1.stl"
# fname_object = 'data/objects/banana.obj'

object_mesh = trimesh.load_mesh(fname_object)

# Calculate the bounding box of the object mesh
bbox = object_mesh.bounds

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the object mesh
ax.plot_trisurf(object_mesh.vertices[:, 0], object_mesh.vertices[:, 1], object_mesh.vertices[:, 2],
                triangles=object_mesh.faces, color=(0.8, 0.8, 0.8, 0.5), edgecolor='k')

# Define the vertices of the bounding box
vertices = np.array([
    bbox[0],
    [bbox[1][0], bbox[0][1], bbox[0][2]],
    [bbox[1][0], bbox[1][1], bbox[0][2]],
    [bbox[0][0], bbox[1][1], bbox[0][2]],
    [bbox[0][0], bbox[0][1], bbox[1][2]],
    [bbox[1][0], bbox[0][1], bbox[1][2]],
    bbox[1],
    [bbox[0][0], bbox[1][1], bbox[1][2]]
])

# Define the edges of the bounding box
edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]
])

ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'r-')
for edge in edges:
    ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], 'r-')

# # Set the plot limits
# ax.set_xlim(bbox[0][0], bbox[1][0])
# ax.set_ylim(bbox[0][1], bbox[1][1])
# ax.set_zlim(bbox[0][2], bbox[1][2])

# Label the object mesh and bounding box
ax.text(bbox[0][0], bbox[0][1], bbox[0][2], 'Object Mesh', color='red')
ax.text(bbox[1][0], bbox[1][1], bbox[1][2], 'Bounding Box', color='blue')

# Show the plot
plt.show()
