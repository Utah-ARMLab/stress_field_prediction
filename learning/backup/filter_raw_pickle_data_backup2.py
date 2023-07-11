import open3d
import os
import numpy as np
import pickle
import timeit
import sys
import argparse

sys.path.append("../")
from utils.process_data_utils import *
from utils.miscellaneous_utils import pcd_ize, down_sampling, scalar_to_rgb, print_color
from utils.stress_utils import *


""" 
Filter raw data collected by Isabella. Remove unnecessary information.
"""

dgn_dataset_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset"
raw_data_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/raw_pickle_data"
filtered_data_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/filtered_data"
os.makedirs(filtered_data_path, exist_ok=True)

visualization = True
fruit_names = ["apple", "lemon", "potato", "strawberry", "eggplant", "tomato", "cucumber"]

# for idx, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))[0:]):
# for idx, object_name in enumerate(["sphere04"]):
for idx, file_name in enumerate(sorted(os.listdir(raw_data_path))):

    object_name = os.path.splitext(file_name)[0]

    for k in range(100):    # 100 grasp poses
        
        print("======================")
        print(object_name, idx)
        
        # if not any([fruit_name in object_name for fruit_name in ["potato","eggplant","cucumber"]]):
        #     break
        
        mesh = trimesh.load(os.path.join(dgn_dataset_path, object_name, f"{object_name}.stl"))


        
        file_name = os.path.join(raw_data_path, f"{object_name}_grasp_{k}.pickle")        
        if not os.path.isfile(file_name):
            print(f"{file_name} not found")
            break
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
  
        # Get data for all 50 time steps
        all_object_gripper_combined_particle_states = data["world_pos"]  # shape (50, num_particles, 3)
        all_stresses = data["stress"]  # shape (50, num_particles)
        all_forces = data["force"]  # shape (50, num_particles)
        young_modulus = data["node_mod"]  # shape (num_particles, 3)
        node_type = data["node_type"]  # shape (num_particles,)
        tet_indices = data["cells"]  # shape (num_particles, 4)
        print("all_object_gripper_combined_particle_states.shape:", all_object_gripper_combined_particle_states.shape)
        print("tet_indices.shape:", tet_indices.shape)
        print(max(tet_indices.flatten()), min(tet_indices.flatten()))
        first_object_index = next((i for i, num in enumerate(node_type) if num > 1), None)  # all particles from this index to the end, belong to the deformable object. Other particles belong to the gripper.
        print("first_object_index:", first_object_index)
        tet_indices = tet_indices[np.where(np.min(tet_indices, axis=1) >= first_object_index)[0]] - first_object_index
        
        tfn = data["tfn"]

        # rotation_matrix = trimesh.transformations.euler_matrix(*list(tfn[:3]))
        # rotation_matrix[3,:3] = tfn[3:]
        # mesh.apply_transform(rotation_matrix)
        sampled_pc = trimesh.sample.sample_surface_even(mesh, count=1024)[0]+ np.array([0,0,1.0])# + np.array([tfn[3],tfn[5],1.0])  #+ np.array([0,0,1.0])
        sampled_pc[:, [1, 2]] = sampled_pc[:, [2, 1]]
        if any([fruit_name in object_name for fruit_name in fruit_names]):
            print_color(object_name)
            sampled_pc[:,2] *= -1
                
        for i in range(1):     # 50 time steps
            object_gripper_combined_particle_state = all_object_gripper_combined_particle_states[i]
            stress = all_stresses[i]
            force = all_forces[i]
            
            # print(tet_indices.shape)
            # print(len(set(list(tet_indices.flatten()))))
            # print(node_type)

            
            object_full_pc = object_gripper_combined_particle_state[first_object_index:]
            gripper_pc = object_gripper_combined_particle_state[:first_object_index]
            print("object_full_pc.shape, gripper_pc.shape:", object_full_pc.shape, gripper_pc.shape)
            
            # print(np.mean(object_full_pc, axis=0))
            # print(np.mean(sampled_pc, axis=0))
            # print("offset:", np.mean(sampled_pc, axis=0)-np.mean(object_full_pc, axis=0))
            # print("force:", force)
            # print(tfn)
            
            # pcd_full = pcd_ize(object_full_pc, color=[0,0,0])
            # pcd_gripper = pcd_ize(gripper_pc, color=[1,0,0])
            # pcd_test = pcd_ize(sampled_pc, color=[0,1,0])
            # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # # open3d.visualization.draw_geometries([pcd_full, pcd_test, coor.translate((0,1,0))])
            # open3d.visualization.draw_geometries([pcd_full, pcd_test, pcd_gripper])
            
            print("tet_indices.shape:", tet_indices.shape)
            print(max(tet_indices.flatten()), min(tet_indices.flatten()))

            # # for point in object_full_pc:
            # # # for point in [np.array([10000,10000,10000]]:
            # #     is_inside = is_point_inside_mesh(point, vertices=object_full_pc, tetrahedra=tet_indices)
            # #     print("is_inside", is_inside)
            # #     # assert is_inside
            # start_time = timeit.default_timer()
            # is_inside = is_point_inside_mesh(object_full_pc[:10], vertices=object_full_pc, tetrahedra=tet_indices)
            # print("Time:", timeit.default_timer()-start_time)
            # print(is_inside.shape)
            # print("sum(is_inside):", sum(is_inside))

            vertices=object_full_pc
            tet=tet_indices[0]
            v = [vertices[tet[0]], vertices[tet[1]], vertices[tet[2]], vertices[tet[3]]]
            tetra, origin = Tetrahedron(v)
            test = pointInside(object_full_pc[0], tetra, origin)
            print(test.shape)
            print(tet)
            
            print("==============")
            
            vertices = object_full_pc
            points = object_full_pc[:10]
            start_time = timeit.default_timer()
            
            # v = np.array([[vertices[tetra[0]], vertices[tetra[1]], vertices[tetra[2]], vertices[tetra[3]]] for tetra in tet_indices])
            v = vertices[tet_indices]
            print("Time:", timeit.default_timer()-start_time)
            print("v.shape:", v.shape)
            tetra, origin = Tetrahedron_vectorized(v)            
            test = pointInside_vectorized(points, tetra, origin).squeeze()
            
            
            print(test.shape)
            result = np.any(test, axis=0)
            print(result.shape, sum(result))
            print("Time:", timeit.default_timer()-start_time)
            # print(np.where(test))
            # print(tet_indices[np.where(test)[0]])
        
                        
            break   
        
        
        
        break     
    
    
    
    
    
    