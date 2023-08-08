import open3d
import os
import numpy as np
import pickle
import timeit
import sys
sys.path.append("../")
sys.path.append("../graspsampling-py-defgraspsim")

from graspsampling import sampling, utilities, hands, io, visualize

from utils.miscellaneous_utils import pcd_ize, down_sampling, write_pickle_data, print_color
from utils.mesh_utils import sample_points_from_tet_mesh, open3d_to_trimesh_mesh
from utils.constants import OBJECT_NAMES
from utils.grasp_utils import sample_grasps 


""" 
Extract points on the surface of a point cloud.
"""

def sample_grasps_2(object_mesh, cls_sampler=sampling.AntipodalSampler, number_of_grasps=50, visualization=False):
    """Sample grasps on a test object."""


    # eulers = [0,np.pi/2,0]
    # transform_mat = transformations.euler_matrix(*eulers)
    # object_mesh.apply_transform(transform_mat)

    
    ### Load environment mesh
    # env_mesh = trimesh.creation.box((0.5,0.5,0.005))    # platform
    # min_z_coordinates_object = min(object_mesh.vertices[:, 2])
    # max_z_coordinates_env = max(env_mesh.vertices[:, 2])
    # env_mesh.apply_translation([0,0,min_z_coordinates_object-max_z_coordinates_env])    # shift the platform to the bottom of the object
    env_mesh = None
    
    ### Load panda gripper
    gripper = hands.create_gripper('panda', 0.04)   # 0.04 makes the gripper open, remove it if want to make gripper close


    ### Instantiate and run sampler (with collision checking)
    if cls_sampler==sampling.AntipodalSampler:
        # sampler = cls_sampler(gripper, object_mesh, number_of_orientations=6) 
        sampler = cls_sampler(gripper, object_mesh, 0.0, 4) # Antipodal
    else:
        sampler = cls_sampler(gripper, object_mesh)    
    filtered_poses = sampling.collision_free_grasps(gripper, object_mesh, sampler, number_of_grasps, env_mesh=env_mesh)
    results = {"poses": filtered_poses}

    assert(len(results['poses']) == number_of_grasps)

    ### Visualize sampled grasps on a test object.
    if visualization:
        # scene = visualize.create_scene(object_mesh, 'panda_tube', **results)    # add panda gripper and object to the scene

        # # if env_mesh is not None:
        # #     env_mesh.visual.face_colors = [250, 0, 0, 255]
        # #     scene.add_geometry(env_mesh)    # add platform to the scene

        # scene.show()    
        
        scene = visualize.create_scene(object_mesh, 'panda', **results)
        scene.show()

    return results  #, gripper, object_mesh, env_mesh


static_data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
data_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/6polygon/varying_stiffness"
data_recording_path = os.path.join(data_main_path, "all_6polygon_data")


start_time = timeit.default_timer() 

grasp_idx_bounds = [0, 100]





for object_name in [f"6polygon0{j}" for j in [8]]:    # 1,2,3,4,5,6,7,8




    ### Get static data
    with open(os.path.join(static_data_recording_path, f"{object_name}.pickle"), 'rb') as handle:
        static_data = pickle.load(handle)

    tet_indices = static_data["tet_indices"]
    homo_mats = static_data["homo_mats"]

    

    for grasp_idx in range(1):        
        print(f"{object_name} - grasp {grasp_idx} started. Time passed: {timeit.default_timer() - start_time}")
        
        get_gripper_pc = True
        
        for force_idx in [0]:
            
            # print(f"{object_name} - grasp {grasp_idx} - force {force_idx} started")
            
            file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{grasp_idx}_force_{force_idx}.pickle")
            if not os.path.isfile(file_name):
                # print_color(f"{file_name} not found")
                break   
            
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)

            full_pc = data["object_particle_state"]

            volumetric_pc = full_pc     #sample_points_from_tet_mesh(full_pc[tet_indices], k=2)
            # pcd_volumetric = pcd_ize(volumetric_pc, color=[0,0,0])

            # alpha = 0.03
            # print(f"alpha={alpha:.3f}")
            # mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_volumetric, alpha)
            # pcd_surface = mesh
            # # open3d.visualization.draw_geometries([pcd_volumetric, pcd_surface.translate((0.12,0,0))])

            # object_mesh = open3d_to_trimesh_mesh(mesh)
            # # grasps = sample_grasps_2(object_mesh, cls_sampler=sampling.AntipodalSampler, 
            # #                        number_of_grasps=30, visualization=True)
            
            grasps = sample_grasps(volumetric_pc, cls_sampler="AntipodalSampler", 
                                   number_of_grasps=10, visualization=True, vis_gripper_name='panda',
                                   alpha=0.03)
            # print(grasps["poses"])


        break
    # break

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        