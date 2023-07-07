# Copyright (c) 2020 NVIDIA Corporation

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
"""Samples grasps on a test object."""


from graspsampling import sampling, utilities, hands, io

import logging
from graspsampling import collision, visualize
import trimesh
import numpy as np
import os
import transformations
np.random.seed(0)


def sample_grasps(object_filename, object_scale, cls_sampler=sampling.AntipodalSampler, number_of_grasps=50, visualization=False):
    """Sample grasps on a test object."""

    ### Load object
    logging.info("Loading", object_filename)
    object_mesh = utilities.instantiate_mesh(file=object_filename, scale=object_scale)
    print("Extents of loaded mesh:", object_mesh.extents)

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
        sampler = cls_sampler(gripper, object_mesh, number_of_orientations=6) 
    else:
        sampler = cls_sampler(gripper, object_mesh)    
    filtered_poses = sampling.collision_free_grasps(gripper, object_mesh, sampler, number_of_grasps, env_mesh=env_mesh)
    results = {"poses": filtered_poses}

    assert(len(results['poses']) == number_of_grasps)
    
    
    ### Visualize sampled grasps on a test object.
    if visualization:
        scene = visualize.create_scene(object_mesh, 'panda', **results)    # add panda gripper and object to the scene

        if env_mesh is not None:
            env_mesh.visual.face_colors = [250, 0, 0, 255]
            scene.add_geometry(env_mesh)    # add platform to the scene

        scene.show()    
    

    return results  #, gripper, object_mesh, env_mesh


if __name__ == "__main__":

    # fname_object = 'data/objects/banana.obj'
    obj_name = "mustard_bottle"
    object_scale = 1
    # # fname_object = f'/home/baothach/stress_field_prediction/examples/{obj_name}/{obj_name}.obj'
    # fname_object = f'/home/baothach/sim_data/stress_prediction_data/objects/{obj_name}/{obj_name}.stl'
    # data_recording_path = f"/home/baothach/sim_data/stress_prediction_data/objects/{obj_name}/{obj_name}_grasps.h5"
    
    mesh_main_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/objects"
    fname_object = os.path.join(mesh_main_path, obj_name, f"{obj_name}.obj")
    data_recording_path = os.path.join(mesh_main_path, obj_name, f"{obj_name}_grasps.h5")
    
    grasps = sample_grasps(fname_object, object_scale, cls_sampler=sampling.AntipodalSampler, number_of_grasps=3, visualization=True)
    grasps["object_scale"] = object_scale
    print("num grasps:", len(grasps["poses"]))

    # ### Save sampled grasps to a h5 file.
    # h5_writer = io.H5Writer(data_recording_path)
    # h5_writer.write(**grasps)

    # AntipodalSampler  SurfaceApproachSampler