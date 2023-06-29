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


from graspsampling import sampling, utilities, hands

import logging
from graspsampling import collision
import trimesh
import numpy as np
# np.random.seed(0)


def test_sampling(cls_sampler=sampling.UniformSampler, number_of_grasps=50):
    
    cls_sampler=sampling.AntipodalSampler
    
    # env_mesh = trimesh.load("/home/baothach/sim_data/stress_prediction_data/objects/bigger_platform.obj")
    env_mesh = trimesh.creation.box((0.5,0.5,0.005))
    
    """Sample grasps on a test object."""
    gripper = hands.create_gripper('panda', 0.04)   # 0.04 makes the gripper open, remove it if want to make gripper close


    # Load object
    # fname_object = 'data/objects/banana.obj'
    obj_name = "rectangle"
    fname_object = f'/home/baothach/stress_field_prediction/examples/{obj_name}/{obj_name}.obj'

    logging.info("Loading", fname_object)
    test_object = utilities.instantiate_mesh(file=fname_object, scale=2)
    
    min_z_coordinates_object = min(test_object.vertices[:, 2])
    max_z_coordinates_env = max(env_mesh.vertices[:, 2])
    env_mesh.visual.face_colors = [250, 0, 0, 255]
    env_mesh.apply_translation([0,0,min_z_coordinates_object-max_z_coordinates_env])
    # # test_object.visual.face_colors = [0, 250, 0, 255]
    # scene = trimesh.Scene([test_object, env_mesh, gripper.mesh])
    # scene.show()
    
    # print("Extents of loaded mesh:", test_object.extents)

    # # Instantiate and run sampler
    sampler = cls_sampler(gripper, test_object)
    
    results = sampler.sample(number_of_grasps)

    # in_collision = collision.in_collision_with_gripper(gripper, test_object, results['poses'])
    # print(in_collision)
    
    filtered_poses = sampling.collision_free_grasps(gripper, test_object, sampler, number_of_grasps, env_mesh=env_mesh)
    results = {"poses": filtered_poses}
    print(len(filtered_poses))

    print(len(results['poses']))

    # print(results)
    # assert('poses' in results)
    assert(len(results['poses']) == number_of_grasps)

    return gripper, test_object, results, env_mesh


if __name__ == "__main__":
    test_sampling()
