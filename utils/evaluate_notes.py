

"""
6polygon02: grasp 6
6polygon03:
6polygon04: grasp 0
6polygon05:
6polygon06: grasp 0
6polygon07:
6polygon08: grasp 6   
lemon02: 11, 27

Open3D Visualizer camera config:
***6polygon04: 
img_resolution = [1000,1000]
cam_position=[0.0, 0.0, 1.0]
cam_target = [0, 0, 0]
cam_up_vector = [0, 1, 0] 

***6polygon06:
img_resolution = [1000,1000]
cam_position=[0.1, 0.15, 0.1]
cam_target = [0, 0, 0]
cam_up_vector = [0, 0, 1]

***6polygon08:
img_resolution = [1000,1000]
cam_position=[0.2, -0.2, 0.15]
cam_target = [0, 0, 0]
cam_up_vector = [0, 0, 1]

selected_objects = []
selected_objects += \
[f"lemon0{j}" for j in [1,2,3]] + \
[f"strawberry0{j}" for j in [1,2,3]] + \
[f"tomato{j}" for j in [1]] + \
[f"apple{j}" for j in [3]] + \
[f"potato{j}" for j in [3]]
selected_objects += ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol"]
selected_objects += [f"cylinder0{j}" for j in range(1,9)] + [f"box0{j}" for j in range(1,8)] \
                + [f"ellipsoid0{j}" for j in [1,2,3,4]] + [f"sphere0{j}" for j in [1,3,4,6]]
selected_objects += ["box08", "box09", "ellipsoid05"]

"""