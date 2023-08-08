import os
import numpy as np
import timeit
from utils.constants import OBJECT_NAMES
from utils.miscellaneous_utils import print_color

# pkg_path = "./"
# os.chdir(pkg_path)

start_time = timeit.default_timer() 
grasp_idx_bounds = [0, 100]     # [0, 100]

selected_objects = \
[f"lemon0{j}" for j in [1,2,3]] + \
[f"strawberry0{j}" for j in [1,2,3]] + \
[f"tomato{j}" for j in [1]] + \
[f"apple{j}" for j in [3]] + \
[f"potato{j}" for j in [3]]

for object_name in selected_objects:
# for object_name in [f"apple{j}" for j in [3]]:
    for grasp_idx in range(*grasp_idx_bounds):
        # grasp_idx = 9
        print_color(f"===================== Started {object_name} - grasp {grasp_idx}")
        
        os.system(f"python3 run_grasp_evaluation.py --object={object_name} --grasp_ind {grasp_idx}")
        print_color(f"Done {object_name} - grasp {grasp_idx}\n")
        

# dgn_dataset_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset"
# num_objects = len(os.listdir(dgn_dataset_path))
# # selected_primitive_names = ["6polygon", "8polygon", "cuboid", "cylinder", "sphere", "ellipsoid"]

# # for i, object_name in enumerate(sorted(os.listdir(dgn_dataset_path))):
# # for i, object_name in enumerate(["ellipsoid01"]):
# for i, object_name in enumerate([f"box0{j}" for j in [1,2,3,4,5,6,7]]):    # 1,2,3,4,5,6,7,8
#     # # if "annulus" in object_name[:7]:
#     # #     continue
#     # if not any([prim_name in object_name for prim_name in selected_primitive_names]):   # if object does NOT belong to any of the selected primitives.
#     #     continue

#     print_color(f"Started object {object_name}: {i+1}/{num_objects}")
#     os.system(f"python3 collect_static_data.py --object={object_name}")
        
print_color(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )