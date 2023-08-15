import os
import numpy as np
import timeit
from utils.constants import OBJECT_NAMES
from utils.miscellaneous_utils import print_color

# pkg_path = "./"
# os.chdir(pkg_path)

start_time = timeit.default_timer() 
grasp_idx_bounds = range(0, 20)     # range(0, 100)

selected_objects = []
# selected_objects += \
# [f"lemon0{j}" for j in [1,2,3]] + \
# [f"strawberry0{j}" for j in [1,2,3]] + \
# [f"tomato{j}" for j in [1]] + \
# [f"apple{j}" for j in [3]] + \
# [f"potato{j}" for j in [3]]
# selected_objects += ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol"]
# selected_objects += [f"cylinder0{j}" for j in range(1,9)] + [f"box0{j}" for j in range(1,8)] \
#                 + [f"ellipsoid0{j}" for j in [1,2,3,4]] + [f"sphere0{j}" for j in [1,3,4,6]]
# selected_objects += ["box08", "box09", "ellipsoid05"]
selected_objects += ["mustard_bottle", "strawberry02", "lemon02"]  # , "mustard_bottle", "strawberry02", "lemon02"


# for object_name in selected_objects:
#     for grasp_idx in grasp_idx_bounds:
        
#         grasp_idx = np.random.choice(100, size=None, replace=False)

#         print_color(f"===================== Started {object_name} - grasp {grasp_idx}")
        
#         os.system(f"python3 run_grasp_evaluation.py --object={object_name} --grasp_ind {grasp_idx}")
#         print_color(f"Done {object_name} - grasp {grasp_idx}\n")
        
        
##################################******************************************

num_objects = len(selected_objects)
# for i, object_name in enumerate([f"box0{j}" for j in [1,2,3,4,5,6,7]]):    # 1,2,3,4,5,6,7,8

for i, object_name in enumerate(selected_objects):

    print_color(f"Started object {object_name}: {i+1}/{num_objects}")
    os.system(f"python3 collect_static_data.py --object={object_name}")
        
print_color(f"DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )