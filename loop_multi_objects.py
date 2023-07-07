import os
import numpy as np
import timeit
from utils.constants import OBJECT_NAMES

# pkg_path = "./"
# os.chdir(pkg_path)

start_time = timeit.default_timer() 
grasp_idx_bounds = [0, 1]


# for object_name in OBJECT_NAMES:
#     for grasp_idx in range(*grasp_idx_bounds):
#         os.system(f"python3 run_grasp_evaluation.py --object={object_name} --mode=pickup --grasp_ind {grasp_idx}")

dgn_dataset_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset"
for object_name in os.listdir(dgn_dataset_path):
    os.system(f"python3 collect_static_data.py --object={object_name}")
        
print(f"DONE! You burned {(timeit.default_timer() - start_time)/3600} trees" )