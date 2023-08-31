import os
import numpy as np
import timeit
import sys
sys.path.append("../")
from utils.miscellaneous_utils import print_color

start_time = timeit.default_timer() 

object_lists = []
object_lists.append([f"6polygon0{j}" for j in [3,4,5,6,7,8]])
object_lists.append([f"6polygon0{j}" for j in [3,5,6,7,8]])

weight_folder_names = ["all_6polygon_open_gripper_new_3", "all_6polygon_open_gripper_new_4"]

for i, object_list in enumerate(object_lists):

    print_color(f"Started {weight_folder_names[i]} ... \n")
    
    object_names_argument = " ".join(object_list)
    os.system(f"python3 trainer_joint_test.py --object_names {object_names_argument} --weight_folder_name {weight_folder_names[i]}")
        
    print_color(f"\nDONE {weight_folder_names[i]}! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees\n" )
    
    
print_color(f"\nALL DONE! You burned {(timeit.default_timer() - start_time)/3600:.2f} trees" )