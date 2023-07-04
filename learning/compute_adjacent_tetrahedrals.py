import numpy as np
import os
import pickle
import meshio
import sys
sys.path.append("../")
from utils.stress_utils import *
from utils.constants import OBJECT_NAMES


adjacent_tetrahedrals_save_path = "/home/baothach/shape_servo_data/stress_field_prediction/adjacent_tetrahedrals"
data_recording_path = "/home/baothach/shape_servo_data/stress_field_prediction/data"
os.makedirs(adjacent_tetrahedrals_save_path, exist_ok=True)


for object_name in OBJECT_NAMES:
    
    file_name = os.path.join(data_recording_path, f"{object_name}_grasp_{0}_force_{1}.pickle")
    if not os.path.isfile(file_name):
        print(f"{file_name} not found")
        break 
    
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
        (tet_indices, tet_stress) = data["tet"]
        tet_indices = np.array(tet_indices).reshape(-1,4)

    save_data = get_adjacent_tetrahedrals_of_vertex(tet_indices)
    with open(os.path.join(adjacent_tetrahedrals_save_path, f"{object_name}.pickle"), 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 






