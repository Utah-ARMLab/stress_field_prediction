import pickle
import os
import numpy as np
import sys
sys.path.append("../")
from utils.miscellaneous_utils import read_youngs_value_from_urdf
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness/box01"

stiffnesses = []
for grasp_idx in range(0,100):
    filename = os.path.join(data_path, f"soft_body_grasp_{grasp_idx}.urdf")
    stiffnesses.append(float(read_youngs_value_from_urdf(filename)))

for i in range(3,8):
    print("==============")
    print(f"1e{i} <= young < 1e{i+1}")
    print(np.count_nonzero((np.array(stiffnesses) >= 10**i) & (np.array(stiffnesses) < 10**(i+1))))


stiffnesses = np.array(stiffnesses)/1e4   
# stiffnesses = np.log(stiffnesses)
print("min, max:", min(stiffnesses), max(stiffnesses))



# Create a scatter plot
plt.scatter(range(len(stiffnesses)), stiffnesses, color='b', marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Scatter Plot of Data')
plt.show()