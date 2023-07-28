import pickle
import os
import numpy as np
import sys
sys.path.append("../")
from utils.miscellaneous_utils import read_youngs_value_from_urdf
import seaborn as sns
import matplotlib.pyplot as plt

data_path = "/home/baothach/stress_field_prediction/sim_data/stress_prediction_data/dgn_dataset_varying_stiffness/6polygon04"

stiffnesses = []
for grasp_idx in range(0,100):
    filename = os.path.join(data_path, f"soft_body_grasp_{grasp_idx}.urdf")
    stiffnesses.append(float(read_youngs_value_from_urdf(filename)))
stiffnesses = np.array(stiffnesses)/1e4   
# stiffnesses = np.log(stiffnesses)
print("min, max:", min(stiffnesses), max(stiffnesses))
    
# Create a scatter plot
plt.scatter(range(len(stiffnesses)), stiffnesses, color='b', marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Scatter Plot of Data')
plt.show()