import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import os

Kp = 0.008

grasp_orientation = "height_grasp"   # options: width_grasp, height_grasp

save_path = f"recorded_forces/{grasp_orientation}"
with open(os.path.join(save_path, f"Kp={Kp}.pickle"), 'rb') as handle:
    data = np.array(pickle.load(handle))


plt.figure(figsize=(16, 10), dpi=80)

# xs = np.arange(data.shape[0])
plt.plot(data[:,0], label='Desired Left', color=[1,0,0,1], linewidth=8)
plt.plot(data[:,1], label='Desired Right', color=[1,0,0,1],linewidth=8)
plt.plot(data[:,2], label='Measured Left', color=[0,1,0,1],linewidth=1)
plt.plot(data[:,3], label='Measured Right', color=[0,0,1,1],linewidth=1)



plt.title(f'Desired and Measured Force Over Time (Kp={Kp})', fontsize=40)
plt.xlabel('Frame Number', fontsize=40)
plt.ylabel('Force (N)', fontsize=40)
plt.legend(prop={'size': 24})
plt.xticks(fontsize=24)
plt.yticks(fontsize=24, rotation=0)
# # plt.subplots_adjust(bottom=0.15)
# plt.ylim([0,6])
plt.ylim([0,4])

plt.savefig(f'../visualization/figures/{grasp_orientation}/measured_vs_desired_force_Kp={Kp}.png')  #, bbox_inches='tight', pad_inches=0.1)
plt.show()
