import matplotlib.pyplot as plt
import numpy as np
import pickle5 as pickle
import os


weight_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset/weights/run2(conv1d)"
with open(os.path.join(weight_path, "saved_losses_accuracies.pickle"), 'rb') as handle:
    data = pickle.load(handle)
    
train_stress_losses = data['train_stress_losses']
test_stress_losses = data["test_stress_losses"]
train_accuracies = data["train_accuracies"]
test_accuracies = data["test_accuracies"]

# Create Figure 1
fig1, ax1 = plt.subplots()
ax1.plot(train_stress_losses, color=[1,0,0,1], linewidth=1, label='Train')
ax1.plot(test_stress_losses, color=[0,0,1,1], linewidth=1, label='Test')
ax1.set_title('Stress Loss')
ax1.legend()

# Create Figure 2
fig2, ax2 = plt.subplots()
ax2.plot(train_accuracies, color=[1,0,0,1], linewidth=1, label='Train')
ax2.plot(test_accuracies, color=[0,0,1,1], linewidth=1, label='Test')
ax2.set_title('Occupancy Accuracy')
ax2.legend()


# # xs = np.arange(data.shape[0])
# plt.plot(data[:,0], label='Desired Left', color=[1,0,0,1], linewidth=8)
# plt.plot(data[:,1], label='Desired Right', color=[1,0,0,1],linewidth=8)
# plt.plot(data[:,2], label='Measured Left', color=[0,1,0,1],linewidth=1)
# plt.plot(data[:,3], label='Measured Right', color=[0,0,1,1],linewidth=1)



# plt.title(f'Desired and Measured Force Over Time (Kp={Kp})', fontsize=40)
# plt.xlabel('Frame Number', fontsize=40)
# plt.ylabel('Force (N)', fontsize=40)
# plt.legend(prop={'size': 24})
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24, rotation=0)
# # plt.ylim([0,6])
# # plt.ylim([0,4])

# # plt.savefig(f'../visualization/figures/random_tests/measured_vs_desired_force_Kp={Kp}.png')
# plt.savefig(f'/home/baothach/Downloads/test.png')
plt.show()
