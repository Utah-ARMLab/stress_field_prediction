from grasp_planner import *
import os
import pickle
import matplotlib.pyplot as plt
import argparse

'''
Do gradient descent on the provided grasp pose to maximize/minimize deformation
'''

def plot_curve(xs, ys_1, x_label="epochs", y_label="losses", label_1="train_losses", title="grasp planner BCE train losses", path="/home/shinghei/Downloads/figures"):
    fig, ax = plt.subplots()
    ax.plot(xs, ys_1, label=label_1)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='training for grasp planner')
    parser.add_argument('--seed', type=int, default=2021, help='random seed for sampling')
    parser.add_argument('--init_grasp_idx', type=int, default=0, help='idx of the grasp to use as initial condition for gradient descent in the samples')
    parser.add_argument('--is_max_deformation', type=int, default=0, help='0: minimize deformation, 1: maximize deformation')
    parser.add_argument('--object_name', type=str, default="6polygon04", help='object name')
    parser.add_argument('--grasp_planner_data_root_path', type=str, default="/home/shinghei/Downloads/grasp_planner_data", help='path to the root folder containing grasp planner training data')
    parser.add_argument('--stress_net_model_path', type=str, default="/home/shinghei/Downloads/shinghei_stuff/all_6polygon_open_gripper/epoch_193", help='path to stress net weights')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs for training grasp planner')
    parser.add_argument('--force', type=int, default=15, help='force exerted on the object')
    parser.add_argument('--young_modulus', type=int, default=5, help='stiffness of the object')
    

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_grasp_idx = args.init_grasp_idx
    maximize_deformation = args.is_max_deformation==1

    object_name = args.object_name
    grasp_planner_data_root_path = args.grasp_planner_data_root_path
    grasp_planner_training_data_path = os.path.join(grasp_planner_data_root_path, f"{object_name}/grasp_planner_training_data.pickle")
    
    model_path = args.stress_net_model_path
    
    if maximize_deformation:
        grasp_planner_optimized_pose_path = os.path.join(grasp_planner_data_root_path, object_name, f"optimized_pose_init_{init_grasp_idx}_max_deformation") #f"/home/shinghei/Downloads/grasp_planner_data/optimized_pose_{object_name}_init_{init_grasp_idx}_max_deformation"
    else:
        grasp_planner_optimized_pose_path = os.path.join(grasp_planner_data_root_path, object_name, f"optimized_pose_init_{init_grasp_idx}_min_deformation") #f"/home/shinghei/Downloads/grasp_planner_data/optimized_pose_{object_name}_init_{init_grasp_idx}_min_deformation"
    os.makedirs(grasp_planner_optimized_pose_path, exist_ok=True)

    num_epochs = args.num_epochs + 1

    force = args.force
    young_modulus = args.young_modulus
    B = 1
    
    

    ############### load and prepare data ###################

    with open(grasp_planner_training_data_path, 'rb') as handle:
        data = pickle.load(handle) 

    partial_pc = data["object_pc"] # shape (1, num_pts, 3)
    gripper_pc = data["gripper_pc"] # shape (1, num_pts, 3)
    query = data["query"] # shape (num_queries, 3)
    occupancy_0_force = data["pred_occupancy_0"] # shape (num_queries,)
    grasp_samples = data["grasp_samples"]["poses"] #shape (num_samples, 6)
    world_to_obj_homo_mat = data["world_to_obj_homo_mat"]

    print("partial pc: ", partial_pc.shape)
    print("gripper pc: ", gripper_pc.shape)
    print("query: ", query.shape)
    print("occupancy: ", occupancy_0_force.shape)
    print("grasp data type: ", type(grasp_samples))
    print("num grasp samples: ", len(grasp_samples))

    augmented_partial_pc = np.concatenate([partial_pc, np.tile(np.array([[force, young_modulus]]), 
                                                        (1, partial_pc.shape[1], 1))], axis=2) 
    
    augmented_gripper_pc = np.concatenate([gripper_pc, np.tile(np.array([[0, 0]]), 
                                                (1, gripper_pc.shape[1], 1))], axis=2)
    
    print("aug partial pc: ", augmented_partial_pc.shape)
    print("aug gripper pc: ", augmented_gripper_pc.shape)
    print("num queries: ", query.shape[0])
    print("num points on each pc: ", partial_pc.shape[1])

    device = torch.device("cuda")

    augmented_partial_pc = torch.from_numpy(augmented_partial_pc).float().to(device).squeeze(0)
    augmented_gripper_pc = torch.from_numpy(augmented_gripper_pc).float().to(device).squeeze(0)
    query = torch.from_numpy(query).float().to(device).unsqueeze(0).expand(B, -1, -1)
    occupancy_0_force = torch.from_numpy(occupancy_0_force).float().to(device).unsqueeze(0).expand(B, -1)
    world_to_obj_homo_mat = torch.from_numpy(world_to_obj_homo_mat).float().to(device)

    print("after casting as torch tensor: ")
    print("aug partial pc: ", augmented_partial_pc.shape)
    print("aug gripper pc: ", augmented_gripper_pc.shape)
    print("query: ", query.shape)
    print("occupancy: ", occupancy_0_force.shape)
    print("world to obj homo mat: ", world_to_obj_homo_mat)


    model = StressNet2(num_channels=5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    init_euler_angles = grasp_samples[init_grasp_idx][3:]
    init_translation_vec = grasp_samples[init_grasp_idx][0:3]
    print("euler angles: ", init_euler_angles)
    print("translation_vec: ", init_translation_vec)

    
    ############# start training ##############################
   
    grasp_planner = GraspPlanner(init_euler_angles, init_translation_vec).to(device)

    optimizer = optim.Adam(grasp_planner.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    start_time = timeit.default_timer()
    losses = []
    for epoch in range(0, num_epochs):
        print(f"================ Epoch {epoch}")
        print(f"Time passed: {(timeit.default_timer() - start_time)/60:.2f} mins\n")
        
        optimizer.zero_grad()
        predicted_occ = grasp_planner(query, augmented_partial_pc, augmented_gripper_pc, model, world_to_obj_homo_mat) # shape (B, num_queries)
        
        if maximize_deformation:
            loss = -nn.BCELoss()(predicted_occ, occupancy_0_force)
        else:
            loss = nn.BCELoss()(predicted_occ, occupancy_0_force) #chamfer_distance_batched(query, object_pc[:, :3], predicted_classes) 
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        scheduler.step()

        if epoch==(num_epochs-1):
            data = {"optimized_euler_angles": grasp_planner.euler_angles.data.cpu().detach(), "optimized_translation_vec": grasp_planner.translation_vec.data.cpu().detach(), "loss": losses[-1]}
            with open(os.path.join(grasp_planner_optimized_pose_path, f"epoch {epoch}.pickle"), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    xs = [i for i in range(num_epochs)]
    if maximize_deformation:
        plot_curve(xs, losses, x_label="epochs", y_label="losses", label_1="train_losses", title=f"Negative BCE train losses", path=f"{grasp_planner_data_root_path}/{object_name}/figures/figures_init_{init_grasp_idx}")
    else:
        plot_curve(xs, losses, x_label="epochs", y_label="losses", label_1="train_losses", title=f"BCE train losses", path=f"{grasp_planner_data_root_path}/{object_name}/figures/figures_init_{init_grasp_idx}")


