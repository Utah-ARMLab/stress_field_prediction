import open3d
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from isaacgym import gymapi
import torchvision
from torchvision.utils import make_grid


def compute_pointcloud(D_i, S_i, V_inv, P, w, h, min_z, segmentationId_dict, object_name="deformable", device="cuda"):
    '''
    All matrices should be torch tensor: 

    D_i = depth buffer for env i (h x w)
    S_i = segmentation buffer for env i (h x w)
    V_inv = inverse of camera view matrix (4 x 4)
    P = camera projection matrix (4 x 4)
    w = width of camera 
    h = height of camera
    min_z = the lowest z value allowed
    '''
    D_i = D_i.to(device)
    S_i = S_i.to(device)
    V_inv = V_inv.to(device)
    P = P.to(device)

    if object_name == "deformable":
        all_segmentationIds = list(segmentationId_dict.values())       
        for segmentationId in all_segmentationIds:
            D_i[S_i == segmentationId] = -10001  # Ignore any points from robot and other rigid objects.
   
    elif object_name in segmentationId_dict:
        D_i[S_i != segmentationId_dict[object_name]] = -10001  # Ignore any points that don't have the correct segmentationId.
    
    else:
        raise SystemExit("Error: Wrong object name, cannot compute point cloud.")

    fu = 2/P[0,0]
    fv = 2/P[1,1]

    center_u = w/2
    center_v = h/2

    # pixel indices
    k = torch.arange(0, w).unsqueeze(0) # shape = (1, w)
    t = torch.arange(0, h).unsqueeze(1) # shape = (h, 1)
    K = k.expand(h, -1).to(device) # shape = (h, w)
    T = t.expand(-1, w).to(device) # shape = (h, w)

    U = -(K - center_u)/w # image-space coordinate
    V = (T - center_v)/h # image-space coordinate

    X2 = torch.cat([(fu*D_i*U).unsqueeze(0), (fv*D_i*V).unsqueeze(0), D_i.unsqueeze(0), torch.ones_like(D_i).unsqueeze(0).to(device)], dim=0) # deprojection vector, shape = (4, h, w)
    X2 = X2.permute(1,2,0).unsqueeze(2) # shape = (h, w, 1, 4)
    V_inv = V_inv.unsqueeze(0).unsqueeze(0).expand(h, w, 4, 4) # shape = (h, w, 4, 4)
    # Inverse camera view to get world coordinates
    P2 = torch.matmul(X2, V_inv) # shape = (h, w, 1, 4)
    #print(P2.shape)
    
    # filter out low points and get the remaining points
    points = P2.reshape(-1, 4)
    depths = D_i.reshape(-1)
    mask = (depths >= -3) 
    points = points[mask, :]
    mask = (points[:, 2]>min_z)
    points = points[mask, :]
    
    return points[:, :3].cpu().numpy().astype('float32') 

def get_partial_pointcloud_vectorized(gym, sim, env, cam_handle, cam_prop, segmentationId_dict, object_name="deformable", color=None, min_z=0.005, visualization=False, device="cuda"):
    '''
    Remember to render all camera sensors before calling this method in isaac gym simulation
    '''
    gym.render_all_camera_sensors(sim)
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)

    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, segmentationId_dict, object_name, device)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points

def get_partial_point_cloud(gym, sim, env, cam_handle, cam_prop, vis=False):

    cam_width = cam_prop.width
    cam_height = cam_prop.height
    # Render all of the image sensors only when we need their output here
    # rather than every frame.
    gym.render_all_camera_sensors(sim)

    points = []
    # print("Converting Depth images to point clouds. Have patience...")

    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)


    # Get the camera view matrix and invert it to transform points from camera to world
    # space
    
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 11] = -10001

    centerU = cam_width/2
    centerV = cam_height/2
    for k in range(cam_width):
        for t in range(cam_height):
            if depth_buffer[t, k] < -3:
                continue

            u = -(k-centerU)/(cam_width)  # image-space coordinate
            v = (t-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[t, k]  # depth buffer value
            X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
            p2 = X2*vinv  # Inverse camera view to get world coordinates
            # print("p2:", p2)
            if p2[0, 2] > 0.005:
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])

    if vis:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        open3d.visualization.draw_geometries([pcd]) 

    return np.array(points).astype('float32') 
 

def display_images(images, num_columns=4, output_file=None):

    """
    Display N images in a grid layout of size num_columns x np.ceil(N/num_columns) using pytorch.
    
    1.Input: 
    images: a list of torch tensor images, shape (3,H,W).
    
    """

    # num_images = len(images)   
    Grid = make_grid(images, nrow=num_columns, padding=0)
    
    # display result
    img = torchvision.transforms.ToPILImage()(Grid)
    
    if output_file is not None:
        # Save the figure to the specified output file
        img.save(output_file)
    else:
        # Display figure to screen
        img.show() 


def visualize_camera_views(gym, sim, env, cam_handles, resolution=[1000,1000], output_file=None):
    images = []
    gym.render_all_camera_sensors(sim)

    for cam_handle in cam_handles:
        image = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR).reshape((resolution[0],resolution[1],4))[:,:,:3]
        print(image.shape)
        images.append(torch.from_numpy(image).permute(2,0,1) )

    display_images(images, output_file=output_file)