import numpy as np
import trimesh
import open3d
import sys
import os


def find_folder_directory(folder_name):
    """
    Recursively searches for the absolute path of a folder (given its name), 
    in the current working directory and its parent directories.

    Parameters:
        folder_name (str): The name of the folder to search for.

    Returns:
        str or None: The absolute path to the folder if found, else None.
    """
    
    current_dir = os.getcwd()   # absolute path of the current working directory
    while current_dir != "/" and os.path.basename(current_dir) != folder_name:
        current_dir = os.path.dirname(current_dir)
    if os.path.basename(current_dir) == folder_name:
        return current_dir
    else:
        return None  # Folder not found

def sample_grasps(object_asset, cls_sampler="AntipodalSampler", object_scale=1, number_of_grasps=50, 
                  visualization=False, vis_gripper_name='panda', 
                  save_grasps_dir=None,
                  **surface_reconstruction_params):

    """
    Sample grasps; either from object mesh, or object full point cloud.

    Parameters:
        object_asset (str, trimesh, or numpy arrays): 
                If str: path to object mesh. 
                If trimesh: object mesh. 
                If np array shape (num_points, 3): object full point cloud.
        cls_sampler: type of sampling scheme. For example: AntipodalSampler, SurfaceApproachSampler. For more sampling schemes, refer to sampling.py.
        vis_gripper_name: for visualization purpose only. The gripper name that you want to use to 
                        visualize the sampled grasps. Options: 'panda', 'panda_tube'.
        save_grasps_dir: if is not None, export the sampled grasps to an h5 file.              

    Returns:
        results (dict): results["poses"] contains all the sampled grasp.

    Example usage:
        grasps = sample_grasps(object_full_point_cloud, cls_sampler="AntipodalSampler", 
                                number_of_grasps=10, visualization=True, vis_gripper_name='panda',
                                alpha=0.03)
    """
    
    project_root_directory = find_folder_directory("stress_field_prediction")        
    sys.path.append(os.path.join(project_root_directory, "graspsampling-py-defgraspsim"))
    
    from graspsampling import sampling, utilities, hands, io, visualize
    from utils.miscellaneous_utils import pcd_ize
    from utils.mesh_utils import open3d_to_trimesh_mesh
    
    sampler_dict = \
    {
        "AntipodalSampler": sampling.AntipodalSampler,
        "SurfaceApproachSampler": sampling.SurfaceApproachSampler
    }
    
    cls_sampler = sampler_dict[cls_sampler]

    if isinstance(object_asset, str):
        ### Load object        
        object_mesh = utilities.instantiate_mesh(file=object_mesh, scale=object_scale)
        print(f"Loading object mesh from {object_mesh} ...")
    
    elif isinstance(object_asset, trimesh.Trimesh):
        print(f"Loading trimesh object directly ...")
        object_mesh = object_asset
    
    elif isinstance(object_asset, np.ndarray):
        
        """ 
        open3d Surface Reconstruction:
        http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
        """
        
        print(f"Load object as full point cloud, then reconstruct the surface mesh ...")
        reconstructed_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_ize(object_asset), 
                                                                                surface_reconstruction_params['alpha'])
        object_mesh = open3d_to_trimesh_mesh(reconstructed_mesh)

    else:
        raise ValueError("Invalid object asset.")   
    
     
    ### Load panda gripper
    gripper = hands.create_gripper('panda', 0.04)   # 0.04 makes the gripper open, remove it if want to make gripper close


    ### Instantiate and run sampler (with collision checking)
    if cls_sampler==sampling.AntipodalSampler:
        # sampler = cls_sampler(gripper, object_mesh, number_of_orientations=6) 
        sampler = cls_sampler(gripper, object_mesh, 0.0, 4) # Antipodal
    else:
        sampler = cls_sampler(gripper, object_mesh)    
    filtered_poses = sampling.collision_free_grasps(gripper, object_mesh, sampler, number_of_grasps)
    results = {"poses": filtered_poses}

    assert(len(results['poses']) == number_of_grasps)
    
    
    ### Visualize sampled grasps on a test object.
    if visualization:       
        scene = visualize.create_scene(object_mesh, vis_gripper_name, **results)
        scene.show()

    if save_grasps_dir is not None:
        ### Save sampled grasps to a h5 file.
        h5_writer = io.H5Writer(save_grasps_dir)
        h5_writer.write(**results)        

    return results 