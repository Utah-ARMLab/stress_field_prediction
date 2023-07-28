import os
import numpy as np


density = 1000
# youngs = "5e4"   #1e4
poissons = 0.3
attach_dist = 0.0
scale = 1
youngs_bounds = np.log([1e4, 5e5])

for object_name in [f"6polygon0{j}" for j in [3,5,6,7,8]]:

    object_urdf_path = f"../sim_data/stress_prediction_data/dgn_dataset_varying_stiffness/{object_name}"

    os.makedirs(object_urdf_path,exist_ok=True)

    for grasp_idx in range(0,100):
        
        youngs = np.random.uniform(low=youngs_bounds[0], high=youngs_bounds[1])
        youngs = np.exp(youngs)
        youngs = "{:.2e}".format(youngs).replace("+", "")    # convert float number to scientific notation string
        
        cur_urdf_path = object_urdf_path + f"/soft_body_grasp_{grasp_idx}.urdf"
        f = open(cur_urdf_path, 'w')

        urdf_str = f"""<?xml version="1.0" encoding="utf-8"?>    

        <robot name="{object_name}">
            <link name="{object_name}">    
                <fem>
                    <origin rpy="0.0 0.0 0.0" xyz="0 0 0" />
                    <density value="{density}" />
                    <youngs value="{youngs}"/>
                    <poissons value="{poissons}"/>
                    <damping value="0.0" />
                    <attachDistance value="{attach_dist}"/>
                    <tetmesh filename="{object_name+".tet"}"/>
                    <scale value="{scale}"/>
                </fem>
            </link>

        </robot>
        """

        f.write(urdf_str)
        f.close()


