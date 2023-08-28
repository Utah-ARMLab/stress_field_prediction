import os
import numpy as np


density = 1000
youngs = "5e4"
poissons = 0.3
attach_dist = 0.0
scale = 1


selected_objects = ["strawberry02", "lemon02", "mustard_bottle"]

for object_name in selected_objects:

    object_urdf_path = f"../sim_data/stress_prediction_data/dgn_dataset_varying_stiffness/{object_name}"

    os.makedirs(object_urdf_path,exist_ok=True)
       
    cur_urdf_path = object_urdf_path + f"/evaluate_soft_body.urdf"
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


