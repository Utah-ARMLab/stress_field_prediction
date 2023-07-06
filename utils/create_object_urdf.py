import os
from constants import OBJECT_NAMES


density = 1000
youngs = "5e4"   #1e4
poissons = 0.3
attach_dist = 0.0
scale = 1

for object_name in OBJECT_NAMES[4:]:

    object_urdf_path = f"../sim_data/stress_prediction_data/objects/{object_name}"

    os.makedirs(object_urdf_path,exist_ok=True)
 
    cur_urdf_path = object_urdf_path + "/soft_body.urdf"
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

# <youngs value="{round(youngs)}"/>
