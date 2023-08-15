import os

# selected_objects = []
# selected_objects += \
# [f"lemon0{j}" for j in [1,2,3]] + \
# [f"strawberry0{j}" for j in [1,2,3]] + \
# [f"tomato{j}" for j in [1]] + \
# [f"apple{j}" for j in [3]] + \
# [f"potato{j}" for j in [3]]
# selected_objects += ["bleach_cleanser", "crystal_hot_sauce", "pepto_bismol"]
# selected_objects += [f"cylinder0{j}" for j in range(1,9)] + [f"box0{j}" for j in range(1,10)] \
#                 + [f"ellipsoid0{j}" for j in range(1,6)] + [f"sphere0{j}" for j in [1,3,4,6]]


source_dir = "/home/baothach/shape_servo_data/stress_field_prediction/all_primitives/processed"
target_dir = "/home/dvrk/shape_servo_data/stress_field_prediction/all_primitives"

# source_dir = "/home/baothach/shape_servo_data/stress_field_prediction/static_data_original"
# target_dir = "/home/dvrk/shape_servo_data/stress_field_prediction"


os.system(f"rsync -av --progress {source_dir} strange-dvrk:{target_dir}")