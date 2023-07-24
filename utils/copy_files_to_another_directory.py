import shutil
import os

source_dir = '/home/baothach/shape_servo_data/stress_field_prediction/6polygon04_data'
destination_dir = '/home/baothach/shape_servo_data/stress_field_prediction/6polygon/all_6polygon_data'

# List all files in the source directory
files_to_copy = sorted(os.listdir(source_dir))
print("Number of files to be transferred:", len(files_to_copy))
# print(files_to_copy)

# Loop through each file and copy it to the destination directory
for file_name in files_to_copy:
    source_file_path = os.path.join(source_dir, file_name)
    destination_file_path = os.path.join(destination_dir, file_name)
    shutil.copy2(source_file_path, destination_file_path)

print("Files copied successfully!")