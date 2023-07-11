import argparse
import os
import trimesh
import shutil
import h5py
import pymeshlab
import random
import numpy as np
import subprocess
import sys
import timeit

LOCAL = True
print("Top header")
import xml.etree.ElementTree as ET

if LOCAL:
	# sys.path.insert(1, '/home/isabella/carbgym/DeformableGrasping/graspsampling-py')
	sys.path.insert(1, '/home/isabella/deformable_object_grasping/graspsampling-py-internship1')

else:
	sys.path.insert(1, '/ws-mount/DeformableGrasping/graspsampling-py')

import graspsampling
from graspsampling import sampling, utilities, hands, io

sys.path.insert(1, '..')
import mesh_to_tet

def get_original_stl_path(base_path, folder_name):
	all_files = os.listdir(os.path.join(base_path, folder_name))
	original_stl = [f for f in all_files if ".stl" in f and not "processed" in f and not "temp" in f]
	assert(len(original_stl) == 1)
	return os.path.join(base_path, folder_name, original_stl[0])

def simplify_mesh(ms, target_nv):
	# https://stackoverflow.com/questions/65419221/how-to-use-pymeshlab-to-reduce-vertex-number-to-a-certain-number/65424578#65424578
	numFaces = 100 + 2*target_nv

	while (ms.current_mesh().vertex_number() > target_nv):
		ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
		numFaces = numFaces - (ms.current_mesh().vertex_number() - target_nv)
	return

def edge_lengths_from_corners(corners):
	e1 = np.linalg.norm(corners[1] - corners[0])
	e2 = np.linalg.norm(corners[2] - corners[1])
	e3 = np.linalg.norm(corners[4] - corners[0])
	return [e1, e2, e3]

def main(dir_name, soft_body_original_path):
	print("Received dir name", dir_name)
	stls_folder = dir_name 

	if not LOCAL:
		stls_folder = os.path.join("/ws-mount", "deformable_object_grasping", "abc_dataset", dir_name)


	processed_tag = "_processed"
	temp_tag = "_temp"
	ftetwild_path = '/home/isabella/fTetWild/build/FloatTetwild_bin'
	if not LOCAL:
		ftetwild_path = '/ws-mount/fTetWild/build/FloatTetwild_bin'

	gripper = hands.create_gripper('panda_visual')
	number_of_grasps = 100


	# Move files into subfolders if necessary
	for file in os.listdir(stls_folder):
		if ".stl" in file and processed_tag not in file:
			obj_name = file.split("_")[0]
			folder_path = os.path.join(stls_folder, obj_name)
			if not os.path.isdir(folder_path):
				os.mkdir(folder_path)
			src_path = os.path.join(stls_folder, file)
			dest_path = os.path.join(stls_folder, obj_name + ".stl")
			os.rename(src_path, dest_path)
			shutil.move(dest_path, folder_path)


	# Iterate through meshes in directory
	# tet_sizes = []
	# folders_with_tets = []



	for folder in sorted(os.listdir(stls_folder)):
		print("=======", folder)

		# If not a directory (e.g. platform.obj, continue)
		if not os.path.isdir(os.path.join(stls_folder, folder)):
			continue



		loop_start = timeit.default_timer()
		# Add condition to skip if the final file is there
		existing_files = os.listdir(os.path.join(stls_folder, folder))
		num_files = len(existing_files)

		# Rename files by dropping the part of the name after number

		if num_files == 1:
			original_stl_name = existing_files[0]
			if not original_stl_name.count("_") == 0: 
				assert(original_stl_name.split(".")[-1] == "stl")
				object_num = original_stl_name.split("_")[0]
				os.rename(os.path.join(stls_folder, folder, original_stl_name), os.path.join(stls_folder, folder, object_num + ".stl"))


		if any(".tet" in ef for ef in existing_files) and num_files >= 5:
			print("Folder already good", folder)
			continue


		# Get file paths
		fp_names = dict()
		fp_names['temp_path'] = os.path.join(stls_folder, folder, folder + temp_tag + ".stl")
		fp_names['processed_path'] = os.path.join(stls_folder, folder, folder + processed_tag + ".stl")
		fp_names['original_stl_path'] = os.path.join(stls_folder, folder, folder + ".stl")
		fp_names['dot_mesh_path'] = os.path.join(stls_folder, folder, folder + ".mesh")
		fp_names['dot_tet_path'] = os.path.join(stls_folder, folder, folder + ".tet")
		fp_names['grasps_path'] = os.path.join(stls_folder, folder, folder + "_grasps.h5")
		fp_names['soft_body_original_path'] = soft_body_original_path #os.path.join(stls_folder, folder, "..", "..", "..", "example_soft_body_original.urdf")
		# if not LOCAL:
			# fp_names['soft_body_original_path'] = os.path.join(stls_folder, folder, "..", "..", "..", "example_soft_body_original.urdf")
		fp_names['soft_body_urdf_path'] = os.path.join(stls_folder, folder, "soft_body.urdf")

		files_to_keep = []
		files_to_keep.append(fp_names['original_stl_path'])
		files_to_keep.append(fp_names['dot_tet_path'])
		files_to_keep.append(fp_names['grasps_path'])
		files_to_keep.append(fp_names['soft_body_urdf_path'])
		files_to_keep.append(fp_names['processed_path'])
		all_files_to_keep = " ".join(files_to_keep)



		# '''
		# Add soft_body.urdf file if non existent
		tree = ET.parse(fp_names['soft_body_original_path'])
		root = tree.getroot()
		params = {'tetmesh': folder + ".tet"}
		for key, value in params.items():
			for attribute in root.iter(key):
				attribute.set('filename', str(value))
		tree.write(fp_names['soft_body_urdf_path'])

		# Take largest volume body. If not watertight, take convex hull
		obj_mesh = trimesh.load(fp_names['original_stl_path'])

		bodies = obj_mesh.split()
		if len(bodies) == 0:
			print("No bodies")
			continue
		body_volumes = [o.volume for o in bodies]
		highest_vol_body = bodies[np.argmax(body_volumes)]
		if not highest_vol_body.is_watertight:
			print("Not watertight")
			assert(False)
			highest_vol_body = trimesh.convex.convex_hull(highest_vol_body)
		
		highest_vol_body.export(fp_names['temp_path'])

		# Take just the exterior if multiple parts to the mesh. Note: This is bad for hollow objects!
		ms = pymeshlab.MeshSet()
		ms.load_new_mesh(fp_names['temp_path'])
		m = ms.current_mesh()

		# Simplify vertex count
		simplify_mesh(ms, 5000)

		# Scale down mesh if needed
		bb = m.bounding_box()
		min_dim = min(bb.dim_x(), bb.dim_y(), bb.dim_z())
		max_dim = max(bb.dim_x(), bb.dim_y(), bb.dim_z())
		scale_factor = 1.0

		ms.apply_filter('transform_scale_normalize', axisx=scale_factor, axisy=scale_factor, axisz=scale_factor)
		ms.save_current_mesh(fp_names['processed_path'])


		# Sample grasps on the object
		print("Sampling grasps")
		grasp_sampling_object = utilities.instantiate_mesh(file=fp_names['processed_path'], scale=1.0)
		cls_sampler = graspsampling.sampling.AntipodalSampler
		sampler = cls_sampler(gripper, grasp_sampling_object, 0.0, 4) # Antipodal
		results, grasp_success = graspsampling.sampling.collision_free_grasps(gripper, grasp_sampling_object, sampler, number_of_grasps)
		if len(results) == 0:
			print("Grasping sampling failed")
			continue
		results = np.asarray(results)
		results = {'poses': results}

		# Write grasps to h5
		print("Writing grasps to h5 file:", fp_names['grasps_path'])
		hf = h5py.File(fp_names['grasps_path'], 'w')
		hf.create_dataset('poses', data=results['poses'])
		hf.close()

		#Convert .stl to .mesh
		print("Converting .stl to .mesh")

		try:
			subprocess.call([ftetwild_path, '-i', fp_names['processed_path'], '-o', fp_names['dot_mesh_path'], '-q'], timeout=300)
		except:
			print("fTetWild failed")
			continue

		# Convert .mesh to .tet
		mesh_exists = False
		for me in range(100):
			mesh_exists = os.path.exists(fp_names['dot_mesh_path'])
			if mesh_exists:
				break
		if not mesh_exists:
			print("Mesh file does not exist", fp_names['dot_mesh_path'])
			continue

		print("Converting .mesh to .tet")
		mesh_to_tet.convert_mesh_to_tet(fp_names['dot_mesh_path'], fp_names['dot_tet_path'])
		# '''

		# Delete all intermediate files
		for file in os.listdir(os.path.join(stls_folder, folder)):
			if not file in all_files_to_keep:
				os.remove(os.path.join(stls_folder, folder, file)) 


if __name__ == "__main__":
	# Create command line flag options
	print("Starting in main")
	parser = argparse.ArgumentParser(description='Options')
	parser.add_argument('--dir', required=True, help="Folder name for object results") # eg. 0-99
	parser.add_argument('--example_urdf', required=True, help="Path to example_soft_body_original.urdf")
	args = parser.parse_args()
	main(args.dir, args.example_urdf)

	# python process_object_models.py --dir=../simple_geometric_dataset --example_urdf=../example_soft_body_original.urdf