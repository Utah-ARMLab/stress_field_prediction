import tensorflow as tf
import json
import os
import sys
sys.path.append("../")
from utils.miscellaneous_utils import pcd_ize
import numpy as np
np.set_printoptions(threshold=np.inf)

mgn_dataset_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"

# Load the meta.json file
with open((os.path.join(mgn_dataset_path, 'meta.json'))) as f:
    meta_data = json.load(f)

# Define the mapping of dtype strings to TensorFlow data types
dtype_mapping = {
    'int32': tf.int32,
    'float32': tf.float32,
    'string': tf.string,
}



# tfrecord_file = os.path.join(mgn_dataset_path, "5e5_pd", '6polygon04.tfrecord')

# for file_name in sorted(os.listdir(os.path.join(mgn_dataset_path, "5e5_pd"))):
for file_name in ['annulus02.tfrecord']:

    tfrecord_file = os.path.join(mgn_dataset_path, "raw_tfrecord_data", file_name)

    # Create a TFRecordDataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_file)


    data_name = "stress"  #"world_pos" "node_mod" "stress"

    feature_description = {
        data_name: tf.io.FixedLenFeature([], tf.string),
    }

    shape = meta_data['features'][data_name]['shape']


    # # Define a function to count the elements
    # def count_elements(count, _):
    #     return count + 1

    # # Use reduce to count the elements in the dataset
    # count = dataset.reduce(tf.constant(0), count_elements)

    # # Print the count
    # print("Length of the dataset:", count.numpy())

    # Iterate over the dataset and print the "mesh_pos" feature values
    for record in dataset:
        # mesh_pos = record[data_name]
        # mesh_pos = tf.io.decode_raw(mesh_pos, tf.float32)
        # print(mesh_pos)
        # break

        # Parse the record using the feature description
        parsed_record = tf.io.parse_single_example(record, feature_description)
        
        # Decode the values of the parsed record
        # For example, if 'feature_name2' is a byte string, you can decode it as follows:
        dtype = meta_data['features'][data_name]['dtype']
        tf_dtype = dtype_mapping.get(dtype, None)
        ans = tf.io.decode_raw(parsed_record[data_name], tf_dtype)
        
        # # print(ans.numpy().reshape(shape).shape)
        # print(ans.numpy().reshape(shape).squeeze()[-1])
        
        if data_name == "node_mod":
            res = ans.numpy().reshape(shape).squeeze()[-1]
            assert res > 13.1 and res <13.2
        
        elif data_name == "world_pos":
            world_pos = ans.numpy().reshape(shape).squeeze()
            print(world_pos.shape)
            # for pc in world_pos[0:]:
            #     pcd_ize(pc, vis=True)
            # pcd_ize(world_pos[0], vis=True)
        
        elif data_name == "stress":
            stress = ans.numpy().reshape(shape).squeeze()
            print(stress.shape)
            print(stress)
        
        break

