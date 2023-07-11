import tensorflow as tf
import json
import os
import pickle
import numpy as np

def read_data_from_tfrecord(dataset, data_name):
    feature_description = {
        data_name: tf.io.FixedLenFeature([], tf.string),
    }

    shape = data_configs['features'][data_name]['shape']
    dtype = data_configs['features'][data_name]['dtype']
    tf_dtype = dtype_mapping.get(dtype, None)

    # Iterate over the dataset 
    datas = []
    for record in dataset:

        # Parse the record using the feature description
        parsed_record = tf.io.parse_single_example(record, feature_description)
        
        # Decode the values of the parsed record
        if dtype != 'string':
            data = tf.io.decode_raw(parsed_record[data_name], tf_dtype)
        else:
            data = parsed_record
        
        # Convert to numpy and reshape to the original shape
        if dtype != 'string':
            data = data.numpy().reshape(shape).squeeze()
        else:
            data = data[data_name].numpy().decode()
        datas.append(data)
        
    return datas

mgn_dataset_main_path = "/home/baothach/shape_servo_data/stress_field_prediction/mgn_dataset"
mgn_converted_pickle_path = os.path.join(mgn_dataset_main_path, "raw_pickle_data")

# Load the meta.json file
with open((os.path.join(mgn_dataset_main_path, 'meta.json'))) as f:
    data_configs = json.load(f)
    print(data_configs.keys())

# Define the mapping of dtype strings to TensorFlow data types
dtype_mapping = {
    'int32': tf.int32,
    'float32': tf.float32,
    'string': tf.string,
}


for file_name in sorted(os.listdir(os.path.join(mgn_dataset_main_path, "raw_tfrecord_data")))[20:]:
# for file_name in ['6polygon01.tfrecord']:

    object_name = os.path.splitext(file_name)[0]

    tfrecord_file = os.path.join(mgn_dataset_main_path, "raw_tfrecord_data", file_name)

    # Create a TFRecordDataset from the file
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # # Count dataset size
    # def count_elements(count, _):
    #     return count + 1
    # count = dataset.reduce(tf.constant(0), count_elements)
    # print("Length of the dataset:", count.numpy())

    data_dict = {}  # include datas for all 100 grasps 
    data_names = [name for name in data_configs['features']]
    for data_name in data_names:    
        # data_name = "name"
        datas = read_data_from_tfrecord(dataset, data_name) # len(datas) = 100, all datas for 100 grasp poses
        data_dict[data_name] = datas
    
    # Split data_dict into each individual grasp index
    for grasp_idx in range(len(data_dict["world_pos"])):    
        saved_data = {}
        for data_name in data_names:   
            saved_data[data_name] = data_dict[data_name][grasp_idx]
        
        with open(os.path.join(mgn_converted_pickle_path, f"{object_name}_grasp_{grasp_idx}.pickle"), 'wb') as handle:
            pickle.dump(saved_data, handle, protocol=pickle.HIGHEST_PROTOCOL)     




        
