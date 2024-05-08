import h5py
import numpy as np

def generate_hdf5_data(num_pairs, dim, file_name):
    with h5py.File(file_name, 'w') as hf:
        data_pairs = np.random.uniform(-1, 1, (num_pairs, 2, dim))
        hf.create_dataset('dataset', data=data_pairs)

# 生成并保存数据
num_pairs = 100000
dim = 128
file_name = "data_pairs.h5"
generate_hdf5_data(num_pairs, dim, file_name)
