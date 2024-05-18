import  h5py
import os,pdb

dir_path = "/Users/pranavd/Pranav/frb/fetch/h5_files/less_h5"
for file in os.listdir(dir_path):
    full_h5_path = os.path.join(dir_path, file)
    with h5py.File(full_h5_path, "r") as f:
        if  full_h5_path=="/Users/pranavd/Pranav/frb/fetch/h5_files/less_h5/test1.hdf5":
            for key in f.keys():
                print(full_h5_path)
                pdb.set_trace()
            if key[:-1]=="data_dm_time":
                f.move(key, "data_dm_time")
            else:
                f.move(key,"data_freq_time")
            