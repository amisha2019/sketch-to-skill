import h5py

# Open the HDF5 file in read/write mode
with h5py.File('/fs/nexus-projects/Sketch_VLM_RL/RoboTurkPilot/bins-Can/demo.hdf5', 'r+') as f:
    # Update the environment name in the metadata
    f.attrs['env'] = 'PickPlaceCan'

