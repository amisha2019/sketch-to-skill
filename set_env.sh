# add root to python path
export PYTHONPATH=$PWD:$PYTHONPATH
conda activate sketchtoskill

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=/fs/nexus-projects/Sketch_VLM_RL/amishab/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fs/nexus-projects/Sketch_VLM_RL/amishab/.mujoco/mujoco210/bin
