import numpy as np
import plotly.graph_objects as go

import os
import sys
current_dir = os.getcwd()
top_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(top_dir)
top_dir = top_dir.replace("\\", "/")

DATA_DIR = top_dir + "/data/super_resolution/datasets"

def plot_density(volume_matrix, surface_num=4, opacity=0.1):
    """
    Function for visualising a 3d volume

     Variables
    volume_matrix: a 3 dimenstional numpy array of any side lenghts
    surface_num: kind of like resolution but higher does not allways mean better 
    """
    step_num_x = complex(0,len(volume_matrix))
    step_num_y = complex(0,len(volume_matrix[0]))
    step_num_z = complex(0,len(volume_matrix[0][0]))
    X, Y, Z = np.mgrid[0:1:step_num_x, 0:1:step_num_y, 0:1:step_num_z]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_matrix.flatten(),
        isomin=0,
        isomax=1,
        opacity= opacity, # needs to be small to see through all surfaces
        surface_count=surface_num, # needs to be a large number for good volume rendering
        ))
    
    fig.show()


data = np.load(DATA_DIR + "/snap_007_tensors.npz")

sample = data["region 0, 0, 300"] # shape (256,256,256,6)
step = 2**4
sample = sample[::step,::step,::step,0] * 1e+28 
sample = sample / np.linalg.norm(sample) * 10

plot_density(sample)