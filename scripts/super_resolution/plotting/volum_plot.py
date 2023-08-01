import numpy as np
import plotly.graph_objects as go


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

# sample volumes
n = 40

a = np.zeros((n,n,n))
for i in range(n):
    a[i, i, i] = 1

b = np.random.randn(n, n, n)

# plotting
plot_density(b)