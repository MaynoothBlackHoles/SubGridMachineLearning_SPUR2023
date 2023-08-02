"""
Script to hold data generating, classifying, pre processing and testing functions
"""

import numpy as np
from multiprocessing import Pool


def check_voxel(tensor, x, y, z):
    """
    Checks a if a given coorindate (x, y, z) of a tensor is star forming or not

    return: True or False
    """

    number_density = tensor[0, x, y, z]
    H2_fraction    = tensor[1, x, y, z]
    freefall_time  = tensor[2, x, y, z]
    cooling_time   = tensor[3, x, y, z]
    divergence     = tensor[4, x, y, z]
    
    if (number_density >= 100) and (H2_fraction >= 1e-3) and (freefall_time < cooling_time) and (divergence < 0):
        return True
    else:
        return False



def check_tensor(tensor, Full_list=False):
    """
    Checks whether a tensor if star forming or not by checking if any entry is star forming.
    
     Variables
    tensor: input tensor
    Full_list: If this is set to be True then this function will output each entry of the tensor which is star forming in a list

    return: True or False
    """

    matrices, x, y, z = tensor.shape
    star_forming_pixels = []
    
    if Full_list:
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_voxel(tensor, i, j, k):
                        star_forming_pixels.append([i, j, k])

        return star_forming_pixels
    
    else:
        break_state = False
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    if check_voxel(tensor, i, j, k):
                        break_state = True
                        break
                if break_state:
                    break
            if break_state:
                break
        
        return break_state



def check_starForming(tensor):
    if check_tensor(tensor):
        return 1 # star forming tensor
    else:
        return 0 # non star forming tensor
    
    
###############################################################################
#                  Subgrid Model with Stars and Black Holes                   #
###############################################################################

# NOTE: the below functions assume regions are represented by tensors whose
# shape is (Nx, Ny, Nz, Nc) - where Nc = 6 is the number of channels. This is
# in contrast to the pytorch tensors used by the pytorch networks which
# assume a shape (Nc, Nx, Ny, Nz).

# Critical values for this subgrid model.
H2_crit       = 4.2 # in units of 1e-6
rho_crit_star = 15  # in units of 1.6e-28
rho_crit_BH   = 60  # in units of 1.6e-28


def site_forms_compact_object(X, i, j, k):
    """
    This function checks if site (i, j, k) of region X forms a compact object.

    returns:
        0 - if the site does not form a compact object
        1 - if the site forms a star
        2 - if the site forms a black hole
    """
    density = X[i, j, k, 0]
    H2_frac = X[i, j, k, 1]
    vel_div = X[i, j, k, 5]

    if (vel_div < 0) and (H2_frac > H2_crit):
        if density >= rho_crit_BH:
            return 2

        elif density >= rho_crit_star:
            return 1

    return 0



def get_compact_object_sites(X):
    """
    This function finds the sites of region X that form a compact object.

    returns:
        two lists 'star_sites' and 'BH_sites' containing (i, j, k) coordinates
        of star forming and black hole forming sites respectively.
    """
    star_sites = []
    BH_sites = []

    Nx, Ny, Nz, Nc = X.shape
    for i in range(Nx):
        print(i)
        for j in range(Ny):
            for k in range(Nz):
                forms_compact_object = site_forms_compact_object(X, i, j, k)

                if forms_compact_object == 1:
                    star_sites.append((i, j, k))

                elif forms_compact_object == 2:
                    BH_sites.append((i, j, k))

    return star_sites, BH_sites



def _check_regions_for_compact_objects(kX):
    k, X = kX
    print('Checking ' + k)
    return (k, get_compact_object_sites(X))



def check_regions_for_compact_objects(Xs):
    """
    This function iterates through the regions contained in Xs and returns
    a list of 'name'-'tuple' pairs where the name is the name of the region
    and the tuple contains a list of star forming sites and a list of black
    hole forming sites.
    """
    with Pool(21) as p:
        return p.map(_check_regions_for_compact_objects, Xs.items())



def create_labels_for_snap(filename):
    """
    Creates a dictionary of labels for the tensors stored in the file with the 
    given filename.
    """
    Xs = np.load(filename)
    labels_filename = filename.replace("tensors", "labels")
    labels_list = None

    with Pool(21) as p:
        labels_list = p.map(create_label, Xs.items())

    labels = {X_name : label for (X_name, label) in labels_list}

    np.savez(labels_filename, **labels)



def create_label(kX):
    """
    Returns a (name, label) tuple for the given (name, region) pair

    regions are labeled as follows:
        0 - not star forming or black hole forming
        1 - star forming
        2 - black hole forming
        3 - star and black hole forming
    """
    k, X = kX
    print("Getting labels for " + k)
    stars, black_holes = get_compact_object_sites(X)
    if stars:
        if black_holes:
            label = 3
        else:
            label = 1
    else:
        if black_holes:
            label = 2
        else:
            label = 0

    return (k, label)