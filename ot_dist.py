import ot
import numpy as np
import math

def pd_to_meas(pd, grid_size, grid_step):
    """
    transform persistence diagram into persistence measure
    """
	meas = np.zeros(grid_size)
	for x in pd:
		i = math.floor(x[0]/grid_step)
		j = math.floor(x[1]/grid_step)
		if (i<grid_size[0]) and (j<grid_size[1]):
			meas[i,j] += 1
	return meas.T

def dist_mat(grid_width, grid_step, p):
    """
    compute distance matrix of grid points
    """
	size = int(grid_width*(grid_width+1)/2)+1
	grid = []
	for i in range(grid_width):
		for j in range(i+1, grid_width+1):
			grid.append([grid_step*i,grid_step*j])
            
	M = np.zeros([size, size])
	for i in range(size-1):
		for j in range(size-1):
			M[i,j] = np.power(abs(grid[i][0]-grid[j][0]),p) + np.power(abs(grid[i][1]-grid[j][1]),p)
	# diagonal 
	for k in range(size-1):
		M[k,size-1] = np.power((grid[k][1]-grid[k][0])/2,p)
		M[size-1,k] = np.power((grid[k][1]-grid[k][0])/2,p)
	return M

def ot_dist(mu, nu, Mp):
    """
    compute OT_p^p distance between two measures mu and nu.
    """
	length = np.shape(mu)[0]
	mu = mu.T[np.triu_indices(length)].tolist()
	nu = nu.T[np.triu_indices(length)].tolist()
	mu.append(sum(mu))
	nu.append(sum(nu))

	return ot.emd2(mu,nu,Mp)
