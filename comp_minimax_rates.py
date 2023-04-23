import numpy as np
from ot_dist import *
from wavelet_estimator import WaveletDensityEstimator


def empirical_EPD (diagrams):
    dgms = np.reshape(diagrams,(diagrams.shape[0]*diagrams.shape[1],3))
    return dgms[np.argsort(dgms[:,2])]

def nbpts_PH(diagrams) :
    ph0 = len(np.where(np.isclose(diagrams[:,2],0))[0])
    ph1 = len(np.where(np.isclose(diagrams[:,2],1))[0])
    ph2 = len(np.where(np.isclose(diagrams[:,2],2))[0])
    return ph0, ph1, ph2

def get_PH_dim(diagrams,dim) :
    ph0, ph1, ph2 = nbpts_PH(diagrams)
    if dim == 0 :
        diag = diagrams[:ph0, :2]
    if dim == 1 :
        diag = diagrams[ph0:ph0+ph1, :2]
    if dim == 2 :
        diag = diagrams[ph0 + ph1:, :2]
    return diag

def get_PH_alldims(diagrams):
    ph0, ph1, ph2 = nbpts_PH(diagrams)
    PH0 = diagrams[:ph0, :2]
    PH1 = diagrams[ph0:ph0+ph1, :2]
    PH2 = diagrams[ph0 + ph1:, :2]
    return PH0, PH1, PH2


# load data
dgms = np.load('data/torus.npy')
#dgms = np.load('data/doubletorus.npy')

# construct grid
R = 1.975 * np.sqrt(8)
x = np.arange(0,2,.025)
y = np.arange(0,2,.025)
X,Y = np.meshgrid(x, y)

# compute distance matrices for OT-metric
M1 = dist_mat(80,0.025,1)
M2 = dist_mat(80,0.025,2)
M3 = dist_mat(80,0.025,3)
M4 = dist_mat(80,0.025,4)
M = [M1,M2,M3,M4]

# compute approximation of EPD
diagrams_all_samples = empirical_EPD(dgms)
diagrams_all = get_PH_alldims(diagrams_all_samples)
data = diagrams_all[1]
EPD = pd_to_meas(data,[80,80],0.025)
EPD_norm = EPD/ EPD.sum()

#
res = []
res_nb_coeff = np.zeros([10,50])
for j in np.arange(0,10,1): # 10 runs for averaging
    res_tmp = []
    for NN in np.arange(10,510,10):
        diagrams_all_samples_NN = empirical_EPD(dgms[(j*500):(j*500)+NN,:,:])
        diagrams_all_NN = get_PH_alldims(diagrams_all_samples_NN)
        data_NN = diagrams_all_NN[1]
        
        # compute wavelet estimator and empirical mean
        res_NN = np.zeros([80,80])
        nb_coeff = 0
        for k in np.arange(2,9):
            wt = WaveletDensityEstimator(wave_name='haar',N=NN,R=R,k=k,j0=9,J=11,lvl=25,thresholding=0) # here: thresholding tau = 0
            wt.fit(data_NN)
            res_NN += wt.pdf(X,Y)
            res_NN[res_NN<0] = 0
            nb_coeff += wt.number_coeff_nonzero
        
        res_nb_coeff[j,int(NN/10)-1] = nb_coeff
        
        res_NN_norm = res_NN/res_NN.sum()
        emp_mean = pd_to_meas(data_NN,[80,80],0.025)
        emp_mean_norm = emp_mean/ emp_mean.sum()
        
        # compute OT losses
        tmp = ot_dist(res_NN_norm,EPD_norm,M[0]), ot_dist(emp_mean_norm,EPD_norm,M[0]), ot_dist(res_NN_norm,EPD_norm,M[1]), ot_dist(emp_mean_norm,EPD_norm,M[1]), ot_dist(res_NN_norm,EPD_norm,M[2]), ot_dist(emp_mean_norm,EPD_norm,M[2]), ot_dist(res_NN_norm,EPD_norm,M[3]), ot_dist(emp_mean_norm,EPD_norm,M[3])
        res_tmp.append(np.array(res_tmp))
    res.append(np.array(res_tmp))
res_avg = np.sum(res,axis=0)/10 # average over 10 runs
res_avg_nb = np.sum(res_nb_coeff,axis=0)/10

np.save('results/torus/OT_losses_avg_tau0.npy',res_avg)
np.save('results/torus/nb_coeff_avg_tau0.npy',res_avg_nb)
