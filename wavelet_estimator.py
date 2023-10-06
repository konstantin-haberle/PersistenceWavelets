from __future__ import division
import numpy as np
import math
import pywt
from functools import partial
from scipy.interpolate import interp1d
from scipy.special import gamma

class WaveletDensityEstimator(object):
    def __init__(self, wave_name, N, R, k, j0, J, supp=1, lvl=15, thresholding=0):
        self.wave = pywt.Wavelet(wave_name)
        self.R = R # set R = 2*sqrt(2)*t_max
        self.k = k
        self.j0 = j0
        self.J = J 
        self.N= N
        self.supp = supp
        self.lvl = lvl
        self.number_coeff = 0
        self.number_coeff_nonzero = 0
        self.pdf = None
        if thresholding !=0:
            self.thresholding = lambda j: 2**(j-j0)*(j)/np.sqrt(N)*thresholding
        else:
            self.thresholding = lambda j: 0

    def fit(self, data):
        """
        fit estimator to data. data is an np array of dimension N x M x 2, N = samples, M = number of points in PD
        """
        self.calc_wavefuns()
        self.calc_coeffs(data)
        self.pdf = self.calc_pdf()
        return True

    def calc_wavefuns(self):
        """
        construction of the scaled and translated wavelet functions
        """
        phi, psi, x = self.wave.wavefun(level=self.lvl)
        self.supp = x.max()
        phi = interp1d(x, phi, fill_value=0.0, bounds_error=False)
        psi = interp1d(x, psi, fill_value=0.0, bounds_error=False)
        phi1 = lambda y : phi(self.supp*y)*np.sqrt(self.supp)
        psi1 = lambda y : psi(self.supp*y)*np.sqrt(self.supp)

        phi2 = self.wt_tensor(phi1, phi1)
        phi2 = self.coord_trans(phi2, self.R)
        psi_a = self.wt_tensor(psi1, phi1)
        psi_a = self.coord_trans(psi_a, self.R)
        psi_b = self.wt_tensor(phi1, psi1)
        psi_b = self.coord_trans(psi_b, self.R)
        psi_c = self.wt_tensor(psi1, psi1)
        psi_c = self.coord_trans(psi_c, self.R)
        ##
        phi2 = partial(self.wt_scale_trans, phi2, self.R)
        psi_a = partial(self.wt_scale_trans, psi_a, self.R)
        psi_b = partial(self.wt_scale_trans, psi_b, self.R)
        psi_c = partial(self.wt_scale_trans, psi_c, self.R)

        self.wave_funs = [phi2,psi_a,psi_b,psi_c]
    
    @staticmethod
    def wt_tensor(f,g):
        return lambda t1,t2 : f(t1) * g(t2)

    @staticmethod
    def coord_trans(f,R):
        u = lambda t1,t2 : (t1+t2)/(np.sqrt(2)*R) + 1/2
        v = lambda t1,t2 : (t2-t1)/(np.sqrt(2)*R) 
        return lambda t1,t2 : f(u(t1,t2),v(t1,t2))/R
    
    @staticmethod
    def wt_scale_trans(f,R,j,m,n):
        return lambda t1,t2 : (2**j)*f((2**j)*t1+R*(m-n-1/2+2**(j-1))/np.sqrt(2),(2**j)*t2+R*(m+n-1/2+2**(j-1))/np.sqrt(2))
 
    def calc_coeffs(self, data):
        """
        compute wavelet coefficients from data
        """
        self.coeffs = []
        alpha_phi = lambda j,m,n : np.sum(self.wave_funs[0](j,m,n)(data[:,0],data[:,1])/self.N)
        beta_psi_a = lambda j,m,n : np.sum(self.wave_funs[1](j,m,n)(data[:,0],data[:,1])/self.N)
        beta_psi_b = lambda j,m,n : np.sum(self.wave_funs[2](j,m,n)(data[:,0],data[:,1])/self.N)
        beta_psi_c = lambda j,m,n : np.sum(self.wave_funs[3](j,m,n)(data[:,0],data[:,1])/self.N)
        
        betas_a = []
        betas_b = []
        betas_c = []
        for j in range(self.j0, self.J+1):
            n_start = 2**(j-self.k-1)
            n_end = 2**(j-self.k)
            m_start = 2**(j-1)
            m_end = 2**j

            betas_a.append(np.zeros([m_end-m_start,n_end-n_start]))
            betas_b.append(np.zeros([m_end-m_start,n_end-n_start]))
            betas_c.append(np.zeros([m_end-m_start,n_end-n_start]))
            if j == self.j0:
                alphas = np.zeros([m_end-m_start,n_end-n_start])
            Mm = (((data[:,0]+data[:,1])/(self.R*np.sqrt(2))+1/2)*2**(j)).astype(int)
            Nn = (((-data[:,0]+data[:,1])/(self.R*np.sqrt(2)))*2**(j)).astype(int)
            MN = np.unique(np.column_stack((Mm,Nn)),axis=0)
            for mn in MN:
                m = mn[0]
                n = mn[1]
                m_start2 = m_start + n
                m_end2 = m_end - n 
                if n>= n_start and n<n_end and m>=m_start2 and m<m_end2:
                    if j == self.j0:
                        alphas[m-m_start,n-n_start] = alpha_phi(self.j0,-m,-n)
                        self.number_coeff +=1
                    betas_a[j-self.j0][m-m_start,n-n_start] = beta_psi_a(j,-m,-n)
                    betas_b[j-self.j0][m-m_start,n-n_start] = beta_psi_b(j,-m,-n)
                    betas_c[j-self.j0][m-m_start,n-n_start] = beta_psi_c(j,-m,-n)
                    self.number_coeff +=3
                        
        self.coeffs = [alphas,betas_a,betas_b,betas_c]

    def calc_pdf(self):
        """
        compute estimated probability density function
        """
        def pdffun(X,Y):
            alphas = self.coeffs[0]
            betas_a = self.coeffs[1]
            betas_b = self.coeffs[2]
            betas_c = self.coeffs[3]
            phi = self.wave_funs[0]
            psi_a = self.wave_funs[1]
            psi_b = self.wave_funs[2]
            psi_c = self.wave_funs[3]

            res = 0

            for j in range(self.j0, self.J+1):
                n_start = 2**(j-self.k-1)
                n_end = 2**(j-self.k)
                #m_start = 0
                m_start = 2**(j-1)
                m_end = 2**j
                m_start2 = 2**(j-1)
                m_end2 = 2**j
                for n in range(n_start,n_end):
                    m_start2 = m_start + n
                    m_end2 = m_end - n
                    for m in range(m_start2,m_end2):
                        if j == self.j0:
                            alpha = alphas[m-m_start,n-n_start]
                            if np.abs(alpha) > self.thresholding(j):
                                res += alpha*phi(self.j0,-m,-n)(X,Y)
                                self.number_coeff_nonzero +=1
                        beta_a = betas_a[j-self.j0][m-m_start,n-n_start]
                        if np.abs(beta_a) > self.thresholding(j):
                            res += beta_a*psi_a(j,-m,-n)(X,Y)
                            self.number_coeff_nonzero +=1
                        beta_b = betas_b[j-self.j0][m-m_start,n-n_start]
                        if np.abs(beta_b) > self.thresholding(j):
                            res += beta_b*psi_b(j,-m,-n)(X,Y)
                            self.number_coeff_nonzero +=1
                        beta_c = betas_c[j-self.j0][m-m_start,n-n_start]
                        if np.abs(beta_c) > self.thresholding(j):
                            res += beta_c*psi_c(j,-m,-n)(X,Y)
                            self.number_coeff_nonzero +=1
            return res
        return pdffun
