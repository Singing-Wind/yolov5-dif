import numpy as np
from scipy.stats import norm

def a_k_half(k, reg_max_ft,ec=0.000001): #0~0.5
    p = 0.5 + k / (2 * reg_max_ft + ec)   # 0.5~7.5  k(15)
    return norm.ppf(p)

def compute_gauss_keys_half(reg_max_ft,key_scale=5.0,ec=0.000001):
    # 示例：k = 6，计算 a0 到 a4
    keys_ppf = [a_k_half(k, reg_max_ft,ec) for k in range(reg_max_ft)]
    # keys_ppf = keys_ppf + [2*keys_ppf[-1]]
    keys_ppf = np.asarray(keys_ppf) #keys_ppf[rgmax=16]
    keys_ppf = keys_ppf * key_scale # getattr(opt, 'key_scale', 4)

    return keys_ppf ##keys_ppf[rgmax=16]

def a_k(k, reg_max_ft, offset=0): #0~0.5
    p = 0.5 + (offset+k) / (reg_max_ft - 1)   # 0.5~7.5  k(15)
    return norm.ppf(p)

def compute_gauss_keys(reg_max_ft,key_sym=0,key_scale=5.0):
    # 示例：k = 6，计算 a0 到 a4
    if key_sym: #getattr(opt, 'key_sym', 0): #Symmetry
        keys_ppf = [a_k(k, reg_max_ft, 0.5) for k in range(reg_max_ft//2)][:-1] + [2.0]
        keys_ppf = [-1 * kppf for kppf in reversed(keys_ppf)] + keys_ppf
    else: #0 centered
        keys_ppf = [a_k(k, reg_max_ft, 0) for k in range(reg_max_ft//2)]
        keys_ppf = [-2*keys_ppf[-1]] + [-1 * kppf for kppf in reversed(keys_ppf[1:])] + keys_ppf

    keys_ppf = np.asarray(keys_ppf) #keys_ppf[rgmax=16]
    keys_ppf = keys_ppf * key_scale # getattr(opt, 'key_scale', 4)

    return keys_ppf ##keys_ppf[rgmax=16]

def compute_gauss_ft_keys(all_labels_ft,reg_max_ft,key_sym=0,key_scale=5.0):#all_labels_ft[ft_coef,4*nt]
    keys_ppf = compute_gauss_keys(reg_max_ft,key_sym=0,key_scale=5.0)#keys_ppf[rgmax=16]
    keys_ppf = keys_ppf.reshape(1, -1)#keys_ppf[rgmax]->keys_ppf[1,rgmax=16]
    cgm = np.std(all_labels_ft, axis=-1) #cgm[ft_coef]
    cgm = cgm / cgm[0] #cgm[n]
    cgm = np.repeat(cgm.reshape(-1, 1), 4, axis=-1).reshape(-1,1) #cgm[4n,1]
    
    ft_keys = cgm * keys_ppf #cgm[4n,1]*keys_ppf[1,rgmax]->ft_keys[4n,rgmax]
    return ft_keys #ft_keys[4n,rgmax]

def compute_ft_keys(all_labels_ft, rgmax):
    # all_labels_ft: 2D array of shape [ft_coef, nt*4]
    ft_coef, dim = all_labels_ft.shape
    ft_keys = np.zeros((ft_coef, rgmax))

    # quantile steps
    quantiles = np.linspace(0, 1, rgmax)

    for i in range(ft_coef):
        row = all_labels_ft[i]
        ft_keys[i] = np.quantile(row, quantiles)

    return ft_keys #ft_keys[rgmax]