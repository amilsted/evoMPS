# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 11:15:35 2015

@author: ash
"""
cimport numpy as np

ctypedef np.ndarray ndcmp2d
ctypedef np.ndarray ndcmp3d

cpdef np.ndarray eps_l_noop(x, ndcmp3d A1, ndcmp3d A2)

cpdef np.ndarray eps_l_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout)

cpdef np.ndarray eps_r_noop(x, ndcmp3d A1, ndcmp3d A2)
        
cpdef np.ndarray eps_r_noop_inplace(x, ndcmp3d A1, ndcmp3d A2, ndcmp2d ndout)
 